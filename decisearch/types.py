"""Runtime state for the DeciSearch workflow."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any


def clean_text(value: Any, max_chars: int = 500) -> str:
    text = "" if value is None else str(value)
    text = " ".join(text.strip().split())
    return text[:max_chars]


def split_location(location: str) -> tuple[str, str]:
    text = (location or "").strip()
    if "::" in text:
        file_part, func = text.split("::", 1)
    elif ":" in text:
        file_part, func = text.split(":", 1)
    else:
        file_part, func = text, ""
    return file_part.strip(), func.strip()


def normalize_path(path: str, workspace_root: Path) -> str:
    raw = (path or "").strip()
    if not raw:
        return ""
    raw_path = Path(raw).expanduser()
    if not raw_path.is_absolute():
        raw_path = workspace_root / raw_path
    try:
        return str(raw_path.resolve())
    except OSError:
        return str(raw_path)


def normalize_location(location: str, workspace_root: Path) -> str:
    file_part, func = split_location(location)
    file_path = normalize_path(file_part, workspace_root)
    return f"{file_path}:{func}" if func else file_path


@dataclass
class EvidenceItem:
    file: str
    function: str = ""
    location: str = ""
    mode: str = ""
    evidence: str = ""
    source_tool: str = ""
    support: str = "new"
    confidence: float = 0.0
    worker_id: str = ""

    def normalized_location(self) -> str:
        if self.location:
            return self.location
        if self.file and self.function:
            return f"{self.file}:{self.function}"
        return self.file

    def key(self) -> str:
        return self.normalized_location()

    def to_dict(self) -> dict[str, Any]:
        return {
            "location": self.normalized_location(),
            "file": self.file,
            "function": self.function,
            "mode": self.mode,
            "evidence": clean_text(self.evidence, 600),
            "source_tool": self.source_tool,
            "support": self.support,
            "confidence": round(float(self.confidence or 0.0), 3),
            "worker_id": self.worker_id,
        }


@dataclass
class EvidenceBoard:
    workspace_root: Path
    items: list[EvidenceItem] = field(default_factory=list)
    _index: dict[str, int] = field(default_factory=dict)

    def add(self, item: EvidenceItem) -> bool:
        item.location = normalize_location(item.normalized_location(), self.workspace_root)
        item.file, item.function = split_location(item.location)
        key = item.key()
        if not key:
            return False

        existing_idx = self._index.get(key)
        if existing_idx is not None:
            existing = self.items[existing_idx]
            if item.confidence > existing.confidence:
                existing.confidence = item.confidence
            if item.support != "new" and existing.support in {"new", "weak"}:
                existing.support = item.support
            if item.evidence and item.evidence not in existing.evidence:
                existing.evidence = clean_text(f"{existing.evidence} {item.evidence}", 800)
            return False

        self._index[key] = len(self.items)
        self.items.append(item)
        return True

    def add_many(self, items: list[EvidenceItem]) -> int:
        return sum(1 for item in items if self.add(item))

    def update_support(self, location: str, support: str, confidence: float | None = None, reason: str = "") -> None:
        key = normalize_location(location, self.workspace_root)
        idx = self._index.get(key)
        if idx is None:
            file_path, func = split_location(key)
            self.add(
                EvidenceItem(
                    file=file_path,
                    function=func,
                    location=key,
                    support=support,
                    confidence=confidence or 0.0,
                    evidence=reason,
                    source_tool="verifier",
                )
            )
            return

        item = self.items[idx]
        item.support = support or item.support
        if confidence is not None:
            item.confidence = confidence
        if reason:
            item.evidence = clean_text(f"{item.evidence} verifier: {reason}", 800)

    def top_items(self, limit: int = 12, include_rejected: bool = False) -> list[EvidenceItem]:
        items = self.items if include_rejected else [x for x in self.items if x.support != "rejected"]
        support_rank = {"verified": 4, "weak": 3, "new": 2, "duplicate": 1, "rejected": 0}
        return sorted(
            items,
            key=lambda x: (support_rank.get(x.support, 0), x.confidence, bool(x.function)),
            reverse=True,
        )[:limit]

    def candidate_locations(self, limit: int = 12) -> list[str]:
        return [item.normalized_location() for item in self.top_items(limit=limit)]

    def compact_json(self, limit: int = 12) -> str:
        return json.dumps([item.to_dict() for item in self.top_items(limit=limit)], ensure_ascii=False, indent=2)

    def to_xml(self, max_locations: int = 8, max_context: int = 8) -> str:
        primary = self.top_items(limit=max_locations)
        primary_keys = {item.key() for item in primary}
        context = [item for item in self.top_items(limit=max_locations + max_context) if item.key() not in primary_keys]
        loc_lines = "\n".join(item.normalized_location() for item in primary)
        ctx_lines = "\n".join(item.normalized_location() for item in context[:max_context])
        return (
            "<locations_to_modify>\n"
            f"{loc_lines}\n"
            "</locations_to_modify>\n\n"
            "<related_context>\n"
            f"{ctx_lines}\n"
            "</related_context>"
        )
