TOOLS_DESCRIPTION = [
  {
    "type": "function",
    "function": {
      "name": "read_file",
      "description": "Reads a file from the local filesystem. You can access any file directly by using this tool.\n\nImportant:\n- The path parameter MUST be an absolute path, not a relative path\n- By default, it reads up to 1000 lines starting from the beginning of the file\n- You can optionally specify a line offset and limit (especially handy for long files), but it's recommended to read the whole file by not providing these parameters\n- Results are returned with line numbers starting at 1\n",
      "parameters": {
        "type": "object",
        "properties": {
          "path": {
            "type": "string",
            "description": "Absolute path to the file to read"
          },
          "start_line": {
            "type": "number",
            "description": "Starting line number"
          },
          "end_line": {
            "type": "number",
            "description": "Ending line number"
          }
        },
        "required": [
          "path"
        ]
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "grep",
      "description": "Search file contents using regex patterns. Built on ripgrep for high performance.\n\n- Best for exact symbol/string searches across the codebase. Prefer this over terminal grep/rg.\n- Supports full regex: \"log.*Error\", \"function\\s+\\w+\". Escape special chars for exact matches: \"functionCall\\(\\)\".\n- Pattern syntax follows ripgrep - literal braces need escaping: interface\\{\\} to match interface{}.\n\n- Output Modes:\nfiles_with_matches - shows only file paths (default)\ncontent - shows matching lines with context (auto-enables line numbers)\ncount - shows match counts per file\n\n- Notes:\nUse 'type' or 'glob' parameters only when certain of file type needed.\nAvoid overly broad patterns like '--glob *' which may bypass filters and slow down search.\nSupports parallel calls with different patterns(Recommended to use this tool in a batch of patterns to find files that are potentially useful).",
      "parameters": {
        "type": "object",
        "properties": {
          "pattern": {
            "type": "string",
            "description": "The regular expression pattern to search for in file contents (rg --regexp)"
          },
          "path": {
            "type": "string",
            "description": "File or directory to search in (rg pattern -- PATH). Defaults to Cursor workspace roots."
          },
          "glob": {
            "type": "string",
            "description": "Glob pattern (rg --glob GLOB -- PATH) to filter files (e.g. \"*.js\", \"*.{ts,tsx}\")."
          },
          "output_mode": {
            "type": "string",
            "description": "Output mode: \"content\" shows matching lines (supports -A/-B/-C context, -n line numbers, head_limit), \"files_with_matches\" shows file paths (supports head_limit), \"count\" shows match counts (supports head_limit). Defaults to \"files_with_matches\".",
            "enum": [
              "content",
              "files_with_matches",
              "count"
            ]
          },
          "-B": {
            "type": "number",
            "description": "Number of lines to show before each match (rg -B). Requires output_mode: \"content\", ignored otherwise."
          },
          "-A": {
            "type": "number",
            "description": "Number of lines to show after each match (rg -A). Requires output_mode: \"content\", ignored otherwise."
          },
          "-C": {
            "type": "number",
            "description": "Number of lines to show before and after each match (rg -C). Requires output_mode: \"content\", ignored otherwise."
          },
          "-n": {
            "type": "boolean",
            "description": "Show line numbers in output (rg -n). Defaults to true. Requires output_mode: \"content\", ignored otherwise."
          },
          "-i": {
            "type": "boolean",
            "description": "Case insensitive search (rg -i) Defaults to false"
          },
          "type": {
            "type": "string",
            "description": "File type to search (rg --type). Common types: js, py, rust, go, java, etc. More efficient than glob for standard file types."
          },
          "head_limit": {
            "type": "number",
            "description": "Limit output to first N lines/entries, equivalent to \"| head -N\". Works across all output modes: content (limits output lines), files_with_matches (limits file paths), count (limits count entries). When unspecified, shows all ripgrep results."
          },
          "multiline": {
            "type": "boolean",
            "description": "Enable multiline mode where . matches newlines and patterns can span lines (rg -U --multiline-dotall). Default: false."
          }
        },
        "required": [
          "pattern"
        ]
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "glob",
      "description": "Find files by name patterns using glob syntax. Handles codebases of any size efficiently.\n\n- Accepts standard glob patterns such as:\n*.py - match files in current directory\n**/*.js - search all subdirectories recursively\nsrc/**/*.ts - limit search to specific path\ntest_*.py - match files with prefix\n\n- Notes:\nLimits results to 100 file paths.\nSupports parallel calls with different patterns(Recommended to use this tool in a batch of patterns to find files that are potentially useful).",
      "parameters": {
        "type": "object",
        "properties": {
          "pattern": {
            "type": "string",
            "description": "The glob pattern to match files against"
          },
          "path": {
            "type": "string",
            "description": "The directory to search in. If not specified, the workspace root will be used. IMPORTANT: Omit this field to use the default directory. Must be an absolute path if provided."
          }
        },
        "required": [
          "pattern"
        ]
      }
    }
  }
]