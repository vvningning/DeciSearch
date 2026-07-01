import json
import os
import re
import argparse
parser = argparse.ArgumentParser(description='')
parser.add_argument(
    '--res_path',
    type=str,
    default="",
    help='Path to the results directory',
    required=True
)
parser.add_argument(
    '--output_path',
    type=str,
    default="./parsed_localizations.json",
    help='Path to the output JSON file'
)
parser.add_argument(
    '--worktree_base',
    type=str,
    default="",
    help='Worktree directory path (absolute path required)',
    required=True
)
args = parser.parse_args()

def extract(final_response, workdir):
    files = set()
    functions = []
    locations_pattern = r'<locations_to_modify>(.*?)</locations_to_modify>'
    locations_match = re.search(locations_pattern, final_response, re.DOTALL)
    if locations_match:
        content = locations_match.group(1)
        content = content.strip()
        for line in content.split('\n'):
            text = line.replace(workdir, "")
            text_new = text.replace("::", "<SpecialToken>")
            if ":" in text_new:
                parts = text_new.split(":")
                if len(parts) == 2:
                    files.add(parts[0])
                    functions.append(text.replace("<SpecialToken>", "::"))
                else:
                    print(f"Parser Error: {line}")
            else:
                files.add(text)
    return list(files), functions

res = []
llm_messages = os.listdir(args.res_path)
for llm_message in llm_messages:
    if "llm_messages_" not in llm_message:
        continue
    instance_id = llm_message.replace("llm_messages_", "").replace(".json", "")
    with open(f"{args.res_path}/{llm_message}", "r") as f:
        messages = json.load(f)
        for message in reversed(messages):
            if message['role'] == 'assistant':
                if message['content'] is not None:
                    files, functions = extract(message['content'], f"{args.worktree_base}/{instance_id}/")
                    res.append({"instance_id": instance_id, "found_files": files, "found_functions": functions})
                else:
                    print("Error:", instance_id)
                break
with open(args.output_path, 'w', encoding='utf-8') as f:
    for item in res:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')