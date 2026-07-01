import json
import jsonlines
import os
import re
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from typing import Dict, Set, Tuple, List
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--res_path", type=str, default="", required=True)
args = parser.parse_args()
def extract_file(final_response):
    """Extract file paths from final response"""
    file_paths = set()
    locations_pattern = r'<locations_to_modify>(.*?)</locations_to_modify>'
    locations_matches = re.findall(locations_pattern, final_response, re.DOTALL)
    context_pattern = r'<related_context>(.*?)</related_context>'
    context_matches = re.findall(context_pattern, final_response, re.DOTALL)
    all_content = '\n'.join(locations_matches + context_matches)
    for content in all_content.split("\n"):
        if content.strip() != "":
            if ":" in content:
                file_path = content.split(":")[0]
                file_paths.add(file_path)
            else:
                file_paths.add(content)
    return file_paths

def extract_entities_from_tool(tool_name: str, args: dict, output: str) -> Tuple[Set[str], Set[Tuple[str, int]]]:
    """
    Extract entities from tool output
    
    Returns:
        (files, content_lines)
        - files: Set of file paths
        - content_lines: Set of (file, line_number) tuples
    """
    files = set()
    content_lines = set()
    
    if tool_name == "glob":
        # glob: return file list
        for line in output.split('\n'):
            line = line.strip()
            if line:
                files.add(line)
    
    elif tool_name == "grep":
        output_mode = args.get('output_mode', 'files_with_matches')
        
        if output_mode == 'files_with_matches':
            # files_with_matches: "Found X file(s):\n/path1\n/path2"
            lines = output.split('\n')
            for line in lines:
                line = line.strip()
                if line and not line.startswith('Found ') and not line.endswith('file:') and not line.endswith('files:'):
                    files.add(line)
        
        elif output_mode == 'content':
            # content: "/path/file.py:472:content"
            pattern = r'^(.+?):(\d+):'
            for line in output.split('\n'):
                match = re.match(pattern, line)
                if match:
                    file_path = match.group(1)
                    line_num = int(match.group(2))
                    files.add(file_path)
                    content_lines.add((file_path, line_num))
    
    elif tool_name == "read_file":
        # read_file: extract from parameters
        file_path = args.get('path', '')
        if file_path:
            files.add(file_path)
            
            # Extract line number ranges
            if 'start_line' in args:
                start = args.get('start_line', 1)
                end = args.get('end_line', start)
                for line_num in range(start, end + 1):
                    content_lines.add((file_path, line_num))
            elif 'offset' in args:
                offset = args.get('offset', 1)
                limit = args.get('limit')
                if limit:
                    for line_num in range(offset, offset + limit):
                        content_lines.add((file_path, line_num))
                else:
                    # Read entire file: use special marker
                    content_lines.add((file_path, -1))
            else:
                # No line number specified, read entire file
                content_lines.add((file_path, -1))
    
    return files, content_lines

def calculate_tool_efficiency(tool_name: str, args: dict, output: str,
                              file_history: Set[str], 
                              content_history: Set[Tuple[str, int]]) -> Tuple[float, int, int]:
    r"""
    Calculate tool efficiency g = |E_i \ H| / |E_i|
    
    Returns:
        (efficiency, new_entities, total_entities)
    """
    # Check for valid output
    if not output or output in ["No matches found", "No files found"] or output.startswith("Error:"):
        return 0.0, 0, 0
    
    # Special handling: read_file check for newline characters
    if tool_name == "read_file" and '\n' not in output:
        return 0.0, 0, 0
    
    try:
        # Extract entities returned by current tool
        new_files, new_content = extract_entities_from_tool(tool_name, args, output)
        
        # Calculate new entities
        # Strategy: prioritize content-level entities, fallback to file-level if none
        if new_content:
            # Use content-level (more granular)
            total_entities = len(new_content)
            
            # Handle full file reading case
            if (-1,) in [line for _, line in new_content]:
                # If reading entire file and previously read any content from this file
                file_with_full_read = [f for f, l in new_content if l == -1][0]
                existing_lines = [l for f, l in content_history if f == file_with_full_read and l != -1]
                if existing_lines:
                    # Previously read partial, now read full, conservatively estimate 10% as new
                    new_entities = max(1, total_entities // 10)
                else:
                    # Never read this file before
                    new_entities = total_entities
            else:
                # Normal case: calculate new (file, line_number) pairs
                new_entities_set = new_content - content_history
                new_entities = len(new_entities_set)
        else:
            # Only file-level entities (glob, grep files_with_matches)
            total_entities = len(new_files)
            new_entities_set = new_files - file_history
            new_entities = len(new_entities_set)
        
        # Calculate efficiency
        if total_entities == 0:
            efficiency = 0.0
        else:
            efficiency = new_entities / total_entities
        
        return efficiency, new_entities, total_entities
    
    except Exception as e:
        return 0.0, 0, 0

def update_history(new_files: Set[str], new_content: Set[Tuple[str, int]],
                   file_history: Set[str], content_history: Set[Tuple[str, int]]):
    """Update history records"""
    file_history.update(new_files)
    content_history.update(new_content)
    
def process_single_message(llm_message, data_path):
    """Process a single trajectory file"""
    if llm_message in ["success.json", "error.json", "timing.json"]:
        return None, None
    
    try:
        instance_id = llm_message.replace("llm_messages_", "").replace(".json", "")
        
        # Read messages
        with open(f"{data_path}/{llm_message}", "r") as f:
            messages = json.load(f)
        
        # Build data structure
        id2input = {}
        id2output = {}
        id2turn = {}
        tool_order = []
        turn = 0
        final_response = None
        
        for message in messages:
            if message['role'] == 'assistant':
                turn += 1
                if "tool_calls" in message.keys() and len(message["tool_calls"]) > 0:
                    for tool_call in message["tool_calls"]:
                        tool_id = tool_call['id']
                        id2input[tool_id] = tool_call['function']
                        id2turn[tool_id] = turn
                        tool_order.append(tool_id)
                else:
                    final_response = message['content']
            elif message['role'] == 'tool':
                id2output[message['tool_call_id']] = message['content']
        
        if len(id2input) == 0 or final_response is None:
            return instance_id, None
        
        # Initialize history
        file_history = set()
        content_history = set()
        
        # Statistics data
        tool_efficiencies = []  # Efficiency for each tool
        turn_stats = defaultdict(lambda: {
            'total_tools': 0,
            'total_efficiency': 0.0,
            'efficiencies': []
        })
        
        # Process each tool in order
        for tool_id in tool_order:
            func_info = id2input[tool_id]
            tool_name = func_info['name']
            turn_num = id2turn[tool_id]
            output = id2output.get(tool_id, "")
            
            try:
                args = json.loads(func_info['arguments'])
            except:
                args = {}
            
            # Calculate efficiency
            efficiency, new_entities, total_entities = calculate_tool_efficiency(
                tool_name, args, output, file_history, content_history
            )
            
            tool_efficiencies.append(efficiency)
            turn_stats[turn_num]['total_tools'] += 1
            turn_stats[turn_num]['total_efficiency'] += efficiency
            turn_stats[turn_num]['efficiencies'].append(efficiency)
            
            # Update history
            if efficiency > 0:
                new_files, new_content = extract_entities_from_tool(tool_name, args, output)
                update_history(new_files, new_content, file_history, content_history)
        
        # Calculate average efficiency
        overall_efficiency = np.mean(tool_efficiencies) if tool_efficiencies else 0.0
        
        # Calculate exploration space size
        exploration_size = {
            'files': len(file_history),
            'content_lines': len([c for c in content_history if c[1] != -1]),
            'full_files': len([c for c in content_history if c[1] == -1])
        }
        
        return instance_id, {
            'overall_efficiency': overall_efficiency,
            'tool_efficiencies': tool_efficiencies,
            'turn_stats': dict(turn_stats),
            'exploration_size': exploration_size,
            'total_tools': len(tool_efficiencies)
        }
    
    except Exception as e:
        print(f"❌ Processing {llm_message} failed: {e}")
        return instance_id, None

def calculate_efficiency_parallel(data_path, llm_messages, max_workers=8, output_file=None):
    """Calculate tool efficiency in parallel"""
    all_results = []
    efficiency_records = []
    
    # Global statistics
    global_efficiencies = []
    global_turn_stats = defaultdict(lambda: {
        'total_tools': 0,
        'total_efficiency': 0.0,
        'efficiencies': []
    })
    
    # Process in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_single_message, msg, data_path): msg
            for msg in llm_messages
        }
        
        for future in as_completed(futures):
            msg = futures[future]
            try:
                instance_id, result = future.result()
                if result:
                    # Aggregate global statistics
                    global_efficiencies.extend(result['tool_efficiencies'])
                    
                    # Aggregate per-turn statistics
                    for turn, tstats in result['turn_stats'].items():
                        global_turn_stats[turn]['total_tools'] += tstats['total_tools']
                        global_turn_stats[turn]['total_efficiency'] += tstats['total_efficiency']
                        global_turn_stats[turn]['efficiencies'].extend(tstats['efficiencies'])
                    
                    # Record for saving
                    efficiency_records.append({
                        'instance_id': instance_id,
                        'overall_efficiency': result['overall_efficiency'],
                        'total_tools': result['total_tools'],
                        'exploration_files': result['exploration_size']['files'],
                        'exploration_lines': result['exploration_size']['content_lines']
                    })
                    
                    all_results.append(result)
            except Exception as e:
                print(f"❌ {msg} execution failed: {e}")
    
    # Calculate statistics
    overall_mean_efficiency = np.mean(global_efficiencies) if global_efficiencies else 0.0
    overall_std_efficiency = np.std(global_efficiencies) if global_efficiencies else 0.0
    
    # Print results
    print(f"\n{'='*100}")
    print(f"🔍 Tool Efficiency Analysis (V3 - Continuous Efficiency Values)")
    print(f"{'='*100}")
    print(f"Total tool calls: {len(global_efficiencies)}")
    print(f"Total trajectories: {len(all_results)}")
    print(f"\n{'='*100}")
    print(f"📊 Overall Efficiency Statistics")
    print(f"{'='*100}")
    print(f"Average efficiency (e): {overall_mean_efficiency:.4f}")
    print(f"Standard deviation:     {overall_std_efficiency:.4f}")
    print(f"Median:                 {np.median(global_efficiencies):.4f}")
    
    # Efficiency distribution
    bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    hist, _ = np.histogram(global_efficiencies, bins=bins)
    
    print(f"\nEfficiency distribution:")
    print(f"  [0.0-0.2):  {hist[0]:>6} ({hist[0]/len(global_efficiencies)*100:>5.1f}%)  {'█' * (hist[0] // 50)}")
    print(f"  [0.2-0.4):  {hist[1]:>6} ({hist[1]/len(global_efficiencies)*100:>5.1f}%)  {'█' * (hist[1] // 50)}")
    print(f"  [0.4-0.6):  {hist[2]:>6} ({hist[2]/len(global_efficiencies)*100:>5.1f}%)  {'█' * (hist[2] // 50)}")
    print(f"  [0.6-0.8):  {hist[3]:>6} ({hist[3]/len(global_efficiencies)*100:>5.1f}%)  {'█' * (hist[3] // 50)}")
    print(f"  [0.8-1.0]:  {hist[4]:>6} ({hist[4]/len(global_efficiencies)*100:>5.1f}%)  {'█' * (hist[4] // 50)}")
    
    # Per-turn efficiency statistics
    print(f"\n{'='*100}")
    print(f"📈 Per-turn Efficiency Statistics")
    print(f"{'='*100}")
    print(f"{'Turn':<6} {'Tools':>8} {'Avg Eff':>12} {'Median':>12} {'Std Dev':>12}")
    print(f"{'-'*100}")
    
    for turn in sorted(global_turn_stats.keys()):
        tstats = global_turn_stats[turn]
        mean_eff = np.mean(tstats['efficiencies']) if tstats['efficiencies'] else 0.0
        median_eff = np.median(tstats['efficiencies']) if tstats['efficiencies'] else 0.0
        std_eff = np.std(tstats['efficiencies']) if tstats['efficiencies'] else 0.0
        
        print(f"{turn:<6} {tstats['total_tools']:>8} {mean_eff:>12.4f} {median_eff:>12.4f} {std_eff:>12.4f}")
    
    print(f"{'='*100}\n")
    
    # Save records
    if output_file and efficiency_records:
        with open(output_file, 'w') as f:
            for record in efficiency_records:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        print(f"✅ Efficiency records saved to: {output_file}")
        print(f"   Total {len(efficiency_records)} trajectories\n")
    
    return {
        'overall_efficiency': overall_mean_efficiency,
        'overall_std': overall_std_efficiency,
        'turn_stats': dict(global_turn_stats),
        'distribution': hist
    }

# Main program
if __name__ == "__main__":
    llm_messages = [f for f in os.listdir(args.res_path) if f.startswith('llm_messages_') and f.endswith('.json')]
    
    # Set output file path
    output_file = os.path.join(args.res_path, "efficiency.jsonl")
    
    # Execute analysis
    stats = calculate_efficiency_parallel(
        args.res_path, llm_messages,
        max_workers=8,
        output_file=output_file
    )
