import argparse
from typing import Optional
import json
import numpy as np

def load_jsonl(file_path):
    with open(file_path, 'r') as file:
        data = [json.loads(line) for line in file]
    return data

def load_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

# Filtered instances list
filtered_instances = []

def compute_prf(gt_dict, pred_dict, selected_list=None):
    """
    Compute overall Precision, Recall, F1 without Top-k limitation.
    Considers all predictions made by the model.
    This computes MICRO average (aggregate TP/FP/FN across all instances).
    """
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    valid_instances = []
    pred_counts = []
    gt_counts = []
    
    for instance_id in gt_dict.keys():
        if selected_list and instance_id not in selected_list:
            continue
        if not gt_dict[instance_id]:
            continue
        
        valid_instances.append(instance_id)
        
        gt_locs = set(gt_dict[instance_id])
        pred_locs = set(pred_dict.get(instance_id, []))
        
        # Count predictions and GT for each case
        pred_counts.append(len(pred_locs))
        gt_counts.append(len(gt_locs))
        
        # Calculate TP, FP, FN for this instance
        tp = len(gt_locs & pred_locs)  # Intersection
        fp = len(pred_locs - gt_locs)  # Predicted but not in GT
        fn = len(gt_locs - pred_locs)  # In GT but not predicted
        
        total_tp += tp
        total_fp += fp
        total_fn += fn
    
    # Calculate overall metrics (MICRO average)
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Calculate average counts
    avg_pred = sum(pred_counts) / len(pred_counts) if pred_counts else 0.0
    avg_gt = sum(gt_counts) / len(gt_counts) if gt_counts else 0.0
    
    return {
        'Precision': round(precision, 4),
        'Recall': round(recall, 4),
        'F1': round(f1, 4),
        'avg_pred': round(avg_pred, 2),
        'avg_gt': round(avg_gt, 2)
    }


def parse_gt_location(loc_str, level):
    """
    Parse GT location string to extract file/module/function level info.
    GT format examples:
    - File: "django/conf/global_settings.py"
    - Function: "django/db/models/fields/__init__.py:Field.formfield"
    """
    if level == 'file':
        # For file level, return the file path (before : or ::)
        if '::' in loc_str:
            return loc_str.split('::')[0]
        elif ':' in loc_str:
            return loc_str.split(':')[0]
        return loc_str
    elif level == 'function':
        return loc_str


def parse_pred_location(loc_str, level):
    """
    Parse prediction location string to extract file/module/function level info.
    Pred format: "astropy/timeseries/core.py:BaseTimeSeries._check_required_columns"
    """
    if level == 'file':
        # For file level, return the file path (before first colon)
        if ':' in loc_str:
            return loc_str.split(':')[0]
        return loc_str
    elif level == 'function':
        # For function level, return the full string
        return loc_str

def load_gt_dict(gt_data, level):
    """
    Load GT dictionary from JSONL format.
    gt_data: list of dicts with 'instance_id' and 'localization_gt' keys
    """
    gt_dict = {}
    for item in gt_data:
        instance_id = item['instance_id']
        if level == "file":
            level_key = "files"
        elif level == "function":
            level_key = "functions"
        else:
            raise ValueError(f"Unknown level: {level}")
        localization_gt = item.get('localization_gt') or {}
        gt_dict[instance_id] = localization_gt.get(level_key, [])
    return gt_dict


def load_pred_dict(pred_data, level):
    """
    Load prediction dictionary from JSONL format.
    pred_data: list of dicts with 'instance_id' and 'localization_pred' keys
    """
    pred_dict = {}
    for item in pred_data:
        instance_id = item['instance_id']
        if level == "file":
            level_key = "found_files"
        elif level == "function":
            level_key = "found_functions"
        else:
            raise ValueError(f"Unknown level: {level}")
        # Return empty list if field doesn't exist
        pred_dict[instance_id] = item.get(level_key, [])
    return pred_dict

def eval_w_file(pred_file, gt_path="./data/swe-bench_verified.jsonl", selected_list=None, pred_only=False):
    """
    Evaluate PRF results at file, function, and line levels.
    Also compute agent statistics.
    """
    # Load data from JSONL files
    gt_data = load_jsonl(gt_path)
    pred_data = load_jsonl(pred_file)
    if pred_only:
        pred_ids = {item["instance_id"] for item in pred_data}
        selected_list = pred_ids if selected_list is None else set(selected_list) & pred_ids
    
    print(f"Ground truth instances: {len(gt_data)}")
    print(f"Prediction instances: {len(pred_data)}")
    
    # Calculate PRF for file level (Micro)
    file_gt_dict = load_gt_dict(gt_data, level='file')
    file_pred_dict = load_pred_dict(pred_data, level='file')
    file_prf = compute_prf(file_gt_dict, file_pred_dict, selected_list=selected_list)
    
    # Calculate PRF for function level (Micro)
    function_gt_dict = load_gt_dict(gt_data, level='function')
    function_pred_dict = load_pred_dict(pred_data, level='function')
    function_prf = compute_prf(function_gt_dict, function_pred_dict, selected_list=selected_list)

    # PRF results
    prf_results = {
        'file': file_prf,
        'function': function_prf,
    }
    
    return prf_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pred_file", "-p",
        type=str,
        required=True,
        help="Path to prediction file (json format)",
    )
    parser.add_argument(
        "--gt_path", "-g",
        type=str,
        default="./data/swe-bench_verified.jsonl",
        help="Path to ground-truth JSONL file",
    )
    parser.add_argument(
        "--pred_only",
        action="store_true",
        help="Evaluate only instances that appear in the prediction file",
    )

    args = parser.parse_args()
    
    prf_results = eval_w_file(args.pred_file, gt_path=args.gt_path, pred_only=args.pred_only)
    
    # Format output nicely
    print("\n" + "="*80)
    print("PRF Evaluation Results".center(80))
    print("="*80)
    
    # Print file level PRF
    print("\n📁 File Level PRF:")
    print("-" * 80)
    print(f" Precision: {prf_results['file']['Precision']:.4f}")
    print(f" Recall:    {prf_results['file']['Recall']:.4f}")
    print(f" F1 Score:  {prf_results['file']['F1']:.4f}")
    
    # Print function level PRF
    print("\n🔧 Function Level PRF:")
    print("-" * 80)
    print(f" Precision: {prf_results['function']['Precision']:.4f}")
    print(f" Recall:    {prf_results['function']['Recall']:.4f}")
    print(f" F1 Score:  {prf_results['function']['F1']:.4f}")

    
