"""
Reads the per-example MAPS score array saved by identify_heads.py 
and applies the p% threshold (the Per-Prompt-Style criterion) to 
produce the final lexical task head list for one (prompt_type, template) run.
"""
import os
import argparse
import pickle
import json
import numpy as np
from collections import defaultdict


def maps_pickle_path(save_root, model_name, component_type, d_name, prompt_type, prompt_template_index, correct_or_incorrect="correct"):
    save_path = os.path.join(save_root, model_name, d_name,
                    "Heads", "MAPS_nnsight", f"{component_type}_across_tasks",)
    return os.path.join(
        save_path, 
            f"{d_name}_MAPS_{component_type}_heads_across_tasks_{prompt_type}_{prompt_template_index}_{correct_or_incorrect}.pkl",
    )

def load_maps_scores(pkl_path: str) -> np.ndarray:
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


# def threshold_heads(maps_scores: np.ndarray, threshold: float):
#     """
#     Args:
#         maps_scores: (n_examples, n_layers, n_heads), binary per-prompt scores
#         threshold: p — min fraction of prompts a head must match
#     Returns:
#         head_list: sorted list of (layer, head) tuples
#         fraction_matched: (n_layers, n_heads) array — the actual match rate per head
#     """
#     fraction_matched = maps_scores.mean(axis=0)
#     indices = np.argwhere(fraction_matched >= threshold)
#     head_list = sorted((int(layer), int(head)) for layer, head in indices)
#     return head_list, fraction_matched

def threshold_heads(maps_scores: dict, n_match_list: list, task_list: list, threshold: float):
    """
    Args:
        maps_scores: {n_match: {task: (n_examples, n_layers, n_heads) ndarray}}
        n_match: which n_match key to select
        task_list: list of tasks
        threshold: p -- min fraction of prompts a head must match
    Returns:
        head_list: sorted list of (layer, head) tuples
        fraction_matched: (n_layers, n_heads) array -- the actual match rate per head
    """
    for (task, n_match) in zip(task_list, n_match_list):
        scores = maps_scores[n_match][task]   # (n_examples, n_layers, n_heads)
        fraction_matched = scores.mean(axis=0)
        indices = np.argwhere(fraction_matched >= threshold)
        head_list = sorted((int(layer), int(head)) for layer, head in indices)
    return head_list, fraction_matched

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, 
            help="model name e.g. meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--d_name", type=str, required=True,)
    parser.add_argument("--prompt_type", type=str, required=True, help="prompt type: EP or IP")
    parser.add_argument("--component_type", type=str, required=True, 
        help="component type: lexical_task or Retrieval heads")
    parser.add_argument("--monitor_other_task", type=bool, default=False, 
        help="whether to monitor other task's lexical task heads while running the prompts for a giventarget task")
    parser.add_argument("--save_root", type=str, 
        # default="../output",)
        default="output_1",)
    parser.add_argument("--project_root", type=str, 
        # default="../",
        default="",
        help="directory of the codebase ")
    parser.add_argument("--batch_size", type=int, default=20, help="batch size")
    parser.add_argument("--k", type=int, default=10, help="top k decoded tokens to look for match")
    parser.add_argument("--exp_size", type=int, default=100, help="number of examples to sample from the dataset")
    # parser.add_argument("--dataset_folder", type=str, default="../datasets/abstractive", help="folder of the dataset")
    parser.add_argument("--dataset_folder", type=str, default="datasets/abstractive", help="folder of the dataset")
    parser.add_argument("--remote", type=bool, default=False, help="whether to use NDIF to run model remotely")
    parser.add_argument("--task_relation_dict_type", type=str, default="human",
        help="type of the task relation dict: human, llms, combined")
    parser.add_argument("--threshold", "-p", type=float, default=0.1,
    help="p: minimum fraction of prompts a head must match to count as a lexical task head")

    args = parser.parse_args()

    if args.monitor_other_task:
        task_list = args.task_relation_dict.keys()
    # elif dataset_folder == "../datasets/compositional":
    elif args.dataset_folder == "datasets/compositional":
        # load composition task list 
        with open(os.path.join(args.project_root, "datasets", 
            "dataset_info", f"compositional_task_dict.json"), "r") as f:
            compositional_task_dict = json.load(f)
        task_list = compositional_task_dict[args.d_name]
        assert type(task_list) == list
    else:
        task_list = [args.d_name]

    if args.component_type == "lexical_task":
        component_type = "Relation"

    prompt_template_indices = [5, 10, 20, 30]
    lexical_dict = defaultdict(list)
    model_short_name = args.model_name.split("/")[-1]
    for prompt_template_index in prompt_template_indices:
        pkl_path = maps_pickle_path(
            args.save_root, model_short_name, 
            component_type, args.d_name,
            args.prompt_type, prompt_template_index)
        maps_scores = load_maps_scores(pkl_path)
        # head_list, fraction_matched = threshold_heads(maps_scores, args.threshold)
        head_list, fraction_matched = threshold_heads(maps_scores, n_match_list=[1], task_list=task_list, threshold = args.threshold)
        print(f"{args.prompt_type} heads (p={args.threshold}): {len(head_list)} -> {head_list}")
        lexical_dict[f"{args.prompt_type}_{prompt_template_index}_p{args.threshold}"] = head_list
    
    shared_heads = []
    for key, value in lexical_dict.items():
        for head in value:
            if head in shared_heads:
                continue
            shared_heads.append(head)
    shared_heads.sort()
    shared_heads_output = [tuple(item) for item in shared_heads] 
    save_dir = os.path.join(args.save_root, model_short_name, args.d_name, "Heads", "MAPS_nnsight")
    save_path = os.path.join(
        save_dir,
        f"shared_{args.d_name}_heads_p{args.threshold}_k{args.k}.json",
    )
    with open(save_path, "w") as f:
        json.dump(
            shared_heads_output, f
        )
    print(f"Saved shared head list to {save_path}")
        # print(f"Loaded {pkl_path}")
        # print(f"{args.prompt_type} template {args.template_key}, p={args.threshold}: {len(head_list)} heads")
        # for layer, head in head_list:
        #     print(f"  Layer {layer}, Head {head}: matched {fraction_matched[layer, head]:.1%} of prompts")

