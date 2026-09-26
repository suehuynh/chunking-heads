import os
import json
import argparse
from nnsight import LanguageModel
import torch

from submodules.identify_heads_early_decode import *

torch.set_grad_enabled(False)

if __name__ == "__main__":
    """
    Identify lexical task heads or retrieval heads
    """
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
        default="output",)
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

    args = parser.parse_args()
    model_name = args.model_name
    save_root = args.save_root
    project_root = args.project_root
    d_name = args.d_name
    prompt_type = args.prompt_type
    monitor_other_task = args.monitor_other_task
    batch_size = args.batch_size
    k = args.k
    exp_size = args.exp_size
    component_type = args.component_type
    dataset_folder = args.dataset_folder
    remote = args.remote
    task_relation_dict_type = args.task_relation_dict_type

    if component_type == "lexical_task":
        component_type = "Relation"

    print("task_relation_dict_type", task_relation_dict_type)
    print("model_name", model_name)
    print("d_name", d_name)
    print("remote", remote)
    print("prompt_type", prompt_type)

    # Load
    with open(os.path.join(project_root, "datasets", "dataset_info", 
        f"instruction_dict.json"), "r"
    ) as f:
        instruction_dict = json.load(f)

    if task_relation_dict_type == "human":
        file_name = "task_relation_dict.json"
    elif task_relation_dict_type == "llms":
        file_name = "task_relation_dict_3_llms.json"
    elif task_relation_dict_type == "combined":
        file_name = "task_relation_dict_combined.json"
    else:
        raise ValueError(f"task_relation_dict_type {task_relation_dict_type} not supported")
    with open(os.path.join(project_root, "datasets", "dataset_info", 
        file_name), "r"
    ) as f:
        task_relation_dict = json.load(f)

    # Parameters 
    n_match_list=[1]
    k_list=None
    n_match = None
    
    # n_match_list = None
    # k_list=[20,25]
    # n_match = 1

    if monitor_other_task:
        task_list = task_relation_dict.keys()
    # elif dataset_folder == "../datasets/compositional":
    elif dataset_folder == "datasets/compositional":
        # load composition task list 
        with open(os.path.join(project_root, "datasets", 
         "dataset_info", f"compositional_task_dict.json"), "r") as f:
            compositional_task_dict = json.load(f)
        task_list = compositional_task_dict[d_name]
        assert type(task_list) == list
    else:
        task_list = [d_name]

    print("monitor_other_task", monitor_other_task)
    print("n_match_list", n_match_list)
    print("component_type", component_type)

    # Load model 
    print("To load model")
    model = LanguageModel(
        model_name,
        device_map="auto",
        dispatch=True if not remote else False
    )
    model_name = model_name.split("/")[-1]
    print("Model loaded")

    # Save path 
    if task_relation_dict_type == "human":
        save_path = save_path = os.path.join(save_root, model_name, d_name,
                "Heads", "MAPS_nnsight", f"{component_type}_across_tasks",)
    elif task_relation_dict_type == "llms":
        save_path = save_path = os.path.join(save_root, model_name, d_name,
                "Heads", "MAPS_nnsight", f"{component_type}_across_tasks_3_llms",)
    elif task_relation_dict_type == "combined":
        save_path = save_path = os.path.join(save_root, model_name, d_name,
                "Heads", "MAPS_nnsight", f"{component_type}_across_tasks_combined",)
    else:
        raise ValueError(f"task_relation_dict_type {task_relation_dict_type} not supported")

    if prompt_type == "IP":
        ## load behavior results
        with open(os.path.join(save_root, model_name,
            "across_tasks", "Behavior", "IP_vary_n_inst_behavior.json"), "r") as f:
            result_dict_IP_tasks = json.load(f)
            
        ## get best acc index
            acc_list = [result_dict_IP_tasks[d_name][str(i)]['accuracy'] for i in range(5)]
            best_index = np.argmax(acc_list).item()
            prompt_template_index = best_index
    elif prompt_type == "EP":
        prompt_template_index = 30
    else:
        raise ValueError(f"prompt_type {prompt_type} not supported")

    # Run and save the results
    result = get_save_MAPS_scores_across_tasks_nnsight(
        model=model, model_name=model_name,
        prompt_type=prompt_type, prompt_template_index=prompt_template_index,
        d_name=d_name, correct_or_incorrect="correct",
        task_relation_dict=task_relation_dict,
        batch_size=batch_size, k=k, exp_size=exp_size,
        n_match_list=n_match_list, task_list=task_list, 
        component_type=component_type,
        instruction_dict=instruction_dict,
        save=True, save_root=save_root,     
        k_list=k_list, n_match=n_match,
        dataset_folder=dataset_folder,
        remote=remote, save_path=save_path,
    )
   