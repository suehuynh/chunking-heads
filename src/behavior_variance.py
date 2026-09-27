import os
import json
import argparse
from nnsight import LanguageModel
import torch
torch.set_grad_enabled(False)

from shared_utils.prompt_utils import *

if __name__ == "__main__":
    """
    Perform inference runs and save the behavior variance of a dataset for a given model.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, 
        help="model name e.g. meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--d_name", type=str, required=True,)
    parser.add_argument("--save_root", type=str, 
        # default="../output",)
        default="output",)
    parser.add_argument("--project_root", type=str, 
        # default="../",
        default="",
        help="directory of the codebase ")
    parser.add_argument("--prompt_type", type=str, required=True, help="prompt type: EP or IP")
    parser.add_argument("--batch_size", type=int, default=20, help="batch size")
    parser.add_argument("--remote", type=bool, default=False, help="whether to use NDIF to run model remotely")
    # parser.add_argument("--dataset_folder", type=str, default="../datasets/abstractive", help="folder of the dataset")
    parser.add_argument("--dataset_folder", type=str, default="datasets/abstractive", help="folder of the dataset")

    args = parser.parse_args()
    model_name = args.model_name
    save_root = args.save_root
    project_root = args.project_root
    d_name = args.d_name
    prompt_type = args.prompt_type
    batch_size = args.batch_size
    remote = args.remote
    dataset_folder = args.dataset_folder
    
    print("model_name", model_name)
    print("remote", remote)
    print("prompt_type", prompt_type)
    print("d_name", d_name)
    print("project_root", project_root)

    # Load model 
    print("To load model")
    model = LanguageModel(
        model_name,
        device_map="auto",
        dispatch=True if not remote else False,
    )
    print("Model loaded")
    model_name = model_name.split("/")[-1]

    # Load file if exist, otherwise create a new dictionary
    if prompt_type == "EP":
        file_name = "EP_vary_n_shot_behavior.json"
        prompt_temp_idx_list = [1,2,3,4,5,10,20,30]
    elif prompt_type == "IP":
        file_name = "IP_vary_n_inst_behavior.json"
        prompt_temp_idx_list = [0,1,2,3,4]
    else:
        raise ValueError(f"prompt_type {prompt_type} not supported")
    save_path = os.path.join(save_root, model_name, "across_tasks", "Behavior")
    if os.path.exists(os.path.join(save_path, file_name)):
        with open(os.path.join(save_path, file_name), "r"
        ) as f:
            result_dict = json.load(f)
            print(f"Behavior file {file_name} loaded")
    else:
        print("Behavior file does not exist, creating a new dictionary")
        result_dict = {}
    
    # Load instruction_dict
    with open(os.path.join(project_root, "datasets", "dataset_info", 
        f"instruction_dict.json"), "r"
    ) as f:
        instruction_dict = json.load(f)

    # Run inference and check correctness
    result_dict[d_name] = {}
    for prompt_temp_index_idx in prompt_temp_idx_list:
        result_dict[d_name][prompt_temp_index_idx] = {}
        with open(os.path.join(dataset_folder, f"{d_name}.json")) as f: 
            dataset = json.load(f)
        if prompt_type == "EP":
            prompts, answers, _ = create_few_shot_prompts(
                dataset, n_shot = prompt_temp_index_idx, delimiter = ";", q_bos=" ", a_bos=" ", qa_delimiter=":"
            )
        elif prompt_type == "IP":
            prompts, answers = create_instruction_prompts(dataset, 
            instruction_dict[d_name][str(prompt_temp_index_idx)])
            
        prompt_dict= check_correctness(model=model, prompts=prompts, answers=answers, 
            batch_size=batch_size, return_pred_tokens=False, return_answer_tokens=False,
            remote=remote
        )
        correct_index = prompt_dict['correct_index']
        acc = len(correct_index) / len(prompts)
        result_dict[d_name][prompt_temp_index_idx]['accuracy'] = acc
        result_dict[d_name][prompt_temp_index_idx]['correct_index'] = correct_index
        result_dict[d_name][prompt_temp_index_idx]['n_correct_index'] = len(correct_index)
        result_dict[d_name][prompt_temp_index_idx]['n_dataset'] = len(prompts)
    # Save after each dataset is done 
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with open(os.path.join(save_path, file_name), "w"
    ) as f:
        json.dump(result_dict, f)
    print(f"Behavior file {file_name} saved")