import os
import pickle
import json
import numpy as np
from typing import List
# from transformer_lens import HookedTransformer
from nnsight import LanguageModel
from shared_utils.shared_utils import *
from shared_utils.prompt_utils import *
# from submodules.identify_heads_early_decode import *
from shared_utils.wrapper import get_accessor_config, get_model_specs, ModelAccessor

def get_heads_per_prompt(
    target_task:str=None, prompt_type:str=None, prompt_template_index:int=None,
    n_match:int=None, component_type:str="Relation",
    other_task:str=None, correct_incorrect:str="correct",
    save_root:str=None, model_name:str=None,
):
    save_path = os.path.join(save_root, model_name, target_task,
        "Heads", "MAPS", f"{component_type}_across_tasks",)
    with open(os.path.join(save_path, 
        f"{target_task}_MAPS_{component_type}_heads_across_tasks_{prompt_type}_{prompt_template_index}_{correct_incorrect}.pkl"
    ), "rb") as f:
        result = pickle.load(f)
        
    maps_scores_2_analyze = result[n_match][other_task] # shape: (prompts, layers, heads)

    heads_per_prompt = maps_scores_2_analyze.sum(axis=(1,2)) # shape: (prompts)

    return heads_per_prompt

def get_n_heads_distribution_across_n_match(
    target_task:str=None, other_task:str=None, 
    prompt_type:str=None, prompt_template_index:int=None,
    n_match_list:List[int]=None, component_type:str="Relation",
    save_root:str=None, model_name:str=None,
):
    n_heads_distribution_across_n_match = {}
    for n_match in n_match_list:
        n_heads_per_prompt = get_heads_per_prompt(
            target_task=target_task, other_task=other_task,
            prompt_type=prompt_type, prompt_template_index=prompt_template_index,
            n_match=n_match, component_type=component_type,
            save_root=save_root, model_name=model_name,
        ) # shape: (prompts)
        n_heads_distribution_across_n_match[n_match] = n_heads_per_prompt

    return n_heads_distribution_across_n_match

def get_n_heads_across_n_match_threshold(
    target_task:str=None, other_task:str=None, 
    prompt_type:str=None, prompt_template_index:int=None,
    n_match_list:List[int]=None, threshold_list:List[float]=[0.1, 0.15, 0.2, 0.25, 0.3],  
    save_root:str=None, model_name:str=None,
    component_type:str="Relation",
):
    n_heads_across_n_match_threshold = {}
    for threshold in threshold_list:
        n_heads_across_n_match = {}
        for n_match in n_match_list:
            head_list = get_task_heads_list(
                target_task=target_task, other_task=other_task,
                save_root=save_root, model_name=model_name,
                prompt_type=prompt_type, prompt_template_index=prompt_template_index,
                correct_incorrect="correct",
                n_match=n_match, MAPS_score_threshold=threshold, 
            )
            n_heads_across_n_match[n_match] = len(head_list)
        n_heads_across_n_match_threshold[threshold] = n_heads_across_n_match

    return n_heads_across_n_match_threshold

def get_n_heads_across_k(
    target_task:str=None, other_task:str=None, 
    prompt_type:str=None, prompt_template_index:int=None,
    k_list:List[int]=None,  
    n_match:int=1, threshold:float=0.1,
    save_root:str=None, model_name:str=None,
    component_type:str="Relation",
):
    n_heads_across_k = {}
    for k in k_list:
        head_list = get_task_heads_list_k(
            target_task=target_task, other_task=other_task,
            save_root=save_root, model_name=model_name,
            prompt_type=prompt_type, prompt_template_index=prompt_template_index,
            correct_incorrect="correct",
            k=k, MAPS_score_threshold=threshold, 

        )
        n_heads_across_k[k] = len(head_list)

    return n_heads_across_k

def calc_MAPS_score(
    projection, k, dst_tokens,
    relation=True,
    result_type="mean_of_batch",
    n_match=1,
):
    """
    Calculate the mean score for a batch of examples or the binary score for each example
        1: number of top decoded words matched dst_tokens >= n_match
  
    Calculate MAPS score for  decoded vectors(projection) of shape (batch, d_vocab)
    Args:
        projection: (batch, d_vocab)
        k: int
        dst_tokens: (batch) or (n_options)
        relation: True for relation heads, False for retrieval heads
        result_type: "mean_of_batch" or "each_sample_in_batch"
            "mean_of_batch": return the mean of the batch
            "each_sample_in_batch": return binary_scores_per_batch_item
    Returns: float or (batch,)
        binary_scores_per_batch_item: (batch,)
        or MAPS_score: float
    """
    dst_tokens = dst_tokens.to(projection.device) 
    _, indices = torch.topk(projection, k) # indices (batch, k)

    if relation == False: # dst_tokens: (batch)
        matches = (indices == dst_tokens.unsqueeze(1)) #(batch, k)
        total_matches_per_batch_item = matches.sum(dim=1) #(batch)
        #MAPS_score = matches.float().mean().item()
        batch_item_has_n_matches = (total_matches_per_batch_item >= n_match)
        # Convert boolean to float (True=1.0, False=0.0)
        binary_scores_per_batch_item = batch_item_has_n_matches.float()

    else: # dst_tokens' shape is (n_options,)
        # For each data point, we want to check if any of the top k indices
        # matches any of the n_options dst_tokens. If so, that batch item gets a score of 1.
        
        # Expand dst_tokens (n_options,) to (batch, 1, n_options) for broadcasting
        batch_size = indices.shape[0]
        dst_tokens_expanded = dst_tokens.unsqueeze(0).unsqueeze(1).expand(batch_size, -1, -1)

        # Compare each top-k index with each of the dst_tokens for that batch item
        # (batch, k, 1) == (batch, 1, n_options) results in (batch, k, n_options)
        individual_matches = (indices.unsqueeze(2) == dst_tokens_expanded)

        # For each (batch, k) element, check if it matched ANY of the n_options dst_tokens
        # This results in (batch, k) where True means one of the dst_tokens was matched
        any_dst_token_matched_per_k_index = individual_matches.any(dim=2)

        # Count the total number of top-k indices that matched ANY dst_token
        # This results in (batch)
        total_matches_per_batch_item = any_dst_token_matched_per_k_index.sum(dim=1)

        # Check if the total number of matches is greater than or equal to n
        # This results in (batch)
        batch_item_has_n_matches = (total_matches_per_batch_item >= n_match)

        # # For each batch item, check if ANY of its top-k indices matched ANY of the dst_tokens
        # # This results in (batch) where True means at least one top-k index matched for that batch item
        # batch_item_has_match = any_dst_token_matched_per_k_index.any(dim=1)

        # Convert boolean to float (True=1.0, False=0.0)
        binary_scores_per_batch_item = batch_item_has_n_matches.float()

    if result_type == "mean_of_batch":
        # Calculate the mean of these binary scores across the batch
        MAPS_score = binary_scores_per_batch_item.mean().item()
        return MAPS_score
    elif result_type == "each_sample_in_batch":
        return binary_scores_per_batch_item
    else:
        raise ValueError(f"Invalid result_type: {result_type}")

def calc_MAPS_score_from_activations(model,
    cache, k, dst_tokens, current_batch_size, 
    result_type="mean_of_batch", relation=True,
    device="cuda", apply_ln=False, n_match=1,
):
    """
    Args 
        dst_tokens: (batch) or (n_options)
    Returns: 
        MAPS_scores: 
            (n_layers, n_heads) if mean_of_batch, 
            (current_batch_size, n_layers, n_heads) if each_sample_in_batch
    """
    if result_type == "mean_of_batch":
        MAPS_scores = np.zeros((model.cfg.n_layers, model.cfg.n_heads))
    elif result_type == "each_sample_in_batch": 
        MAPS_scores = np.zeros((current_batch_size, model.cfg.n_layers, model.cfg.n_heads))
        
    for layer in range(model.cfg.n_layers):
        layer_output = cache[f"blocks.{layer}.attn.hook_z"][:,-1,:,:].to(device) # (batch, n_heads, d_head)
        heads_outputs = torch.einsum( 
            'bnh, nhm -> bnm',
            #'batch n_heads d_head, n_heads d_head d_model -> batch n_heads d_model',
            layer_output, model.state_dict()[f"blocks.{layer}.attn.W_O"].to(device),
        )
        if apply_ln:
            heads_outputs = model.ln_final(heads_outputs) # apply final layer norm  # (batch, n_heads, d_model)
        # project to vocab space
        heads_projections = heads_outputs @ model.state_dict()["unembed.W_U"].to(device) # (batch, n_heads, d_vocab)
        for head in range(model.cfg.n_heads):
            projection = heads_projections[:, head] # (batch, d_vocab)
            if result_type == "mean_of_batch":
                MAPS_scores[layer, head] = calc_MAPS_score(
                    projection, k, dst_tokens, relation=relation,
                    result_type=result_type, n_match=n_match)
            elif result_type == "each_sample_in_batch":
                MAPS_scores[:, layer, head] = calc_MAPS_score(
                    projection, k, dst_tokens, relation=relation, 
                    result_type=result_type, n_match=n_match).cpu().numpy() # output of function is (batch,)
    
    return MAPS_scores

import torch

def convert_expand_candidate_strs_to_token_ids_nnsight(
    model, 
    candidate_strs: list[str],
    output_str_tokens: bool = False,
):
    """
    Converts a list of candidate strings to token IDs using an nnsight LanguageModel.
    
    It applies various mutations (spacing, capitalization, punctuation) to find 
    valid single-token representations in the model's vocabulary.
    """
    tokenizer = model.tokenizer
    
    # Store tuples of (token_id, decoded_string) to handle deduplication logic
    unique_tokens = set()

    def process(text):
        # Encode without special tokens to see if it forms exactly one token
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) == 1:
            token_id = ids[0]
            # decode([id]) recovers the human-readable string (e.g., " company" or "国家")
            # preventing artifacts like 'Ġ' or 'åĽ½å®¶'
            token_str = tokenizer.decode([token_id])
            unique_tokens.add((token_id, token_str))

    for string in candidate_strs:
        # 1. Original string
        process(string)

        # 2. Add leading space (very common in GPT/Llama tokenizers)
        process(" " + string)
        
        # 3. Capitalization variations
        process(string.upper())      # WHOLE WORD
        process(string.capitalize()) # First letter
        process(string.lower())      # whole word

        # 4. Punctuation prefixes
        process("." + string)
        process("_" + string)
        process("-" + string)

    # Sort by Token ID for deterministic results
    if not unique_tokens:
        candidate_token_ids = []
        candidate_strs_set = set()
    else:
        sorted_tokens = sorted(list(unique_tokens), key=lambda x: x[0])
        candidate_token_ids = [t[0] for t in sorted_tokens]
        # Create the set of strings for the return value
        candidate_strs_set = {t[1] for t in sorted_tokens}

    if output_str_tokens:
        return candidate_token_ids, candidate_strs_set
    else:
        return torch.tensor(candidate_token_ids)
        
def convert_expand_candidate_strs_to_token_ids(
    model, 
    candidate_strs:list[str],
    output_str_tokens:bool=False,
):
    candidate_strs_set = set()
    # Try both with and without leading space for robustness
    for string in candidate_strs:
        # Original 
        str_tokens = model.to_str_tokens(string, prepend_bos=False)
        if len(str_tokens) == 1:
            candidate_strs_set.add(str_tokens[0])

        # Add leading space (common tokenization pattern)
        spaced_str_tokens = model.to_str_tokens(" " + string, prepend_bos=False)
        if len(spaced_str_tokens) == 1:
            candidate_strs_set.add(spaced_str_tokens[0])

    for string in candidate_strs:  
        # Capitalize the whole word 
        capitalize_whole_word = model.to_str_tokens(string.upper(), prepend_bos=False)
        if len(capitalize_whole_word) == 1:
            candidate_strs_set.add(capitalize_whole_word[0])
        
        # Capitalize the first letter
        capitalize_first_letter = model.to_str_tokens(string.capitalize(), prepend_bos=False)
        if len(capitalize_first_letter) == 1:
            candidate_strs_set.add(capitalize_first_letter[0])

        # lower case the whole word 
        lower_whole_word = model.to_str_tokens(string.lower(), prepend_bos=False)
        if len(lower_whole_word) == 1:
            candidate_strs_set.add(lower_whole_word[0])

        # add "." 
        add_dot = model.to_str_tokens( "." + string, prepend_bos=False)
        if len(add_dot) == 1:
            candidate_strs_set.add(add_dot[0])
        
        # add "_"
        add_underscore = model.to_str_tokens( "_" + string, prepend_bos=False)
        if len(add_underscore) == 1:
            candidate_strs_set.add(add_underscore[0])

        # add "-",
        add_dash = model.to_str_tokens( "-" + string, prepend_bos=False)
        if len(add_dash) == 1:
            candidate_strs_set.add(add_dash[0])

    candidate_strs_list = list(candidate_strs_set)
    candidate_token_ids = [model.tokenizer(candidate_strs_list[i], add_special_tokens=False).input_ids[0] for i in range(len(candidate_strs_list))]

    if output_str_tokens:
        return candidate_token_ids, candidate_strs_set
    else:
        return torch.tensor(candidate_token_ids)

def get_save_MAPS_scores_across_tasks_nnsight_generation(
    model:LanguageModel, model_name:str, prompts:list, 
    d_name:str, n_match_list:list, n_match:int,
    component_type:str="Relation",
    k:int=10, k_list:list=None, 
    task_list:list=None,
    save:bool=True, save_path:str=None,
    batch_size:int=10,
    task_relation_dict:dict=None,
    save_root:str=None,
):
    # Model specs 
    spec = get_model_specs(model)
    n_layers, n_heads, d_model = spec["n_layers"], spec["n_heads"], spec["d_model"]

    if component_type == "Relation":
        relation = True
        component_name = "lexical_task_heads"
    else:
        relation = False

    # --- Get prompt tokens ---
    all_prompt_tokens = model.tokenizer(
        prompts, 
        padding=True, 
        padding_side="left", 
        return_tensors="pt"
    )["input_ids"]

    all_answer_tokens = None 

    # --- Cache heads'output & calculate MAPS scores ---
    num_examples = all_prompt_tokens.shape[0]
    #print("num of examples: ", num_examples)
    results = {}
    if n_match_list is not None:
        for n_match in n_match_list:
            results[n_match] = {k: np.empty((num_examples, n_layers, n_heads)) for k in task_list}
    elif k_list is not None:
        for k in k_list:
            results[k] = {t: np.empty((num_examples, n_layers, n_heads)) for t in task_list}
    else:
        raise ValueError("Either n_match_list or k_list must be provided")

    _, cached_act = cache_act_nnsight(model=model, 
            cache_component_type="heads", 
            all_tokens=all_prompt_tokens,
            all_answer_tokens=all_answer_tokens,
            remote=False, batch_size=batch_size,
        )
    if n_match_list is not None:
        for n_match in n_match_list:
            for task in task_list:
                if relation:
                    # Convert candidate words to token IDs for each task ---
                    candidate_token_ids, _ = convert_expand_candidate_strs_to_token_ids_nnsight(
                        model, 
                        candidate_strs=task_relation_dict[task],
                        output_str_tokens=True
                    )
                    # convert to tensor 
                    candidate_token_ids = torch.tensor(candidate_token_ids)
                else:
                    candidate_token_ids = all_answer_tokens

                metrics_raw = calculate_component_metrics_from_heads_output(
                    model=model, heads_output=cached_act, 
                    dst_tokens=candidate_token_ids,
                    apply_ln=True, k=k, n_match=n_match, 
                    component_type=component_name
                )

                results[n_match][task] = metrics_raw["MAPS_scores"]
    elif k_list is not None:
        for k in k_list:
            for task in task_list:
                if relation:
                    # Convert candidate words to token IDs for each task ---
                    candidate_token_ids, _ = convert_expand_candidate_strs_to_token_ids_nnsight(
                        model, 
                        candidate_strs=task_relation_dict[task],
                        output_str_tokens=True
                    )
                    # convert to tensor 
                    candidate_token_ids = torch.tensor(candidate_token_ids)
                else:
                    candidate_token_ids = all_answer_tokens

                metrics_raw = calculate_component_metrics_from_heads_output(
                    model=model, heads_output=cached_act, 
                    dst_tokens=candidate_token_ids,
                    apply_ln=True, k=k, n_match=n_match, 
                    component_type=component_name
                )

                results[k][task] = metrics_raw["MAPS_scores"]
    else:
        raise ValueError("Either n_match_list or k_list must be provided")

    #print(f"results_{n_match}_{task}.shape", results[n_match][task].shape)

    if save:
        if save_path is None:
            save_path = os.path.join(save_root, model_name, d_name,
                "Heads", "MAPS", f"{component_type}_across_tasks",)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        print(save_path)
        with open(os.path.join(save_path, 
            f"{d_name}_MAPS_{component_type}_heads_across_tasks.pkl"),
        "wb") as f:
            pickle.dump(results, f)

def get_save_MAPS_scores_across_tasks_nnsight(
    model: LanguageModel=None,
    model_name: str=None,
    save_root: str=None,
    prompt_type: str=None,
    prompt_template_index: int=None,
    d_name: str=None,
    correct_or_incorrect: str="correct",
    exp_size: int=None,
    task_relation_dict: dict=None,
    batch_size: int=10, k: int=10, 
    instruction_dict: dict=None,
    save=False, save_path=None,
    component_type: str="Relation",
    task_list: list=None, n_match_list: list=None,
    k_list: list=None, n_match: int=None,
    dataset_folder:str="../datasets/abstractive",
    remote: bool=False,
):
    """
    Args: 
        component_type: "Relation" or "Retrieval"
    Returns: 
        dict of {n_match: {task: (num_examples, n_layers, n_heads)}}
        for each n_match, each task and each example, get the matrix of MAPS socres (0 or 1) for each head
    """
    # Model specs 
    spec = get_model_specs(model)
    n_layers, n_heads, d_model = spec["n_layers"], spec["n_heads"], spec["d_model"]

    if component_type == "Relation":
        relation = True
        component_name = "lexical_task_heads"
    else:
        relation = False
    # Get correct/incorrect prompt index 
    if prompt_type == "EP":
        behavior_file_name = f"{prompt_type}_vary_n_shot_behavior.json"
    else:
        behavior_file_name = f"{prompt_type}_vary_n_inst_behavior.json"
    path = os.path.join(save_root, model_name, 
        "across_tasks", "Behavior", behavior_file_name)
    with open(path, "r") as f:
        behavior_data = json.load(f)

    exp_size = exp_size
    correct_index = behavior_data[d_name][str(prompt_template_index)]['correct_index']
    index_2_analyze = correct_index
    if correct_or_incorrect == "incorrect":
        n_dataset = behavior_data[d_name][str(prompt_template_index)]['n_dataset']
        incorrect_index = set(np.arange(n_dataset)) - set(correct_index)
        incorrect_index = list(sorted(incorrect_index))
        index_2_analyze = incorrect_index
       
    if exp_size is not None: 
        index_2_analyze = index_2_analyze[:exp_size]

    # Get prompt tokens 
    token_dict = get_prompt_token(model=model, d_name=d_name,
        prompt_type=prompt_type, 
        prompt_index=prompt_template_index, 
        instruction_dict=instruction_dict,    
        return_answer_tokens=True,
        dataset_folder=dataset_folder
        )

    all_prompt_tokens = token_dict['prompt_tokens'][index_2_analyze]
    all_answer_tokens = token_dict['answer_tokens'][index_2_analyze]

    # --- Cache heads'output & calculate MAPS scores ---
    num_examples = all_prompt_tokens.shape[0]
    #print("num of examples: ", num_examples)
    results = {}
    if n_match_list is not None:
        for n_match in n_match_list:
            results[n_match] = {k: np.empty((num_examples, n_layers, n_heads)) for k in task_list}
    elif k_list is not None:
        for k in k_list:
            results[k] = {t: np.empty((num_examples, n_layers, n_heads)) for t in task_list}
    else:
        raise ValueError("Either n_match_list or k_list must be provided")

    _, cached_act = cache_act_nnsight(model=model, 
            cache_component_type="heads", 
            all_tokens=all_prompt_tokens,
            all_answer_tokens=all_answer_tokens,
            remote=remote, batch_size=batch_size,
        )
    if n_match_list is not None:
        for n_match in n_match_list:
            for task in task_list:
                if relation:
                    # Convert candidate words to token IDs for each task ---
                    candidate_token_ids, _ = convert_expand_candidate_strs_to_token_ids_nnsight(
                        model, 
                        candidate_strs=task_relation_dict[task],
                        output_str_tokens=True,
                    )
                    # convert to tensor 
                    candidate_token_ids = torch.tensor(candidate_token_ids)
                else:
                    candidate_token_ids = all_answer_tokens

                metrics_raw = calculate_component_metrics_from_heads_output(
                    model=model, heads_output=cached_act, 
                    dst_tokens=candidate_token_ids,
                    apply_ln=True, k=k, n_match=n_match, 
                    component_type=component_name,
                )

                results[n_match][task] = metrics_raw["MAPS_scores"]
    elif k_list is not None:
        for k in k_list:
            for task in task_list:
                if relation:
                    # Convert candidate words to token IDs for each task ---
                    candidate_token_ids = convert_expand_candidate_strs_to_token_ids_nnsight(
                        model, 
                        candidate_strs=task_relation_dict[d_name],
                        output_str_tokens=True
                    )
                    # convert to tensor 
                    candidate_token_ids = torch.tensor(candidate_token_ids)
                else:
                    candidate_token_ids = all_answer_tokens

                metrics_raw = calculate_component_metrics_from_heads_output(
                    model=model, heads_output=cached_act, 
                    dst_tokens=candidate_token_ids,
                    apply_ln=True, k=k, n_match=n_match, 
                    component_type=component_name
                )

                results[k][task] = metrics_raw["MAPS_scores"]
    else:
        raise ValueError("Either n_match_list or k_list must be provided")
    
    #print(f"results_{n_match}_{task}.shape", results[n_match][task].shape)
    
    if save:
        if save_path is None:
            save_path = os.path.join(save_root, model_name, d_name,
                "Heads", "MAPS", f"{component_type}_across_tasks",)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        #print(save_path)

        with open(os.path.join(save_path, 
            f"{d_name}_MAPS_{component_type}_heads_across_tasks_{prompt_type}_{prompt_template_index}_{correct_or_incorrect}.pkl"),
        "wb") as f:
            pickle.dump(results, f)
        print("saved to", save_path)

    return results

        
    
def get_save_MAPS_scores_across_tasks(
    model: HookedTransformer=None,
    model_name: str=None,
    save_root: str=None,
    prompt_type: str=None,
    prompt_template_index: int=None,
    d_name: str=None,
    correct_or_incorrect: str="correct",
    exp_size: int=None,
    task_relation_dict: dict=None,
    batch_size: int=10, k: int=10, 
    result_type: str="mean_of_batch",
    instruction_dict: dict=None,
    save=False, save_path=None,
    apply_ln: bool=False, component_type: str="Relation",
    task_list: list=None, n_match_list: list=None,
    k_list: list=None, n_match: int=None,
):
    """
    Args: 
        component_type: "Relation" or "Retrieval"
    Returns: 
        dict of {n_match: {task: (num_examples, n_layers, n_heads)}}
        for each n_match, each task and each example, get the matrix of MAPS socres (0 or 1) for each head
    """
    if component_type == "Relation":
        relation = True
    else:
        relation = False
    # Get correct/incorrect prompt index 
    if prompt_type == "EP":
        behavior_file_name = f"{prompt_type}_vary_n_shot_behavior.json"
    else:
        behavior_file_name = f"{prompt_type}_vary_n_inst_behavior.json"
    path = os.path.join(save_root, model_name, 
        "across_tasks", "Behavior", behavior_file_name)
    with open(path, "r") as f:
        behavior_data = json.load(f)

    correct_index = behavior_data[d_name][str(prompt_template_index)]['correct_index']
    index_2_analyze = correct_index
    if correct_or_incorrect == "incorrect":
        n_dataset = behavior_data[d_name][str(prompt_template_index)]['n_dataset']
        incorrect_index = set(np.arange(n_dataset)) - set(correct_index)
        incorrect_index = list(sorted(incorrect_index))
        index_2_analyze = incorrect_index
       
    if exp_size is not None: 
        index_2_analyze = index_2_analyze[:exp_size]

    # --- Get prompt tokens ---
    token_dict = get_prompt_token(model=model, d_name=d_name,
        prompt_type=prompt_type, 
        prompt_index=prompt_template_index, 
        instruction_dict=instruction_dict,    
        return_answer_tokens=True)

    all_prompt_tokens = token_dict['prompt_tokens'][index_2_analyze]
    all_answer_tokens = token_dict['answer_tokens'][index_2_analyze]

    # --- Cache heads'output & calculate MAPS scores ---
    num_examples = all_prompt_tokens.shape[0]
    print("num of examples: ", num_examples)
    results = {}
    if k_list is not None:
        for k in k_list:
            results[k] = {t: np.empty((num_examples, model.cfg.n_layers, model.cfg.n_heads)) for t in task_list}
    elif n_match_list is not None:
        for n_match in n_match_list:
            results[n_match] = {t: np.empty((num_examples, model.cfg.n_layers, model.cfg.n_heads)) for t in task_list}
    else:
        raise ValueError("Either k_list or n_match_list must be provided")
    
    for i in range(0, num_examples, batch_size):
        batch_start_index = i
        batch_end_index = min(i + batch_size, num_examples)
        input_batch = all_prompt_tokens[batch_start_index:batch_end_index]
        input_batch = input_batch.to(model.cfg.device)
        answer_tokens_batch = all_answer_tokens[batch_start_index:batch_end_index]
        current_batch_size = input_batch.shape[0]
        clean_run_cache = {}
        model.remove_all_hook_fns()
        model.add_caching_hooks([f"blocks.{layer}.attn.hook_z" for layer in range(model.cfg.n_layers)], cache=clean_run_cache)
        model(input_batch)
        model.remove_all_hook_fns()
        
        if k_list is not None:
            for k in k_list:
                for task in task_list:
                    if relation:
                        # Convert candidate words to token IDs for each task ---
                        candidate_token_ids = convert_expand_candidate_strs_to_token_ids(
                            model, 
                            candidate_strs=task_relation_dict[task],
                        )
                    else:
                        candidate_token_ids = answer_tokens_batch

                    MAPS_scores_batch = calc_MAPS_score_from_activations(model,
                        clean_run_cache, k, candidate_token_ids, 
                        current_batch_size, result_type=result_type, apply_ln=apply_ln,
                        n_match=n_match, relation=relation,
                    ) # (current_batch_size, n_layers, n_heads) if result_type == "each_sample_in_batch"

                    results[k][task][batch_start_index:batch_end_index, :, :] = MAPS_scores_batch
                
        elif n_match_list is not None:
            for n_match in n_match_list:
                for task in task_list:
                    if relation:
                        # Convert candidate words to token IDs for each task ---
                        candidate_token_ids = convert_expand_candidate_strs_to_token_ids(
                            model, 
                            candidate_strs=task_relation_dict[task],
                        )
                    else:
                        candidate_token_ids = answer_tokens_batch

                    MAPS_scores_batch = calc_MAPS_score_from_activations(model,
                        clean_run_cache, k, candidate_token_ids, 
                        current_batch_size, result_type=result_type, apply_ln=apply_ln,
                        n_match=n_match, relation=relation,
                    ) # (current_batch_size, n_layers, n_heads) if result_type == "each_sample_in_batch"

                    results[n_match][task][batch_start_index:batch_end_index, :, :] = MAPS_scores_batch
    
    #print(f"results_{n_match}_{task}.shape", results[n_match][task].shape)
    
    if save:
        if save_path is None:
            if k_list is not None:
                save_path = os.path.join(save_root, model_name, d_name,
                "Heads", "MAPS", f"{component_type}_across_tasks_vary_k",)
            elif n_match_list is not None:
                save_path = os.path.join(save_root, model_name, d_name,
                    "Heads", "MAPS", f"{component_type}_across_tasks",)
            else:
                raise ValueError("Either k_list or n_match_list must be provided")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        with open(os.path.join(save_path, 
            f"{d_name}_MAPS_{component_type}_heads_across_tasks_{prompt_type}_{prompt_template_index}_{correct_or_incorrect}.pkl"),
        "wb") as f:
            pickle.dump(results, f)

    return results

def get_intersection_matrix(
    dict_of_component_list_y, 
    dict_of_component_list_x, 
    ratio=False,
):
    """
    Get the intersection matrix of two sets of components across tasks.

    Args:
        dict_of_component_list_y: A dictionary where keys are dataset names and values are lists of components.
        dict_of_component_list_x: A dictionary where keys are dataset names and values are lists of components.
        datasets: A list of dataset names.
        Note: y because it corresponds to the rows of the matrics -> y-axis of the plot 
        grouped_components: a dictionary keyed by layer (integer), value is a list of component indices 

    Returns:
        A matrix (2D numpy array) of intersection counts between components across tasks, 
        and a dictionary of intersection components across tasks.
    """
    datasets = list(dict_of_component_list_y.keys())
    N = len(datasets)
    intersection_matrix_count = np.zeros((N,N))
    intersection_matrix_component = {}
    all_combined = []

    for i,(d_1)  in enumerate(datasets):
        R = dict_of_component_list_y[d_1]
        for j, (d_2) in enumerate(datasets):
            C = dict_of_component_list_x[d_2]
            intersection, count = find_overlapping_component_list(R, C)    

            if ratio:
                smaller_set = min(len(R), len(C))
                if smaller_set == 0:
                    intersection_matrix_count[i][j] = 0
                else:
                    intersection_matrix_count[i][j] = count / smaller_set
            else:
                intersection_matrix_count[i][j] = count
            
            intersection_matrix_component[d_1, d_2] = intersection
    return intersection_matrix_count, intersection_matrix_component

def find_overlapping_component_list(
    list1: List,
    list2: List
) -> List:
    """
    Finds the overlapping list of (layer_idx, head_idx) tuples between two input lists.

    This function is optimized by converting the input lists to sets before
    performing the intersection operation, which is highly efficient (O(min(len(set1), len(set2))))
    compared to nested loops (O(len(list1) * len(list2))).

    Args:
        list1: The first list of components, e.g., [(0, 1), (2, 3)].
        list2: The second list of components, e.g., [(1, 0), (2, 3)].

    Returns:
        A new list containing only the (layer_idx, head_idx) tuples that are
        present in both input lists.
    """
    # 1. Convert the lists to sets for efficient intersection.
    # set1 = set(list1)
    # set2 = set(list2)
    set1 = {tuple(item) for item in list1}
    set2 = {tuple(item) for item in list2}

    # 2. Compute the intersection of the two sets.
    # The '&' operator performs set intersection.
    overlapping_set = set1 & set2

    # 3. Convert the resulting set back to a list (the order is not guaranteed).
    overlapping_list = list(overlapping_set)
    return overlapping_list, len(overlapping_list)
    