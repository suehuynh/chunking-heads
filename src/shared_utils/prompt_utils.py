import random
import torch
from tqdm.auto import tqdm
import os
import json
# from transformer_lens import HookedTransformer
from nnsight import LanguageModel
# from transformers import AutoTokenizer

# from Fix_incorrect_prompts.Fix_incorrect_prompts_utils import *


def filter_token_pairs_by_length(raw_dataset, nn_model, input_len=1, output_len=1):
    filtered_dataset = []

    for item in raw_dataset:
        input_tokens = nn_model.tokenizer(" " + item["input"], add_special_tokens=False)["input_ids"]
        output_tokens = nn_model.tokenizer(" " + item["output"], add_special_tokens=False)["input_ids"]

        if len(input_tokens) == input_len and len(output_tokens) == output_len:
            filtered_dataset.append(item)

    return filtered_dataset

def create_few_shot_prompts(data, n_shot, delimiter, q_bos, a_bos, qa_delimiter):

    prompts = []
    answers = []
    query_index = []

    # For each possible last example
    # for i in range(n_shot, len(data)):
    for i in range(0, len(data)):
        context = ""
        for j in range(i - n_shot, i):
            example = data[j]
            context += f"{q_bos}{example['input']}{qa_delimiter}{a_bos}{example['output']}{delimiter}"

        query = data[i]
        context += f"{q_bos}{query['input']}{qa_delimiter}"

        prompts.append(context)
        answers.append(a_bos + query["output"])
        query_index.append(i)

    return prompts, answers, query_index

def create_random_output_prompts(data, n_shot, delimiter, q_bos, a_bos, qa_delimiter):
    prompts = []

    # For each possible last example
    for i in range(0, len(data)):
        context = ""
        for j in range(i - n_shot, i):
            example = data[j]
            special_token = f'<|reserved_special_token_{j}|>'
            context += f"{q_bos}{example['input']}{qa_delimiter}{a_bos}{special_token}{delimiter}"

        query = data[i]
        context += f"{q_bos}{query['input']}{qa_delimiter}"

        prompts.append(context)

    return prompts

def create_corrupt_prompts_output_shuffle(data, n_shot, delimiter, q_bos, a_bos, qa_delimiter):
    """
    Creates corrupted data by shuffling the outputs, then uses create_few_shot_prompts
    """
    # Create a copy of data to avoid modifying the original
    corrupt_data = []
    outputs = [d["output"] for d in data]
    shuffled_outputs = random.sample(outputs, len(outputs))
    for i, d in enumerate(data):
        corrupt_data.append({"input": d["input"], "output": shuffled_outputs[i]})

    return create_few_shot_prompts(corrupt_data, n_shot, delimiter, q_bos, a_bos, qa_delimiter)


def create_corrupt_prompts_input_shuffle(data, n_shot, delimiter, q_bos, a_bos, qa_delimiter):
    """
    Creates corrupted prompts where the INPUTS of the few-shot examples within
    each prompt's context are shuffled. The OUTPUTS remain paired with their
    original position's input slot. The final query INPUT remains unchanged.
    """

    corrupt_prompts = []
    corrupt_answers = []

    if len(data) <= n_shot:
        print("Warning: Not enough data points for the specified n_shot.")
        return [], []

    original_inputs = [d["input"] for d in data]
    indices = list(range(len(data)))
    random.shuffle(indices)

    globally_shuffled_inputs = random.sample(original_inputs, len(original_inputs))

    for i in range(0, len(data)):
        corrupt_context = ""
        for j in range(i - n_shot, i):
            input_for_context = globally_shuffled_inputs[j]
            output_for_context = data[j]["output"]
            corrupt_context += f"{q_bos}{input_for_context}{qa_delimiter}{a_bos}{output_for_context}{delimiter}"

        original_query_input = data[i]["input"]
        corrupt_context += f"{q_bos}{original_query_input}{qa_delimiter}"
        corrupt_prompts.append(corrupt_context)
        original_answer = data[i]["output"]
        corrupt_answers.append(a_bos + original_answer)

    return corrupt_prompts, corrupt_answers


def create_instruction_prompts(dataset, instruction):
    assert "{input}" in instruction

    prompts, answers = [], []
    for data in dataset:
        prompts.append(instruction.format(input=data["input"]))
        answers.append(" " + data["output"])
    return prompts, answers

def create_instruction_prompts_corrupt_query(dataset, instruction):
    prompts = []
    for data in dataset:
        random_query = " " + "<|reserved_special_token_1|>"
        prompts.append(instruction.format(input=random_query))
    return prompts

def return_correct_index(model, prompts, answers, 
    batch_size=10, remote=False, query_index=None,
    return_pred_tokens=False
):
    correct_index = []

    prompt_tokens = model.tokenizer(
        prompts, 
        padding=True, 
        padding_side="left", 
        return_tensors="pt"
    )["input_ids"]
    answer_tokens = model.tokenizer(
        answers,
        add_special_tokens=False,
        padding=True,
        padding_side="right",
        return_tensors="pt",
    )["input_ids"][:, 0]
    
    pred_tokens = []
    # for i in tqdm(range(0, len(prompts), batch_size)):
    for i in range(0, len(prompts), batch_size):
        batch_prompt_tokens = prompt_tokens[i : i + batch_size]
        batch_answer_tokens = answer_tokens[i : i + batch_size]
        batch_pred_tokens = (
            model.trace(batch_prompt_tokens, trace=False, remote=remote).logits[:, -1, :].argmax(dim=-1).cpu()
        )
        if return_pred_tokens:
            pred_tokens.extend(batch_pred_tokens.tolist())

        batch_correct = torch.where(batch_answer_tokens == batch_pred_tokens)[0].tolist()
        if query_index is not None:
            correct_index += [query_index[idx+i] for idx in batch_correct]
        else:
            correct_index += [idx + i for idx in batch_correct]
    
    # print(f"Accuracy: {len(correct_index)/len(prompts)} ({len(correct_index)}/{len(prompts)})")
    if return_pred_tokens:
        return correct_index, pred_tokens
    else:
        return correct_index


def create_target_output_prompts(data, n_shot, delimiter, q_bos, a_bos, qa_delimiter):
    prompts = []

    # For each possible last example
    for i in range(0, len(data)):
        context = ""
        for j in range(i - n_shot, i):
            example = data[j]
            special_token = 'target'
            context += f"{q_bos}{example['input']}{qa_delimiter}{a_bos}{special_token}{delimiter}"

        query = data[i]
        context += f"{q_bos}{query['input']}{qa_delimiter}"

        prompts.append(context)

    return prompts

def create_corrupt_prompts_abstract_causal(data, n_shot, delimiter, q_bos, a_bos, qa_delimiter):
    """
    : input output ;
    """

    corrupt_prompts = []
    corrupt_answers = []

    if len(data) <= n_shot:
        print("Warning: Not enough data points for the specified n_shot.")
        return [], []

    for i in range(0, len(data)):
        corrupt_context = ""
        for j in range(i - n_shot, i):
            input_for_context = data[j]["input"]
            output_for_context = data[j]["output"]
            corrupt_context += f"{qa_delimiter}{q_bos}{input_for_context}{a_bos}{output_for_context}{delimiter} "

        original_query_input = data[i]["input"]
        corrupt_context += f"{qa_delimiter}{q_bos}{original_query_input}"
        corrupt_prompts.append(corrupt_context)
        original_answer = data[i]["output"]
        corrupt_answers.append(a_bos + original_answer)

    return corrupt_prompts, corrupt_answers
 
def create_random_query_prompts(data, n_shot, delimiter, q_bos, a_bos, qa_delimiter):

    prompts = []

    # For each possible last example
    for i in range(0, len(data)):
        context = ""
        for j in range(i - n_shot, i):
            example = data[j]
            context += f"{q_bos}{example['input']}{qa_delimiter}{a_bos}{example['output']}{delimiter}"

        query = " " + "<|reserved_special_token_1|>"
        #context += f"{q_bos}{query}{qa_delimiter}"
        context += f"{query}{qa_delimiter}"

        prompts.append(context)

    return prompts

def create_zs_prompts(dataset, return_answers=False):
    prompts, answers = [], []
    for data in dataset:
        prompts.append(" " + data["input"] +":")
        answers.append(" " + data["output"])
    if return_answers:
        return prompts, answers
    else:
        return prompts

def create_minimal_corrupt_prompts(dataset):
    return [":"] * len(dataset)

def split_and_extract(input_string, delimiter=';'):
    """
    Splits a string by a given delimiter and returns the last part.

    Args:
        input_string (str): The string to be split.
        delimiter (str): The character or string to split by.

    Returns:
        str: The last part of the string, stripped of leading/trailing whitespace.
             Returns None if the input string is empty or invalid.
    """
    if not isinstance(input_string, str) or not input_string:
        return None

    # Split the string into a list of parts
    parts = input_string.split(delimiter)
    
    # The [-1] index gets the last element of the list.
    # .strip() removes any leading or trailing whitespace.
    last_part = parts[-1]
    
    return last_part

def generate_few_shot_prompts(d_name=None, model=None, 
    INPUT_LENGTH=None, OUTPUT_LENGTH=None, corruption_type="zs",
    EXP_SIZE=100, filter_correct=True, n_shot=5,
    output_correct_index=False, correct_index_list=None,
    batch_size=10, dataset_folder="../datasets/abstractive",
):
    
    with open(os.path.join(dataset_folder, f"{d_name}.json")) as f: 
        dataset = json.load(f)
    # print("\nd_name: ", d_name)

    if INPUT_LENGTH is not None and OUTPUT_LENGTH is not None:
        # Filter by length 
        dataset = filter_token_pairs_by_length(dataset, model, input_len=INPUT_LENGTH, output_len=OUTPUT_LENGTH) 

    if correct_index_list is not None: 
        dataset = [dataset[i] for i in correct_index_list]

    # Get prompts and answers 
    few_shot_prompts, answers, query_index = create_few_shot_prompts(
        dataset, n_shot = n_shot, delimiter = ";", q_bos=" ", a_bos=" ", qa_delimiter=":")
    if corruption_type in ["random_output", "relation"]:
        corrupt_few_shot_prompts = create_random_output_prompts(
            dataset, n_shot = 5, delimiter = ";", q_bos=" ", a_bos=" ", qa_delimiter=":")
    elif corruption_type =="shuffle_input":
        corrupt_few_shot_prompts, _, = create_corrupt_prompts_input_shuffle(
            dataset, n_shot = 5, delimiter = ";", q_bos=" ", a_bos=" ", qa_delimiter=":")
    # elif corruption_type =="abstract_causal":
    #     corrupt_few_shot_prompts, _ = create_corrupt_prompts_abstract_causal(
    #         task_dataset, n_shot = 5, delimiter = ";", q_bos=" ", a_bos=" ", qa_delimiter=":")
    elif corruption_type =="shuffle_output":
        corrupt_few_shot_prompts, _, _ = create_corrupt_prompts_output_shuffle(
            dataset, n_shot = 5, delimiter = ";", q_bos=" ", a_bos=" ", qa_delimiter=":")
    elif corruption_type =="target_output":
        corrupt_few_shot_prompts = create_target_output_prompts(
            dataset, n_shot = 5, delimiter = ";", q_bos=" ", a_bos=" ", qa_delimiter=":")
    elif corruption_type in ["random_query", "query"]:
        corrupt_few_shot_prompts = create_random_query_prompts(
            dataset, n_shot = 5, delimiter = ";", q_bos=" ", a_bos=" ", qa_delimiter=":")
    elif corruption_type == "minimal":
        corrupt_few_shot_prompts = create_minimal_corrupt_prompts(dataset)
    

    if filter_correct:
        # Filter out the prompts that are not correct #TODO: confirm the code 
        correct_index = return_correct_index(model, few_shot_prompts, answers, batch_size=batch_size)
        few_shot_prompts = [few_shot_prompts[i] for i in correct_index][:EXP_SIZE]
        answers = [answers[i] for i in correct_index][:EXP_SIZE] 
        if corruption_type == "zs":
            corrupt_few_shot_prompts = [split_and_extract(few_shot_prompts[i], delimiter=';') for i in range(len(few_shot_prompts))]
        else:
            corrupt_few_shot_prompts = [corrupt_few_shot_prompts[i] for i in correct_index][:EXP_SIZE]
        if output_correct_index:
            return few_shot_prompts, corrupt_few_shot_prompts, answers, correct_index
        else:
            return few_shot_prompts, corrupt_few_shot_prompts, answers
    else: 
        if corruption_type =="zs":
            corrupt_few_shot_prompts = [split_and_extract(few_shot_prompts[i], delimiter=';') for i in range(len(few_shot_prompts))]
        if output_correct_index:
            correct_index = return_correct_index(model, few_shot_prompts, answers, batch_size=batch_size)
            return few_shot_prompts, corrupt_few_shot_prompts, answers, correct_index
        else:
            return few_shot_prompts, corrupt_few_shot_prompts, answers

def generate_instruction_prompts(
    d_name=None, 
    INPUT_LENGTH=None, OUTPUT_LENGTH=None, EXP_SIZE=100, 
    model=None, instruction=None, instruction_corrupt=None,
    corruption_type="relation", filter_correct=True,
    output_correct_index=False, correct_index_list=None,
    dataset_folder:str="../datasets/abstractive",
):
    """
    TODO: generate corrupt query prompts based on optimal token length for each task
    Args:
        corruption_type: "relation" or "query" or "zs" or "minimal"
    """
    with open(os.path.join(dataset_folder, f"{d_name}.json")) as f: 
        dataset = json.load(f)
    # print("\nd_name: ", d_name)

    if INPUT_LENGTH is not None and OUTPUT_LENGTH is not None:
        # Filter by length 
        dataset = filter_token_pairs_by_length(dataset, model, input_len=INPUT_LENGTH, output_len=OUTPUT_LENGTH) 
    
    if correct_index_list is not None: 
        dataset = [dataset[i] for i in correct_index_list]

    # Get prompts and answers 
    inst_prompts, answers = create_instruction_prompts(dataset, instruction)
    if corruption_type == "relation":
        corrupt_inst_prompts, _ = create_instruction_prompts(dataset, instruction_corrupt)
    elif corruption_type == "random_query":
        corrupt_inst_prompts = create_instruction_prompts_corrupt_query(dataset, instruction_corrupt)
    elif corruption_type == "zs":
        corrupt_inst_prompts = create_zs_prompts(dataset)
    elif corruption_type == "minimal":
        corrupt_inst_prompts = create_minimal_corrupt_prompts(dataset)
    else:
        raise ValueError(f"Corruption type {corruption_type} not supported")
    
    # print("inst prompts: ", inst_prompts[0])
    # print("corrupt inst prompts: ", corrupt_inst_prompts[0])

    if filter_correct:
        # Filter out the prompts that are not correct #TODO: confirm the code 
        correct_index = return_correct_index(model, inst_prompts, answers)
        inst_prompts = [inst_prompts[i] for i in correct_index][:EXP_SIZE]
        answers = [answers[i] for i in correct_index][:EXP_SIZE] 
        corrupt_inst_prompts = [corrupt_inst_prompts[i] for i in correct_index][:EXP_SIZE]
        if output_correct_index:
            return inst_prompts, corrupt_inst_prompts, answers, correct_index
        else:
            return inst_prompts, corrupt_inst_prompts, answers
    else: 
        if output_correct_index:
            correct_index = return_correct_index(model, inst_prompts, answers)
            return inst_prompts, corrupt_inst_prompts, answers, correct_index
        else:
            return inst_prompts, corrupt_inst_prompts, answers      

def generate_EP_IP_prompts(
    d_name=None, 
    INPUT_LENGTH=None, OUTPUT_LENGTH=None, EXP_SIZE=100, 
    model=None, instruction=None, instruction_corrupt=None,
    IP_corruption_type="relation", filter_correct=True,
    EP_corruption_type="zs",
    output_correct_index=False,
    dataset_folder:str="../datasets/abstractive",
):
    """
    TODO: generate corrupt query prompts based on optimal token length for each task
    Args:
        IP_corruption_type: "relation" or "query" or "zs" or "minimal"
    """
    with open(os.path.join(dataset_folder, f"{d_name}.json")) as f: 
        dataset = json.load(f)
    # print("\nd_name: ", d_name)

    if INPUT_LENGTH is not None and OUTPUT_LENGTH is not None:
        # Filter by length 
        dataset = filter_token_pairs_by_length(dataset, model, input_len=INPUT_LENGTH, output_len=OUTPUT_LENGTH) 

    # Get INST prompts and answers 
    inst_prompts, inst_answers = create_instruction_prompts(dataset, instruction)
    if IP_corruption_type == "relation":
        corrupt_inst_prompts, _ = create_instruction_prompts(dataset, instruction_corrupt)
    elif IP_corruption_type == "random_query":
        corrupt_inst_prompts = create_instruction_prompts_corrupt_query(dataset, instruction_corrupt)
    elif IP_corruption_type == "zs":
        corrupt_inst_prompts = create_zs_prompts(dataset)
    elif IP_corruption_type == "minimal":
        corrupt_inst_prompts = create_minimal_corrupt_prompts(dataset)
    else:
        raise ValueError(f"Corruption type {IP_corruption_type} not supported")

    # Get ICL prompts and answers 
    few_shot_prompts, answers_EP, query_index = create_few_shot_prompts(
        dataset, n_shot = 5, delimiter = ";", q_bos=" ", a_bos=" ", qa_delimiter=":")
    if EP_corruption_type =="random_output":
        corrupt_few_shot_prompts = create_random_output_prompts(
            dataset, n_shot = 5, delimiter = ";", q_bos=" ", a_bos="", qa_delimiter=":")
    elif EP_corruption_type =="shuffle_input":
        corrupt_few_shot_prompts, _, = create_corrupt_prompts_input_shuffle(
            dataset, n_shot = 5, delimiter = ";", q_bos=" ", a_bos=" ", qa_delimiter=":")
    # elif corruption_type =="abstract_causal":
    #     corrupt_few_shot_prompts, _ = create_corrupt_prompts_abstract_causal(
    #         task_dataset, n_shot = 5, delimiter = ";", q_bos=" ", a_bos=" ", qa_delimiter=":")
    elif EP_corruption_type =="shuffle_output":
        corrupt_few_shot_prompts, _, _ = create_corrupt_prompts_output_shuffle(
            dataset, n_shot = 5, delimiter = ";", q_bos=" ", a_bos=" ", qa_delimiter=":")
    elif EP_corruption_type =="target_output":
        corrupt_few_shot_prompts = create_target_output_prompts(
            dataset, n_shot = 5, delimiter = ";", q_bos=" ", a_bos=" ", qa_delimiter=":")
    elif EP_corruption_type =="random_query":
        corrupt_few_shot_prompts = create_random_query_prompts(
            dataset, n_shot = 5, delimiter = ";", q_bos=" ", a_bos=" ", qa_delimiter=":")
    elif EP_corruption_type == "minimal":
        corrupt_few_shot_prompts = create_minimal_corrupt_prompts(dataset)
    elif EP_corruption_type == "zs":
        pass
    else:
        raise ValueError(f"Corruption type {EP_corruption_type} not supported")
    
    if filter_correct:
        correct_index_EP = return_correct_index(model, few_shot_prompts, answers_EP)
        inst_correct_index = return_correct_index(model, inst_prompts, inst_answers)
        correct_index_EP_IP = list(set(correct_index_EP) & set(inst_correct_index))

        few_shot_prompts = [few_shot_prompts[i] for i in correct_index_EP_IP][:EXP_SIZE]
        answers_EP = [answers_EP[i] for i in correct_index_EP_IP][:EXP_SIZE] 
        if EP_corruption_type == "zs":
            correct_query_index = [query_index[i] for i in correct_index_EP_IP]
            dataset = [dataset[i] for i in correct_query_index][:EXP_SIZE]
            corrupt_few_shot_prompts = create_zs_prompts(dataset)
        else:
            corrupt_few_shot_prompts = [corrupt_few_shot_prompts[i] for i in correct_index_EP_IP][:EXP_SIZE]

        inst_correct_index = return_correct_index(model, inst_prompts, inst_answers)
        inst_prompts = [inst_prompts[i] for i in inst_correct_index][:EXP_SIZE]
        inst_answers = [inst_answers[i] for i in inst_correct_index][:EXP_SIZE] 
        corrupt_inst_prompts = [corrupt_inst_prompts[i] for i in inst_correct_index][:EXP_SIZE]
    else: 
        if EP_corruption_type =="zs":
            dataset = [dataset[i] for i in query_index]
            corrupt_few_shot_prompts = create_zs_prompts(dataset)

    if output_correct_index:
        return inst_prompts, corrupt_inst_prompts, inst_answers, few_shot_prompts, corrupt_few_shot_prompts, answers_EP, correct_index_EP_IP
    else:
        return inst_prompts, corrupt_inst_prompts, inst_answers, few_shot_prompts, corrupt_few_shot_prompts, answers_EP

def create_instruction_prompts_from_index(raw_dataset, instruction, index_list):
    assert "{input}" in instruction
    dataset = [raw_dataset[int(i)] for i in index_list]
    prompts, answers = [], []
    for data in dataset:
        prompts.append(instruction.format(input=data["input"]))
        answers.append(" " + data["output"])
    return prompts, answers

def generate_instruction_prompts_from_index(
    d_name=None, index_list=None,
    instruction=None, instruction_corrupt=None,
    corruption_type="random_query", 
    dataset_folder:str="../datasets/abstractive",
):
    """
    TODO: generate corrupt query prompts based on optimal token length for each task
    Args:
        corruption_type: "relation" or "query" or "zs" or "minimal"
    """
    with open(os.path.join(dataset_folder, f"{d_name}.json")) as f: 
        dataset = json.load(f)

    # Get prompts and answers 
    prompts, answers = create_instruction_prompts_from_index(dataset, instruction, index_list)
    if corruption_type == "random_query":
        corrupt_prompts = [instruction_corrupt] * len(index_list)
    elif corruption_type == "zs":
        corrupt_prompts = create_zs_prompts_from_index(dataset, index_list)
    else:
        raise ValueError(f"Corruption type {corruption_type} not supported")
    return prompts, answers, corrupt_prompts

def create_zs_prompts_from_index(raw_dataset, index_list):
    prompts = [" " +raw_dataset[int(i)]["input"] + ":" for i in index_list]
    return prompts

def generate_few_shot_prompts_from_index(d_name=None, 
    corruption_type="random_query",
    index_list=None,
    dataset_folder:str="../datasets/abstractive",
): 
    with open(os.path.join(dataset_folder, f"{d_name}.json")) as f: 
        dataset = json.load(f)
   
    # Get prompts and answers 
    few_shot_prompts, answers = create_few_shot_prompts_from_index(
        dataset, n_shot = 5, delimiter = ";", q_bos=" ", a_bos=" ", qa_delimiter=":", index_list=index_list)

    if corruption_type == "random_query":
        corrupt_prompts = create_random_query_prompts_from_index(
            dataset, n_shot = 5, delimiter = ";", q_bos=" ", a_bos=" ", qa_delimiter=":", index_list=index_list)
    elif corruption_type == "zs":
        corrupt_prompts = create_zs_prompts_from_index(dataset, index_list)
    else:
        raise ValueError(f"Corruption type {corruption_type} not supported")

    return few_shot_prompts, answers, corrupt_prompts

def create_random_query_prompts_from_index(raw_data, n_shot, delimiter, q_bos, a_bos, qa_delimiter, index_list):

    prompts = []

    # For each possible last example
    for i in range(len(index_list)):
        index = index_list[i]
        if type(index) == str:
            index = int(index)
        context = ""
        for j in range(index - n_shot, index):
            example = raw_data[j]
            context += f"{q_bos}{example['input']}{qa_delimiter}{a_bos}{example['output']}{delimiter}"

        query = "<|reserved_special_token_1|>"
        #context += f"{q_bos}{query}{qa_delimiter}"
        context += f"{query}{qa_delimiter}"

        prompts.append(context)

    return prompts

def create_few_shot_prompts_from_index(raw_data, n_shot, delimiter, q_bos, a_bos, qa_delimiter, index_list):
    prompts = []
    answers = []

    # For each possible last example
    for i in range(len(index_list)):
        index = index_list[i]
        if type(index) == str:
            index = int(index)
        context = ""
        for j in range(index - n_shot, index):
            example = raw_data[j]
            context += f"{q_bos}{example['input']}{qa_delimiter}{a_bos}{example['output']}{delimiter}"

        query = raw_data[index]
        context += f"{q_bos}{query['input']}{qa_delimiter}"

        prompts.append(context)
        answers.append(a_bos + query["output"])

    return prompts, answers

def get_prompts_dict_from_top_execution_neuron_prompt_index(
    d_name=None, 
    prompt_corruption_type="zs",
    Exe_neuron_corruption_type="random_query",
    save_root=None, model_name=None,
    dataset_info=None,
    inst_corrupt_query_dict=None,
):
    """
    Get prompts for EP, IP for a task and prompts for another task as control 
    The indices of the prompts are the intersection of indices of the top execution neurons 
    """
    EP_path = os.path.join(save_root, model_name, d_name, "Heads", 
        "causal_mediation", "neuron_path_patching",)
    with open(os.path.join(EP_path, f"top_neurons_{Exe_neuron_corruption_type}_EP.json"), "r") as f:
        top_neurons_dict_EP = json.load(f)

    IP_path = os.path.join(save_root, model_name, d_name, "Heads", 
        "causal_mediation", "neuron_path_patching",)
    with open(os.path.join(IP_path, f"top_neurons_{Exe_neuron_corruption_type}_IP.json"), "r") as f:
        top_neurons_dict_IP = json.load(f)

    EP_index = list(top_neurons_dict_EP.keys())
    EP_index = [int(i) for i in EP_index]
    IP_index = list(top_neurons_dict_IP.keys())
    IP_index = [int(i) for i in IP_index]
    intersect_index = list(set(EP_index) & set(IP_index))
    print("len(intersect_index)", len(intersect_index)) # Get the prompts where both EP and IP have top neurons (both got right)

    EP_prompts, EP_answers, EP_corrupt_prompts = generate_few_shot_prompts_from_index(d_name=d_name, 
        corruption_type=prompt_corruption_type,
        index_list=intersect_index)
    print("EP_prompts", EP_prompts[:2])
    print("EP_answers", EP_answers[:2])
    print("EP_corrupt_prompts", EP_corrupt_prompts[:2])

    IP_prompts, IP_answers, IP_corrupt_prompts = generate_instruction_prompts_from_index(
        d_name=d_name, index_list=intersect_index,
        instruction=dataset_info[d_name]["INST_prompt"], 
        corruption_type=prompt_corruption_type,
        instruction_corrupt=inst_corrupt_query_dict[d_name],
        )
    print("IP_prompts", IP_prompts[:2])
    print("IP_answers", IP_answers[:2])
    print("IP_corrupt_prompts", IP_corrupt_prompts[:2])

    # Control
    if d_name == "antonym":
        d_control_name = "english-german"
    else:
        d_control_name = "english-german"
    IP_prompts_control, _, _ = generate_instruction_prompts_from_index(
        d_name=d_control_name, index_list=intersect_index,
        instruction=dataset_info[d_control_name]["INST_prompt"], 
        #instruction_corrupt=inst_corrupt_query_dict[d_control_name],
        corruption_type=prompt_corruption_type)
    print(len(IP_prompts_control))

    print("IP_prompts_control", IP_prompts_control[:2])

    prompts_dict = {
        "EP_prompts": EP_prompts,
        "EP_answers": EP_answers,
        "EP_corrupt_prompts": EP_corrupt_prompts,
        "IP_prompts": IP_prompts,
        "IP_answers": IP_answers,
        "IP_corrupt_prompts": IP_corrupt_prompts,
        "IP_prompts_control": IP_prompts_control,
        "intersect_index": intersect_index,
    }

    return prompts_dict

def get_prompt_dict_small_large(model: LanguageModel,
    d_name: str = None, 
    batch_size: int=20, 
    return_pred_tokens: bool=False,
    instruction_small: str=0,
    instruction_large: str=0,
    n_shot_small: int=None,
    n_shot_large: int=None,
    prompt_type: str=None,
    dataset_folder:str="../datasets/abstractive",
):
    """
    Run through all the prompts in the dataset. Note: Index are the index relative to the whole dataset.
    small: prompts with small accuracy
    large: prompts with large accuracy
    prompt_type: "IP" or "EP"
    """
    with open(os.path.join(dataset_folder, f"{d_name}.json")) as f: 
        dataset = json.load(f)

    # Get prompts 
    if prompt_type == "IP":
        prompts_small, answers_small = create_instruction_prompts(
            dataset, instruction_small,
        )
    elif prompt_type == "EP":
        prompts_small, answers_small, _ = create_few_shot_prompts(
            dataset, n_shot = n_shot_small, delimiter = ";", q_bos=" ", a_bos=" ", qa_delimiter=":"
        )
    prompt_dict_small = check_correctness(model=model, prompts=prompts_small, answers=answers_small, 
        batch_size=batch_size, return_pred_tokens=return_pred_tokens, return_answer_tokens=True
    )
    if prompt_type == "IP":
        prompts_large, answers_large = create_instruction_prompts(
            dataset, instruction_large,
        )
    elif prompt_type == "EP":
        prompts_large, answers_large, _ = create_few_shot_prompts(
            dataset, n_shot = n_shot_large, delimiter = ";", q_bos=" ", a_bos=" ", qa_delimiter=":"
        )
    prompt_dict_large = check_correctness(model=model, prompts=prompts_large, answers=answers_large, 
        batch_size=batch_size, return_pred_tokens=return_pred_tokens, return_answer_tokens=False)
    
    correct_index_small = prompt_dict_small['correct_index']
    prompt_tokens_small = prompt_dict_small['prompt_tokens']
    correct_index_large = prompt_dict_large['correct_index']
    prompt_tokens_large = prompt_dict_large['prompt_tokens']
    answer_tokens = prompt_dict_small['answer_tokens']

    # Check the overlap between correct_index 
    assert prompt_tokens_small.shape[0] == prompt_tokens_large.shape[0]
    both_correct_index = set(correct_index_small) & set(correct_index_large)
    both_correct_index = list(sorted(both_correct_index))

    # Only correct in small 
    only_small_correct_index = set(correct_index_small) - set(correct_index_large)
    only_small_correct_index = list(sorted(only_small_correct_index))

    # Only correct in large 
    only_large_correct_index = set(correct_index_large) - set(correct_index_small)
    only_large_correct_index = list(sorted(only_large_correct_index))

    prompt_dict = {
        "prompt_tokens_small": prompt_tokens_small,
        "prompt_tokens_large": prompt_tokens_large,
        "answer_tokens": answer_tokens,
        "correct_index_small": correct_index_small,
        "correct_index_large": correct_index_large,
        "only_small_correct_index": only_small_correct_index,
        "only_large_correct_index": only_large_correct_index,
        "both_correct_index": both_correct_index,
        "num_total_prompts": len(dataset),
        "small_acc": len(correct_index_small) / len(dataset),
        "large_acc": len(correct_index_large) / len(dataset),
    }
    if return_pred_tokens:
        prompt_dict["pred_tokens_small"] = prompt_dict_small['pred_tokens']
        prompt_dict["pred_tokens_large"] = prompt_dict_large['pred_tokens']
    return prompt_dict


def get_prompt_dict(model: LanguageModel,
    d_name: str = None, 
    batch_size: int=20, 
    return_pred_tokens: bool=False,
    instruction: str=0,
    n_shot: int=None,
    prompt_type: str=None,
    dataset_folder:str="../datasets/abstractive",
):
    """
    Run through all the prompts in the dataset. Note: Index are the index relative to the whole dataset.
    prompt_type: "IP" or "EP"
    """
    with open(os.path.join(dataset_folder, f"{d_name}.json")) as f: 
        dataset = json.load(f)

    # Get prompts 
    if prompt_type == "IP":
        prompts, answers = create_instruction_prompts(
            dataset, instruction,
        )
    elif prompt_type == "EP":
        prompts, answers, _ = create_few_shot_prompts(
            dataset, n_shot = n_shot, delimiter = ";", q_bos=" ", a_bos=" ", qa_delimiter=":"
        )
    prompt_dict= check_correctness(model=model, prompts=prompts, answers=answers, 
        batch_size=batch_size, return_pred_tokens=return_pred_tokens, return_answer_tokens=True
    )
    
    correct_index = prompt_dict['correct_index']
    prompt_tokens = prompt_dict['prompt_tokens']
    answer_tokens = prompt_dict['answer_tokens']

    prompt_dict = {
        "prompt_tokens": prompt_tokens,
        "answer_tokens": answer_tokens,
        "correct_index": correct_index,
        "num_total_prompts": len(dataset),
        "acc": len(correct_index) / len(dataset),
    }
    if return_pred_tokens:
        prompt_dict["pred_tokens"] = prompt_dict['pred_tokens']

    return prompt_dict

def get_prompt_token(
    model=None, d_name:str=None,
    prompt_type:str=None, 
    prompt_index:int=None, 
    instruction_dict:dict=None,    
    return_answer_tokens:bool=False,
    corrupt_type:str=None,
    inst_random_relation_dict:dict=None,
    dataset_folder:str="../datasets/abstractive",
): 
    """
    Args:
        model: LanguageModel or TransformerLens
    
    """
    token_dict = {}
    dataset = json.load(open(os.path.join(dataset_folder, f"{d_name}.json")))
    if prompt_type == "EP":
        if corrupt_type is None:
            prompts, answers, _ = create_few_shot_prompts(
                dataset, n_shot = prompt_index, delimiter = ";", q_bos=" ", a_bos=" ", qa_delimiter=":"
            )
        elif corrupt_type == "random_query":
            prompts = create_random_query_prompts(
            dataset, n_shot = prompt_index, delimiter = ";", q_bos=" ", a_bos=" ", qa_delimiter=":")
        else:
            raise ValueError(f"corrupt_type {corrupt_type} not supported")
    elif prompt_type == "IP": 
        if corrupt_type is None:
            prompts, answers = create_instruction_prompts(dataset, 
                instruction_dict[d_name][str(prompt_index)],
            )
        elif corrupt_type == "random_query":
            prompts = create_instruction_prompts_corrupt_query(
                dataset, 
                instruction=instruction_dict[d_name][str(prompt_index)])
        elif corrupt_type == "random_relation":
            prompts, _ = create_instruction_prompts(dataset, 
                inst_random_relation_dict[d_name])
        else:
            raise ValueError(f"corrupt_type {corrupt_type} not supported")
    elif prompt_type == "zs":
        prompts, answers = create_zs_prompts(dataset, return_answers=True)
    else:
        raise ValueError(f"prompt_type {prompt_type} not supported")
    
    # print("prompts", prompts[0])
    # if return_answer_tokens:
    #     print("answers", answers[0])

    prompt_tokens = model.tokenizer(
        prompts, 
        padding=True, 
        padding_side="left", 
        return_tensors="pt"
    )["input_ids"]
    token_dict["prompt_tokens"] = prompt_tokens

    if return_answer_tokens:
        answer_tokens = model.tokenizer(
            answers,
            add_special_tokens=False,
            padding=True,
            padding_side="right",
            return_tensors="pt",
        )["input_ids"][:, 0]
        token_dict["answer_tokens"] = answer_tokens
        
    return token_dict

def get_index_from_behavior_dict(
    save_root:str=None, model_name:str=None,
    prompt_type:str=None, prompt_index:int=None,
    d_name:str=None,
    dataset_size:int=None,
    incorrect_or_correct:str="incorrect",
):
    if prompt_type == "EP":
        file_name = "EP_vary_n_shot_behavior.json"
    elif prompt_type == "IP":
        file_name = "IP_vary_n_inst_behavior.json"
    else:
        raise ValueError(f"prompt_type {prompt_type} not supported")
    with open(os.path.join(save_root, model_name,
        "across_tasks", "Behavior", file_name), "r"
    ) as f:
        behavior_dict_tasks = json.load(f) 
    if incorrect_or_correct == "correct":
        index_list = behavior_dict_tasks[d_name][str(prompt_index)]['correct_index']
    elif incorrect_or_correct == "incorrect":
        index_list = sorted(list(set(range(dataset_size)) - set(behavior_dict_tasks[d_name][str(prompt_index)]['correct_index'])))
    else:
        raise ValueError(f"incorrect_or_correct {incorrect_or_correct} not supported")
    return index_list

def check_correctness(
    model: LanguageModel, prompts: list, answers: list, 
    batch_size=10, remote=False, query_index=None,
    return_pred_tokens=False, return_answer_tokens=False
):
    correct_index = []

    prompt_tokens = model.tokenizer(
        prompts, 
        padding=True, 
        padding_side="left", 
        return_tensors="pt"
    )["input_ids"]
    answer_tokens = model.tokenizer(
        answers,
        add_special_tokens=False,
        padding=True,
        padding_side="right",
        return_tensors="pt",
    )["input_ids"][:, 0]
    
    pred_tokens = []
    #for i in tqdm(range(0, len(prompts), batch_size)):
    for i in range(0, len(prompts), batch_size):
        batch_prompt_tokens = prompt_tokens[i : i + batch_size]
        batch_answer_tokens = answer_tokens[i : i + batch_size]
        # batch_pred_tokens = (
        #     model.trace(batch_prompt_tokens, trace=False, remote=remote).logits[:, -1, :].argmax(dim=-1).cpu()
        # )
        with model.trace(batch_prompt_tokens, remote=remote) as tracer:
            batch_pred_tokens_proxy = model.output.logits[:, -1, :].argmax(dim=-1).save()
        batch_pred_tokens = batch_pred_tokens_proxy.cpu()

        if return_pred_tokens:
            pred_tokens.extend(batch_pred_tokens.tolist())

        batch_correct = torch.where(batch_answer_tokens == batch_pred_tokens)[0].tolist()
        correct_index += [idx + i for idx in batch_correct]

    return_dict = {
        "correct_index": correct_index,
        "prompt_tokens": prompt_tokens,
    }
    #print(f"Accuracy: {len(correct_index)/len(prompts)} ({len(correct_index)}/{len(prompts)})")
    if return_pred_tokens:
        return_dict["pred_tokens"] = pred_tokens
    if return_answer_tokens:
        return_dict["answer_tokens"] = answer_tokens
    return return_dict