import functools
from collections import defaultdict
import argparse
import json
import os
import random

import torch
from nnsight import LanguageModel
from torch import Tensor
from tqdm.auto import tqdm

from wrapper import ModelAccessor, get_accessor_config, get_model_specs
from prompt_utils import generate_few_shot_prompts, create_few_shot_prompts, create_zs_prompts, create_corrupt_prompts_input_shuffle, create_corrupt_prompts_output_shuffle


def _unwrap(saved):
    """nnsight >= 0.4 returns the saved tensor directly after the trace exits;
    older versions returned a proxy with `.value`. Accept either."""
    return saved.value if hasattr(saved, "value") else saved


def _model_device(model) -> torch.device:
    """Device of the model's weights. Tokenizers always return CPU tensors, and
    nnsight does not move raw token tensors for you, so inputs must be moved here."""
    try:
        device = next(model.parameters()).device
    except (StopIteration, AttributeError, TypeError):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # remote (NDIF) runs keep undispatched meta weights locally; inputs stay on CPU
    return torch.device("cpu") if device.type == "meta" else device

def find_earliest_receiver(receiver_list: list[tuple[int, int]]) -> tuple[int, int]:
    """
    Finds the computationally earliest component in a list of receivers.
    Order: Layer index first, then Attention Heads < MLP within a layer.
    """

    def compare_receivers(item1: tuple[int, int], item2: tuple[int, int]) -> int:
        layer1, comp1 = item1
        layer2, comp2 = item2
        if layer1 < layer2:
            return -1
        elif layer1 > layer2:
            return 1
        else:
            if comp1 >= 0 and comp2 == -1:
                return -1  # Attn < MLP
            elif comp1 == -1 and comp2 >= 0:
                return 1  # MLP > Attn
            else:
                return 0  # Order between heads doesn't matter

    # Handle empty list case
    if not receiver_list:
        raise ValueError("receiver_list cannot be empty for find_earliest_receiver")
    return sorted(receiver_list, key=functools.cmp_to_key(compare_receivers))[0]


def path_patch_sender_to_receivers(
    model: LanguageModel,
    clean_tokens: Tensor,
    corrupt_tokens: Tensor,
    answer_tokens: Tensor,
    receiver_list: list[tuple[int, int]],
    min_layer: int,
    min_head_or_mlp: int,
    max_sender_layer: int,
    result_shape: tuple[int, int],
    remote: bool = True,
    sender_pos: list[int] = [-1],
    receiver_pos: list[int] = [-1],
    freeze_pos: list[int] = [-1],
    # batch_index: int = 0 # TODO: double check before removing
) -> Tensor:
    """
    Core implementation: Sender -> set of Receivers
    Patch Run 1: Freezes pre-sender & intermediate layers, patches sender correctly respecting causality of nnsight. #TODO: double check
    Patch Run 2: Patches receivers. 

    Args:
        model: Language model.
        clean_tokens: Tokenized clean prompts for the batch.
        corrupt_tokens: Tokenized corrupt prompts for the batch.
        answer_tokens: Tokenized answer tokens for the batch.
        receiver_list: list of receiver components [(layer_idx, head_or_mlp_idx), ...].
        min_layer, min_head_or_mlp: Info about the earliest receiver.
        max_sender_layer: Highest layer index containing senders.
        result_shape: Expected shape of the final result tensor.
        remote: Whether to run model remotely.
        sender_pos: Position(s) to patch activations for the sender and components before the sender.
        receiver_pos: Position(s) to patch activations for the receivers.
        freeze_pos: Position(s) to freeze activations during patch run 1 
    Returns:
        A single tensor (on CPU) containing the normalized effect of each sender
        mediated through the receiver set for this batch.
    """
    #  Setup
    # Every patched run (Run 1 and Run 2) executes on corrupt_tokens, so positions
    # must index the CORRUPT sequence. Clean and corrupt can differ in length
    # (e.g. 5-shot clean vs zero-shot corrupt), so only negative indices like [-1]
    # are well-defined on both; "all_pos" is only valid when the lengths match.
    all_pos = list(range(corrupt_tokens.shape[1]))
    uses_all_pos = "all_pos" in (sender_pos, receiver_pos, freeze_pos)
    if uses_all_pos and clean_tokens.shape[1] != corrupt_tokens.shape[1]:
        raise ValueError(
            f"'all_pos' needs clean and corrupt prompts of equal length, got "
            f"{clean_tokens.shape[1]} vs {corrupt_tokens.shape[1]} tokens. "
            "Use explicit negative positions like [-1] instead."
        )
    if sender_pos == "all_pos": 
        sender_pos = all_pos
    if receiver_pos == "all_pos": 
        receiver_pos = all_pos
    if freeze_pos == "all_pos": 
        freeze_pos = all_pos

    spec = get_model_specs(model)
    n_layers, n_heads, d_model, d_head = spec["n_layers"], spec["n_heads"], spec["d_model"], spec["d_head"]
    accessor_config = get_accessor_config(model)
    accessor = ModelAccessor(model, accessor_config)
    batch_size = len(clean_tokens)

    #  1. Calculate Baseline Logits & Difference
    with torch.no_grad():
        clean_logits_trace = model.trace(clean_tokens, trace=False, remote=remote)
        corrupt_logits_trace = model.trace(corrupt_tokens, trace=False, remote=remote)
        clean_answer_logits = clean_logits_trace["logits"][torch.arange(batch_size), -1, answer_tokens]
        corrupt_answer_logits = corrupt_logits_trace["logits"][torch.arange(batch_size), -1, answer_tokens]

    # This is usually not happenning, keep it as a warning tho.
    baseline_diff = (clean_answer_logits - corrupt_answer_logits).mean()
    epsilon = 1e-8
    if torch.abs(baseline_diff) < epsilon:
        print(f"Warning: Baseline difference {baseline_diff.item():.4f} is close to zero.")
        baseline_diff = torch.sign(baseline_diff) * max(
            torch.abs(baseline_diff), torch.tensor(epsilon, device=baseline_diff.device)
        )

    #  2. Cache Clean Sender & Corrupt Activations (as Proxies)
    clean_attn_proxies = {}  # Stores attn_input from CLEAN run for SENDERS
    clean_mlp_proxies = {}  # Stores mlp_output from CLEAN run for SENDERS
    corrupt_attn_proxies = {}  # Stores attn_input from CORRUPT run for FREEZING
    corrupt_mlp_proxies = {}  # Stores mlp_output from CORRUPT run for FREEZING

    sender_layers_to_cache = set(range(max_sender_layer + 1))
    # Determine which components are actually needed for clean cache (senders)
    valid_sender_components = defaultdict(set)
    for layer in sender_layers_to_cache:
        is_max_sender_layer_mlp_receiver = layer == max_sender_layer and min_head_or_mlp == -1
        valid_sender_components[layer].update(
            range(n_heads)
        )  # Heads are always potential senders up to max_sender_layer
        if not is_max_sender_layer_mlp_receiver:
            valid_sender_components[layer].add(n_heads)

    #  Cache Clean Activations
    with accessor.trace(remote=remote) as tracer_clean_cache:
        with tracer_clean_cache.invoke(clean_tokens):
            for layer in sender_layers_to_cache:
                is_sender_layer_attn = any(idx < n_heads for idx in valid_sender_components.get(layer, set()))
                is_sender_layer_mlp = n_heads in valid_sender_components.get(layer, set())

                # Adding a bunch of try-catches for debugging purpose.
                if is_sender_layer_attn:
                    try:
                        clean_attn_proxies[layer] = accessor.layers[layer].attention.output.unwrap().input[:, sender_pos].save()
                    except Exception as e:
                        print(f"Clean Cache Proxy Save Warning: Attn layer {layer}. {e}")
                if is_sender_layer_mlp:
                    try:
                        clean_mlp_proxies[layer] = accessor.layers[layer].mlp.unwrap().output[:, sender_pos].save()
                    except Exception as e:
                        print(f"Clean Cache Proxy Save Warning: MLP layer {layer}. {e}")

    #  Cache Corrupt Activations
    with accessor.trace(remote=remote) as tracer_corrupt_cache:
        with tracer_corrupt_cache.invoke(corrupt_tokens):
            for layer in range(n_layers):  # Cache ALL layers and ALL positions needed for freezing #TODO: double check to improve efficiency 
                try:
                    corrupt_attn_proxies[layer] = accessor.layers[layer].attention.output.unwrap().input[:, :].save()
                except Exception as e:
                    print(f"Corrupt Cache Proxy Save Warning: Attn layer {layer}. {e}")
                try:
                    corrupt_mlp_proxies[layer] = accessor.layers[layer].mlp.unwrap().output[:, :].save()
                except Exception as e:
                    print(f"Corrupt Cache Proxy Save Warning: MLP layer {layer}. {e}")

    #  3. Path Patching Loop
    results_tensor = torch.zeros(result_shape, device="cpu")
    receiver_comp_indices = {
        rec: (n_heads if rec[1] == -1 else rec[1]) for rec in receiver_list
    }  # Precompute for convenience

    # TODO: double check before deleting them
    receiver_set = set(receiver_list)  # Use set for faster lookup
    max_receiver_layer = max(rec[0] for rec in receiver_list) if receiver_list else -1

    for sender_layer in tqdm(range(max_sender_layer + 1), desc="Patching Senders", leave=False):
        for sender_comp_idx in range(n_heads + 1):
            #  Check: Skip if this component is not a valid sender
            if sender_comp_idx not in valid_sender_components.get(sender_layer, set()):
                continue
            is_sender_attn = sender_comp_idx < n_heads
            # Check if clean proxy exists (should be guaranteed by above check)
            if is_sender_attn and sender_layer not in clean_attn_proxies:
                continue
            if not is_sender_attn and sender_layer not in clean_mlp_proxies:
                continue

            #  Run 1: Freeze PRE-SENDER & INTERMEDIATE layers, Patch Sender -> Collect Patched Receiver Activations
            patched_receiver_activations_proxies = {}  # Store saved PROXIES for receivers
            run1_success = False
            with accessor.trace(remote=remote) as tracer_run1:
                with tracer_run1.invoke(corrupt_tokens):
                    #  Step 1: Freeze layers BEFORE sender at freeze_pos
                    for current_layer in range(sender_layer):
                        if current_layer in corrupt_attn_proxies:
                            accessor.layers[current_layer].attention.output.unwrap().input[:, freeze_pos] = (
                                _unwrap(corrupt_attn_proxies[current_layer])[:, freeze_pos].clone()
                            )
                        if current_layer in corrupt_mlp_proxies:
                            accessor.layers[current_layer].mlp.unwrap().output[:, freeze_pos] = _unwrap(corrupt_mlp_proxies[current_layer])[:, freeze_pos].clone()

                    #  Step 2: Intervene at the SENDER layer
                    mlp_idx_sender = n_heads
                    if is_sender_attn:  # Sender is Attention Head
                        # Freeze-then-patch Attention Input
                        if sender_layer in corrupt_attn_proxies and sender_layer in clean_attn_proxies:
                            current_attn_input = accessor.layers[sender_layer].attention.output.unwrap().input[...]
                            current_attn_input[:, freeze_pos] = _unwrap(corrupt_attn_proxies[sender_layer])[:, freeze_pos].clone()  # Freeze all heads @ freeze_pos
                            current_attn_reshaped = current_attn_input.reshape(batch_size, len(all_pos), n_heads, d_head)
                            clean_attn_tensor = _unwrap(clean_attn_proxies[sender_layer]).reshape(
                                batch_size, len(sender_pos), n_heads, d_head
                            )
                            current_attn_reshaped[:, sender_pos, sender_comp_idx, :] = clean_attn_tensor[
                                :, :, sender_comp_idx, :
                            ].clone()  # Patch sender head
                        # TODO: Need to double check.
                        # *** DO NOT FREEZE MLP OUTPUT in sender layer if sender is ATTN ***

                    else:  # Sender is MLP
                        # Freeze Attention Input first (causally before MLP)
                        if sender_layer in corrupt_attn_proxies:
                            accessor.layers[sender_layer].attention.output.unwrap().input[:, freeze_pos] = (
                                _unwrap(corrupt_attn_proxies[sender_layer])[:, freeze_pos].clone()
                            )
                        # Freeze-then-patch MLP Output
                        if sender_layer in corrupt_mlp_proxies and sender_layer in clean_mlp_proxies:
                            current_mlp_output = accessor.layers[sender_layer].mlp.unwrap().output[:, :]
                            current_mlp_output[:, freeze_pos] = _unwrap(corrupt_mlp_proxies[sender_layer])[:, freeze_pos].clone()  # Freeze
                            current_mlp_output[:, sender_pos] = _unwrap(clean_mlp_proxies[sender_layer]).clone()  # Patch

                    #  Step 3: Freeze INTERMEDIATE layers (after sender, before min_layer)
                    for current_layer in range(sender_layer + 1, min_layer):
                        if current_layer in corrupt_attn_proxies:
                            accessor.layers[current_layer].attention.output.unwrap().input[:, freeze_pos] = (
                                _unwrap(corrupt_attn_proxies[current_layer])[:, freeze_pos].clone()
                            )
                        if current_layer in corrupt_mlp_proxies:
                            accessor.layers[current_layer].mlp.unwrap().output[:, freeze_pos] = _unwrap(corrupt_mlp_proxies[current_layer])[:, freeze_pos].clone()

                    #  Step 4: Handle freezing ATTN heads at min_layer if earliest receiver is MLP
                    # This needs to happen *only if* min_layer was not the sender layer AND earliest receiver is MLP
                    if min_head_or_mlp == -1 and min_layer != sender_layer and min_layer in corrupt_attn_proxies:
                        accessor.layers[min_layer].attention.output.unwrap().input[:, freeze_pos] = (
                            _unwrap(corrupt_attn_proxies[min_layer])[:, freeze_pos].clone()
                        )

                    #  Step 5: Let computation flow naturally for layers >= min_layer (unless ATTN frozen just above)

                    #  Step 6: Save the activation proxy of *each receiver* component
                    for receiver_tuple in receiver_list:
                        rec_layer, rec_head_or_mlp = receiver_tuple
                        rec_comp_idx = receiver_comp_indices[receiver_tuple]
                        if rec_comp_idx < n_heads:
                            # Save the resulting attention input state
                            rec_attn_out_input = accessor.layers[rec_layer].attention.output.unwrap().input[:, receiver_pos]
                            rec_attn_out_reshaped = rec_attn_out_input.reshape(batch_size, len(receiver_pos), n_heads, d_head)
                            patched_receiver_activations_proxies[receiver_tuple] = rec_attn_out_reshaped[
                                :, :, rec_comp_idx, :
                            ].save()
                        else:
                            # Save the resulting MLP output state
                            patched_receiver_activations_proxies[receiver_tuple] = (
                                accessor.layers[rec_layer].mlp.unwrap().output[:, receiver_pos].save()
                            )
            run1_success = True  # Assume success if trace completes without nnsight error

            # An extra check to fail early.
            if not run1_success or len(patched_receiver_activations_proxies) != len(receiver_list):
                print(f"Warning: Run 1 incomplete for sender ({sender_layer},{sender_comp_idx}). Skipping Run 2.")
                continue

            #  Run 2: Patch All Receivers, Get Final Logits
            intervened_logits_obj = None
            run2_success = False
            with accessor.trace(remote=remote) as tracer_run2:
                with tracer_run2.invoke(corrupt_tokens):
                    # Iterate through ALL layers to patch at receiver_pos
                    for current_layer in range(n_layers):
                        mlp_idx = n_heads
                        #  Patch Attention Heads at receiver_pos
                        if current_layer in corrupt_attn_proxies:
                            current_attn_input = accessor.layers[current_layer].attention.output.unwrap().input[...]
                            current_attn_reshaped = current_attn_input.reshape(batch_size, len(all_pos), n_heads, d_head)
                            for head_idx in range(n_heads):
                                receiver_tuple = (current_layer, head_idx)
                                if receiver_tuple in patched_receiver_activations_proxies:
                                    receiver_proxy = patched_receiver_activations_proxies[receiver_tuple]
                                    if receiver_proxy is not None:
                                        current_attn_reshaped[:, receiver_pos, head_idx, :] = _unwrap(receiver_proxy).clone()

                        #  Patch MLP
                        if current_layer in corrupt_mlp_proxies:
                            current_mlp_output = accessor.layers[current_layer].mlp.unwrap().output[...]
                            receiver_tuple = (current_layer, -1)
                            if receiver_tuple in patched_receiver_activations_proxies:
                                receiver_proxy = patched_receiver_activations_proxies[receiver_tuple]
                                if receiver_proxy is not None:
                                    current_mlp_output[:, receiver_pos] = _unwrap(receiver_proxy).clone()

                    # Save the final logits
                    intervened_logits_obj = (
                        accessor.lm_head.unwrap().output[torch.arange(batch_size), -1, answer_tokens].save()
                    )
            run2_success = True  # Assume success if trace completes without nnsight error

            #  Again, fail early
            if not run2_success or intervened_logits_obj is None:
                print(
                    f"Skipping calculation for sender ({sender_layer},{sender_comp_idx}) due to Run 2 failure or save error."
                )
                continue

            # Extract tensor from final logits, and I just don't get how .value works in nnsight
            if hasattr(intervened_logits_obj, "value"):
                intervened_logits_val = intervened_logits_obj.value
            elif isinstance(intervened_logits_obj, torch.Tensor):
                intervened_logits_val = intervened_logits_obj
            else:
                print(
                    f"Unexpected type for intervened_logits_obj: {type(intervened_logits_obj)}. Skipping calculation."
                )
                continue

            # Perform calculation
            corrupt_answer_logits_dev = corrupt_answer_logits.to(intervened_logits_val.device)
            baseline_diff_dev = baseline_diff.to(intervened_logits_val.device)
            intervention_diff = intervened_logits_val.mean() - corrupt_answer_logits_dev.mean()
            # Add small epsilon to denominator to prevent division by zero if baseline_diff is exactly zero
            normalized_effect = (intervention_diff / (baseline_diff_dev + 1e-12)).item()

            results_tensor[sender_layer, sender_comp_idx] = normalized_effect

    return results_tensor


def path_patch_sender_to_receivers_batch(
    model: LanguageModel,
    clean_prompts: list[str],
    corrupt_prompts: list[str],
    answers: list[str],
    receiver_list: list[tuple[int, int]],
    batch_size: int = 8,
    remote: bool = True,
    sender_pos: list[int] = [-1],
    receiver_pos: list[int] = [-1],
    freeze_pos: list[int] = [-1],
) -> torch.Tensor:
    """
    Batched Path Patching: Sender -> Receiver Set
    Patch Run 1: Freezes pre-sender & intermediate layers, patches sender correctly respecting causality of nnsight. #TODO: double check
    Patch Run 2: Patches receivers. 

    Calculates the normalized effect of restoring a single sender's clean activation
    on the final logits, considering only pathways that pass collectively through
    a specified *set* of receiver components. Explicitly freezes non-patched components.

    Args:
        model: Language model to analyze.
        clean_prompts: list of original unmodified prompts.
        corrupt_prompts: list of modified/corrupted prompts.
        answers: list of expected answer strings (only first token is used).
        receiver_list: list of receiver components [(layer_idx, head_or_mlp_idx), ...]
                       defining the mediating set. Use -1 for MLP index.
        batch_size: Size of each processing batch.
        remote: Whether to run model remotely.
        sender_pos: Position(s) to patch activations for the sender and components before the sender.
        receiver_pos: Position(s) to patch activations for the receivers.
        freeze_pos: Position(s) to freeze activations during patch run 1 
    Returns:
        A single tensor containing the normalized contribution scores for each
        valid sender component, mediated by the receiver set.
        Shape depends on the earliest receiver in the set.
    """
    # Input Validation
    n_samples = len(clean_prompts)
    if not (n_samples == len(corrupt_prompts) == len(answers)):
        raise ValueError("Input lists (clean_prompts, corrupt_prompts, answers) must have the same length.")
    if not receiver_list:
        raise ValueError("receiver_list cannot be empty.")

    #  Setup
    spec = get_model_specs(model)
    n_layers, n_heads = spec["n_layers"], spec["n_heads"]

    #  Tokenization
    #  NOTE: Whether the tokenizer prepends <BOS> is not specified here.
    tokenizer_kwargs = {"padding": True, "return_tensors": "pt"}
    clean_tokens = model.tokenizer(clean_prompts, padding_side="left", **tokenizer_kwargs)["input_ids"]
    corrupt_tokens = model.tokenizer(corrupt_prompts, padding_side="left", **tokenizer_kwargs)["input_ids"]
    answer_tokens = model.tokenizer(answers, padding_side="right", add_special_tokens=False, **tokenizer_kwargs)[
        "input_ids"][:, 0]

    device = _model_device(model)
    clean_tokens = clean_tokens.to(device)
    corrupt_tokens = corrupt_tokens.to(device)
    answer_tokens = answer_tokens.to(device)

    #  Determine Sender Range & Result Shape
    min_layer, min_head_or_mlp = find_earliest_receiver(receiver_list)
    if min_head_or_mlp >= 0:  # Earliest receiver is an Attention Head
        max_sender_layer = min_layer - 1
        result_shape = (min_layer, n_heads + 1)
    else:  # Earliest receiver is an MLP
        max_sender_layer = min_layer
        result_shape = (min_layer + 1, n_heads + 1)

    #  Initialize Results
    all_results = torch.zeros(result_shape, device="cpu")

    #  Batch Processing
    for i in tqdm(range(0, n_samples, batch_size), desc="Processing batches"):
        batch_start = i
        batch_end = min(i + batch_size, n_samples)
        current_batch_size = batch_end - batch_start
        clean_batch_tokens = clean_tokens[batch_start:batch_end]
        corrupt_batch_tokens = corrupt_tokens[batch_start:batch_end]
        answer_batch_tokens = answer_tokens[batch_start:batch_end]

        # Call the core function
        batch_results_tensor = path_patch_sender_to_receivers(
            model=model,
            clean_tokens=clean_batch_tokens,
            corrupt_tokens=corrupt_batch_tokens,
            answer_tokens=answer_batch_tokens,
            receiver_list=receiver_list,
            min_layer=min_layer,
            min_head_or_mlp=min_head_or_mlp,
            max_sender_layer=max_sender_layer,
            result_shape=result_shape,
            remote=remote,
            sender_pos=sender_pos,
            receiver_pos=receiver_pos,
            freeze_pos=freeze_pos,
        )
        # Accumulate results
        weight = current_batch_size / n_samples
        all_results += batch_results_tensor.cpu() * weight  # Ensure results are on CPU
        # if torch.cuda.is_available():
        #     torch.cuda.empty_cache()
    return all_results

def path_patch_sender_to_receiver(
    model: LanguageModel,
    clean_tokens: Tensor,
    corrupt_tokens: Tensor,
    answer_tokens: Tensor,
    receiver: tuple[int, int],
    min_layer: int,
    min_head_or_mlp: int,
    max_sender_layer: int,
    result_shape: tuple[int, int],
    remote: bool = True,
    sender_pos: list[int] = [-1],
    receiver_pos: list[int] = [-1],
    freeze_pos: list[int] = [-1],
) -> Tensor:
    """
    Core implementation: Sender -> ONE Receiver. min_layer/min_head_or_mlp/
    max_sender_layer/result_shape come from the EARLIEST receiver across the
    WHOLE identified LTH list (computed once, outside, shared across every
    receiver).
    """
    # Every patched run (Run 1 and Run 2) executes on corrupt_tokens, so positions
    # must index the CORRUPT sequence. Clean and corrupt can differ in length
    # (e.g. 5-shot clean vs zero-shot corrupt), "all_pos" is only valid 
    # when the lengths match.
    all_pos = list(range(corrupt_tokens.shape[1]))
    uses_all_pos = "all_pos" in (sender_pos, receiver_pos, freeze_pos)
    if uses_all_pos and clean_tokens.shape[1] != corrupt_tokens.shape[1]:
        raise ValueError(
            f"'all_pos' needs clean and corrupt prompts of equal length, got "
            f"{clean_tokens.shape[1]} vs {corrupt_tokens.shape[1]} tokens. "
            "Use explicit negative positions like [-1] instead."
        )
    if sender_pos == "all_pos":
        sender_pos = all_pos
    if receiver_pos == "all_pos":
        receiver_pos = all_pos
    if freeze_pos == "all_pos":
        freeze_pos = all_pos

    spec = get_model_specs(model)
    n_layers, n_heads, d_model, d_head = spec["n_layers"], spec["n_heads"], spec["d_model"], spec["d_head"]
    accessor_config = get_accessor_config(model)
    accessor = ModelAccessor(model, accessor_config)
    batch_size = len(clean_tokens)

    # 1. Baseline logits & diff
    with torch.no_grad():
        clean_logits_trace = model.trace(clean_tokens, trace=False, remote=remote)
        corrupt_logits_trace = model.trace(corrupt_tokens, trace=False, remote=remote)
        clean_answer_logits = clean_logits_trace["logits"][torch.arange(batch_size), -1, answer_tokens]
        corrupt_answer_logits = corrupt_logits_trace["logits"][torch.arange(batch_size), -1, answer_tokens]

    baseline_diff = (clean_answer_logits - corrupt_answer_logits).mean()
    epsilon = 1e-8
    if torch.abs(baseline_diff) < epsilon:
        print(f"Warning: Baseline difference {baseline_diff.item():.4f} is close to zero.")
        baseline_diff = torch.sign(baseline_diff) * max(
            torch.abs(baseline_diff), torch.tensor(epsilon, device=baseline_diff.device)
        )

    # 2. Cache clean sender & corrupt activations
    clean_attn_proxies, clean_mlp_proxies = {}, {}
    corrupt_attn_proxies, corrupt_mlp_proxies = {}, {}

    sender_layers_to_cache = set(range(max_sender_layer + 1))
    valid_sender_components = defaultdict(set)
    for layer in sender_layers_to_cache:
        is_max_sender_layer_mlp_receiver = layer == max_sender_layer and min_head_or_mlp == -1
        valid_sender_components[layer].update(range(n_heads))
        if not is_max_sender_layer_mlp_receiver:
            valid_sender_components[layer].add(n_heads)

    with accessor.trace(remote=remote) as tracer_clean_cache:
        with tracer_clean_cache.invoke(clean_tokens):
            for layer in sender_layers_to_cache:
                is_sender_layer_attn = any(idx < n_heads for idx in valid_sender_components.get(layer, set()))
                is_sender_layer_mlp = n_heads in valid_sender_components.get(layer, set())
                if is_sender_layer_attn:
                    try:
                        clean_attn_proxies[layer] = accessor.layers[layer].attention.output.unwrap().input[:, sender_pos].save()
                    except Exception as e:
                        print(f"Clean Cache Proxy Save Warning: Attn layer {layer}. {e}")
                if is_sender_layer_mlp:
                    try:
                        clean_mlp_proxies[layer] = accessor.layers[layer].mlp.unwrap().output[:, sender_pos].save()
                    except Exception as e:
                        print(f"Clean Cache Proxy Save Warning: MLP layer {layer}. {e}")

    with accessor.trace(remote=remote) as tracer_corrupt_cache:
        with tracer_corrupt_cache.invoke(corrupt_tokens):
            for layer in range(n_layers):
                try:
                    corrupt_attn_proxies[layer] = accessor.layers[layer].attention.output.unwrap().input[:, :].save()
                except Exception as e:
                    print(f"Corrupt Cache Proxy Save Warning: Attn layer {layer}. {e}")
                try:
                    corrupt_mlp_proxies[layer] = accessor.layers[layer].mlp.unwrap().output[:, :].save()
                except Exception as e:
                    print(f"Corrupt Cache Proxy Save Warning: MLP layer {layer}. {e}")

    # 3. Path patching loop -- now targets ONE receiver
    results_tensor = torch.zeros(result_shape, device="cpu")
    rec_layer, rec_head_or_mlp = receiver
    rec_comp_idx = n_heads if rec_head_or_mlp == -1 else rec_head_or_mlp

    for sender_layer in tqdm(range(max_sender_layer + 1), desc=f"Senders -> L{rec_layer}C{rec_comp_idx}", leave=False):
        for sender_comp_idx in range(n_heads + 1):
            if sender_comp_idx not in valid_sender_components.get(sender_layer, set()):
                continue
            is_sender_attn = sender_comp_idx < n_heads
            if is_sender_attn and sender_layer not in clean_attn_proxies:
                continue
            if not is_sender_attn and sender_layer not in clean_mlp_proxies:
                continue

            # Run 1: freeze pre-sender & intermediate layers, patch sender -> save the ONE receiver's activation
            patched_receiver_activation_proxy = None
            with accessor.trace(remote=remote) as tracer_run1:
                with tracer_run1.invoke(corrupt_tokens):
                    for current_layer in range(sender_layer):
                        if current_layer in corrupt_attn_proxies:
                            accessor.layers[current_layer].attention.output.unwrap().input[:, freeze_pos] = (
                                _unwrap(corrupt_attn_proxies[current_layer])[:, freeze_pos].clone()
                            )
                        if current_layer in corrupt_mlp_proxies:
                            accessor.layers[current_layer].mlp.unwrap().output[:, freeze_pos] = _unwrap(corrupt_mlp_proxies[current_layer])[:, freeze_pos].clone()

                    if is_sender_attn:
                        if sender_layer in corrupt_attn_proxies and sender_layer in clean_attn_proxies:
                            current_attn_input = accessor.layers[sender_layer].attention.output.unwrap().input[...]
                            current_attn_input[:, freeze_pos] = _unwrap(corrupt_attn_proxies[sender_layer])[:, freeze_pos].clone()
                            current_attn_reshaped = current_attn_input.reshape(batch_size, len(all_pos), n_heads, d_head)
                            clean_attn_tensor = _unwrap(clean_attn_proxies[sender_layer]).reshape(
                                batch_size, len(sender_pos), n_heads, d_head
                            )
                            current_attn_reshaped[:, sender_pos, sender_comp_idx, :] = clean_attn_tensor[
                                :, :, sender_comp_idx, :
                            ].clone()
                    else:
                        if sender_layer in corrupt_attn_proxies:
                            accessor.layers[sender_layer].attention.output.unwrap().input[:, freeze_pos] = (
                                _unwrap(corrupt_attn_proxies[sender_layer])[:, freeze_pos].clone()
                            )
                        if sender_layer in corrupt_mlp_proxies and sender_layer in clean_mlp_proxies:
                            current_mlp_output = accessor.layers[sender_layer].mlp.unwrap().output[:, :]
                            current_mlp_output[:, freeze_pos] = _unwrap(corrupt_mlp_proxies[sender_layer])[:, freeze_pos].clone()
                            current_mlp_output[:, sender_pos] = _unwrap(clean_mlp_proxies[sender_layer]).clone()

                    for current_layer in range(sender_layer + 1, min_layer):
                        if current_layer in corrupt_attn_proxies:
                            accessor.layers[current_layer].attention.output.unwrap().input[:, freeze_pos] = (
                                _unwrap(corrupt_attn_proxies[current_layer])[:, freeze_pos].clone()
                            )
                        if current_layer in corrupt_mlp_proxies:
                            accessor.layers[current_layer].mlp.unwrap().output[:, freeze_pos] = _unwrap(corrupt_mlp_proxies[current_layer])[:, freeze_pos].clone()

                    if min_head_or_mlp == -1 and min_layer != sender_layer and min_layer in corrupt_attn_proxies:
                        accessor.layers[min_layer].attention.output.unwrap().input[:, freeze_pos] = (
                            _unwrap(corrupt_attn_proxies[min_layer])[:, freeze_pos].clone()
                        )

                    # Save just this one receiver's resulting activation
                    if rec_comp_idx < n_heads:
                        rec_attn_out_input = accessor.layers[rec_layer].attention.output.unwrap().input[:, receiver_pos]
                        rec_attn_out_reshaped = rec_attn_out_input.reshape(batch_size, len(receiver_pos), n_heads, d_head)
                        patched_receiver_activation_proxy = rec_attn_out_reshaped[:, :, rec_comp_idx, :].save()
                    else:
                        patched_receiver_activation_proxy = accessor.layers[rec_layer].mlp.unwrap().output[:, receiver_pos].save()

            if patched_receiver_activation_proxy is None:
                print(f"Warning: Run 1 incomplete for sender ({sender_layer},{sender_comp_idx}). Skipping Run 2.")
                continue

            # Run 2: patch just this one receiver, get final logits
            intervened_logits_obj = None
            with accessor.trace(remote=remote) as tracer_run2:
                with tracer_run2.invoke(corrupt_tokens):
                    if rec_comp_idx < n_heads:
                        current_attn_input = accessor.layers[rec_layer].attention.output.unwrap().input[...]
                        current_attn_reshaped = current_attn_input.reshape(batch_size, len(all_pos), n_heads, d_head)
                        if patched_receiver_activation_proxy is not None:
                            current_attn_reshaped[:, receiver_pos, rec_comp_idx, :] = _unwrap(patched_receiver_activation_proxy).clone()
                    else:
                        current_mlp_output = accessor.layers[rec_layer].mlp.unwrap().output[...]
                        if patched_receiver_activation_proxy is not None:
                            current_mlp_output[:, receiver_pos] = _unwrap(patched_receiver_activation_proxy).clone()

                    intervened_logits_obj = (
                        accessor.lm_head.unwrap().output[torch.arange(batch_size), -1, answer_tokens].save()
                    )

            if intervened_logits_obj is None:
                print(f"Skipping calculation for sender ({sender_layer},{sender_comp_idx}) due to Run 2 failure.")
                continue

            if hasattr(intervened_logits_obj, "value"):
                intervened_logits_val = intervened_logits_obj.value
            elif isinstance(intervened_logits_obj, torch.Tensor):
                intervened_logits_val = intervened_logits_obj
            else:
                print(f"Unexpected type for intervened_logits_obj: {type(intervened_logits_obj)}. Skipping.")
                continue

            corrupt_answer_logits_dev = corrupt_answer_logits.to(intervened_logits_val.device)
            baseline_diff_dev = baseline_diff.to(intervened_logits_val.device)
            intervention_diff = intervened_logits_val.mean() - corrupt_answer_logits_dev.mean()
            normalized_effect = (intervention_diff / (baseline_diff_dev + 1e-12)).item()
            results_tensor[sender_layer, sender_comp_idx] = normalized_effect

    return results_tensor

def path_patch_sender_to_receiver_batch(
    model: LanguageModel,
    clean_prompts: list[str],
    corrupt_prompts: list[str],
    answers: list[str],
    receiver: tuple[int, int],
    min_layer: int,
    min_head_or_mlp: int,
    max_sender_layer: int,
    result_shape: tuple[int, int],
    batch_size: int = 8,
    remote: bool = True,
    sender_pos: list[int] = [-1],
    receiver_pos: list[int] = [-1],
    freeze_pos: list[int] = [-1],
) -> torch.Tensor:
    n_samples = len(clean_prompts)
    if not (n_samples == len(corrupt_prompts) == len(answers)):
        raise ValueError("Input lists must have the same length.")

    tokenizer_kwargs = {"padding": True, "return_tensors": "pt"}
    clean_tokens = model.tokenizer(clean_prompts, padding_side="left", **tokenizer_kwargs)["input_ids"]
    corrupt_tokens = model.tokenizer(corrupt_prompts, padding_side="left", **tokenizer_kwargs)["input_ids"]
    answer_tokens = model.tokenizer(answers, padding_side="right", add_special_tokens=False, **tokenizer_kwargs)["input_ids"][:, 0]

    device = _model_device(model)
    clean_tokens = clean_tokens.to(device)
    corrupt_tokens = corrupt_tokens.to(device)
    answer_tokens = answer_tokens.to(device)

    all_results = torch.zeros(result_shape, device="cpu")
    for i in tqdm(range(0, n_samples, batch_size), desc=f"Batches -> receiver {receiver}"):
        batch_end = min(i + batch_size, n_samples)
        current_batch_size = batch_end - i
        batch_results_tensor = path_patch_sender_to_receiver(
            model=model, clean_tokens=clean_tokens[i:batch_end], corrupt_tokens=corrupt_tokens[i:batch_end],
            answer_tokens=answer_tokens[i:batch_end], receiver=receiver, min_layer=min_layer,
            min_head_or_mlp=min_head_or_mlp, max_sender_layer=max_sender_layer, result_shape=result_shape,
            remote=remote, sender_pos=sender_pos, receiver_pos=receiver_pos, freeze_pos=freeze_pos,
        )
        all_results += batch_results_tensor.cpu() * (current_batch_size / n_samples)

    return all_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--positions", type=str, default="last", choices=["last", "all"],
        help="last: patch sender/receiver/freeze at the final token only. "
             "all: every token position; needs clean and corrupt prompts aligned token-by-token, "
             "so it keeps only inputs/outputs of a fixed token length (--input_length/--output_length) "
             "and cannot be used with --corruption_type zs")
    parser.add_argument("--input_length", type=int, default=1,
        help="--positions all only: keep dataset items whose input is exactly this many tokens "
             "(e.g. 5 for park-country, whose park names are never a single token)")
    parser.add_argument("--output_length", type=int, default=1,
        help="--positions all only: keep dataset items whose output is exactly this many tokens")
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
    parser.add_argument("--batch_size", type=int, default=8,
        help="prompts per forward pass during path patching; raise it if GPU memory allows")
    parser.add_argument("--k", type=int, default=10, help="top k decoded tokens to look for match")
    parser.add_argument("--corruption_type", type=str, default="zs",
                        choices=["zs", "shuffle_input", "shuffle_output"],
                        help="create corrupted_prompts based on corruption_type")
    parser.add_argument("--exp_size", type=int, default=300,
        help="max prompts to path patch; tasks with more are randomly subsampled (seeded by --seed)")
    # parser.add_argument("--dataset_folder", type=str, default="../datasets/abstractive", help="folder of the dataset")
    parser.add_argument("--dataset_folder", type=str, default="datasets/abstractive", help="folder of the dataset")
    parser.add_argument("--remote", type=bool, default=False, help="whether to use NDIF to run model remotely")
    parser.add_argument("--task_relation_dict_type", type=str, default="human",
        help="type of the task relation dict: human, llms, combined")
    parser.add_argument("--threshold", "-p", type=float, default=0.1,
    help="p: minimum fraction of prompts a head must match to count as a lexical task head")
    parser.add_argument("--receivers", type=str, nargs="+", default=None, metavar="LAYER,HEAD",
        help="one-off receivers, e.g. --receivers 7,5 9,4; replaces the shared-LTH JSON list")

    args = parser.parse_args()
    if args.positions == "all" and args.corruption_type == "zs":
        parser.error("--positions all needs a corrupt prompt with the same token layout as the clean one; "
                     "a zero-shot prompt has none. Use a shuffle corruption.")
    # Random seed for shuffle input/output
    random.seed(args.seed)

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

    # Load model 
    print("To load model")
    model = LanguageModel(
        args.model_name,
        device_map="auto",
        dispatch=True if not args.remote else False
    )
    model_name = args.model_name.split("/")[-1]
    print("Model loaded")

    # Load receiver list
    save_path = os.path.join(
        args.save_root, model_name, args.d_name, "Heads", "MAPS_nnsight",
        f"shared_{args.d_name}_heads_p{args.threshold}_k{args.k}.json",
    )
    if args.receivers is not None:
        receiver_list = [[int(x) for x in r.split(",")] for r in args.receivers]
    else:
        with open(save_path, "r") as f:
            receiver_list = json.load(f)
    print("receivers:", receiver_list)

    # Load prompts
    if args.prompt_type == "EP":
        file_name = "EP_vary_n_shot_behavior.json"
        file_path = os.path.join(
            args.save_root, model_name, "across_tasks", "Behavior",
            file_name,
        )
        with open(file_path, "r") as f:
            correct_dataset = json.load(f)
        # prompt_temp_idx_list = [1,2,3,4,5,10,20,30]
        prompt_temp_idx_list = [5]
    elif args.prompt_type == "IP":
        file_name = "IP_vary_n_inst_behavior.json"
        prompt_temp_idx_list = [0,1,2,3,4]
    else:
        raise ValueError(f"prompt_type {args.prompt_type} not supported")
    
    for prompt_temp_idx in prompt_temp_idx_list:
        # generate_few_shot_prompts loads the dataset itself
        clean_prompts, corrupt_prompts, clean_answers = generate_few_shot_prompts(
            d_name=args.d_name,
            model=model,
            corruption_type=args.corruption_type,
            filter_correct=False,
            n_shot=prompt_temp_idx,
            batch_size=args.batch_size,
            dataset_folder=args.dataset_folder,
            # all positions: fixed-length inputs/outputs so shuffling never changes a prompt's length
            INPUT_LENGTH=args.input_length if args.positions == "all" else None,
            OUTPUT_LENGTH=args.output_length if args.positions == "all" else None,
        )
        if not clean_prompts:
            raise ValueError(
                f"no prompts left for {args.d_name}"
                + (f": no dataset item has a {args.input_length}-token input and a {args.output_length}-token "
                   f"output. Pick lengths that exist with --input_length/--output_length."
                   if args.positions == "all" else "")
            )
        # cap the prompt count; datasets are sorted (e.g. by country), so sample instead of taking the first N
        if len(clean_prompts) > args.exp_size:
            keep = sorted(random.Random(args.seed).sample(range(len(clean_prompts)), args.exp_size))
            clean_prompts = [clean_prompts[i] for i in keep]
            corrupt_prompts = [corrupt_prompts[i] for i in keep]
            clean_answers = [clean_answers[i] for i in keep]

    # Zero-shot path patching only
    # corrupt_prompts = create_zs_prompts(dataset)

    print(len(corrupt_prompts), len(clean_prompts), len(clean_answers))

    if args.positions == "all":
        # position p must mean the same thing in both prompts of every pair
        clean_lens = [len(model.tokenizer(p)["input_ids"]) for p in clean_prompts]
        corrupt_lens = [len(model.tokenizer(p)["input_ids"]) for p in corrupt_prompts]
        misaligned = [i for i, (a, b) in enumerate(zip(clean_lens, corrupt_lens)) if a != b]
        if misaligned:
            i = misaligned[0]
            raise ValueError(
                f"{len(misaligned)}/{len(clean_prompts)} clean/corrupt pairs differ in token length, "
                f"e.g. #{i}: {clean_lens[i]} vs {corrupt_lens[i]} tokens\n"
                f"  clean  : {clean_prompts[i]!r}\n  corrupt: {corrupt_prompts[i]!r}"
            )
        print(f"all-position patching: {len(clean_prompts)} single-token prompt pairs, all aligned")
    pos = "all_pos" if args.positions == "all" else [-1]

    # List of senders
    spec = get_model_specs(model)
    n_heads = spec["n_heads"]
    # Path patching sender to one LTH at a time
    for receiver in receiver_list:
        # Senders that are all heads above the target receivers
        min_layer, min_head_or_mlp = find_earliest_receiver([receiver])
        if min_head_or_mlp >= 0:
            max_sender_layer = min_layer - 1
            result_shape = (min_layer, n_heads + 1)
        else:
            max_sender_layer = min_layer
            result_shape = (min_layer + 1, n_heads + 1)
        
        results = path_patch_sender_to_receiver_batch(
            model=model, clean_prompts=clean_prompts, corrupt_prompts=corrupt_prompts,
            answers=clean_answers, receiver=receiver,
            min_layer=min_layer, min_head_or_mlp=min_head_or_mlp,
            max_sender_layer=max_sender_layer, result_shape=result_shape,
            batch_size=args.batch_size, remote=args.remote, sender_pos=pos, receiver_pos=pos, freeze_pos=pos)
        
        # save_dir = os.path.join(args.save_root, model_name, args.d_name, "path_patching")
        # shuffle corruptions are random: keep the seed in the folder so different draws never mix
        corruption_dir = args.corruption_type if args.corruption_type == "zs" else f"{args.corruption_type}_seed{args.seed}"
        if args.positions == "all":
            corruption_dir += "_allpos"
        elif args.positions == "last":
                    corruption_dir += "_lastpos"
        save_dir = os.path.join(args.save_root, model_name, args.d_name, "path_patching_per_receiver", corruption_dir)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{args.prompt_type}_L{receiver[0]}H{receiver[1]}_batch_results_tensor.pt")
        torch.save(results, save_path) 
        print(f"Saved logit diff for {receiver} to {save_path}")