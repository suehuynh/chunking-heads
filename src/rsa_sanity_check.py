"""
Gut check: can a perfectly binary (+1/-1) RSA matrix come from a real head output?

For one prompt and one head, compute the position x position cosine-similarity matrix from:
    head_output   z_h @ W_O[:, h]          [seq, d_model]  (what rsa.py uses)
    head_z        z_h                      [seq, d_head]
    attn_out[h]   attention output, coordinate h   [seq]  (indexing a head out of a d_model tensor)
    head_out[0]   head output, coordinate 0        [seq]
and report each matrix's rank and how many cells are ~ +/-1. A binary checkerboard is a
rank-1 sign matrix s_i * s_j, which only happens when each position is one number.

Usage:
    python src/rsa_sanity_check.py --d_name country-capital --layer 5 --head 6
    python src/rsa_sanity_check.py --model_name meta-llama/Llama-3.2-1B --d_name country-capital --layer 5 --head 6
"""
import argparse
import json
import os
import random

import torch
from nnsight import LanguageModel

from rsa import _unwrap, cosine_similarity_matrices
from shared_utils.plot_utils import plot_rsa_grid
from shared_utils.prompt_utils import build_chunk_labeled_prompts, filter_token_pairs_by_length
from shared_utils.wrapper import ModelAccessor, get_accessor_config, get_model_specs

torch.set_grad_enabled(False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--d_name", type=str, default="country-capital")
    parser.add_argument("--layer", type=int, default=5)
    parser.add_argument("--head", type=int, default=6)
    parser.add_argument("--n_shot", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42, help="picks the prompt")
    parser.add_argument("--save_root", type=str, default="output")
    parser.add_argument("--dataset_folder", type=str, default="datasets/abstractive")
    args = parser.parse_args()
    layer, head = args.layer, args.head

    model = LanguageModel(args.model_name, device_map="auto" if torch.cuda.is_available() else "cpu",
                          dispatch=True, dtype=torch.float32)
    d_head = get_model_specs(model)["d_head"]

    with open(os.path.join(args.dataset_folder, f"{args.d_name}.json"), encoding="utf-8") as f:
        dataset = filter_token_pairs_by_length(json.load(f), model, input_len=1, output_len=1)
    query_index = random.Random(args.seed).randrange(len(dataset))
    prompt = build_chunk_labeled_prompts(dataset, model.tokenizer, args.n_shot, [query_index])[0]
    print("prompt:", repr(prompt["prompt"]))

    accessor = ModelAccessor(model, get_accessor_config(model))
    with model.trace(torch.tensor([prompt["input_ids"]])):
        z_all = accessor.layers[layer].attention.output.unwrap().input.save()   # [1, seq, n_heads*d_head]
    z_all = _unwrap(z_all)[0].float()
    w_o = model.state_dict()[f"model.layers.{layer}.self_attn.o_proj.weight"].float().to(z_all.device)
    cols = slice(head * d_head, (head + 1) * d_head)

    head_z = z_all[:, cols]                    # [seq, d_head]
    head_out = head_z @ w_o[:, cols].T         # [seq, d_model]
    attn_out = z_all @ w_o.T                   # [seq, d_model], all heads summed (= o_proj output)
    variants = {
        f"head_output L{layer}H{head} [seq, {head_out.shape[-1]}] (rsa.py)": head_out,
        f"head_z L{layer}H{head} [seq, {d_head}]": head_z,
        f"attn_out L{layer}[:, {head}] [seq, 1] (head index on d_model)": attn_out[:, head:head + 1],
        f"head_output L{layer}H{head}[:, 0] [seq, 1]": head_out[:, :1],
    }

    sims = []
    print(f"\n{'variant':60s} {'rank':>5s} {'cells |sim|>0.99':>17s}")
    for name, x in variants.items():
        s = cosine_similarity_matrices(x.unsqueeze(0).cpu())[0]
        rank = torch.linalg.matrix_rank(s, atol=1e-4).item()
        near_pm1 = (s.abs() > 0.99).float().mean().item()
        print(f"{name:60s} {rank:5d} {near_pm1:16.0%}")
        sims.append(s)

    save_dir = os.path.join(args.save_root, args.model_name.split("/")[-1], args.d_name, "rsa", "sanity_check")
    os.makedirs(save_dir, exist_ok=True)
    html_path = os.path.join(save_dir, f"L{layer}H{head}_rsa_variants.html")
    plot_rsa_grid(torch.stack(sims), [prompt["labels"]] * len(sims), list(variants), n_cols=2,
                  title=f"L{layer}H{head} RSA computed four ways ({args.d_name}, {args.n_shot}-shot, one prompt)",
                  ).write_html(html_path)
    print(f"\nsaved {html_path}")
    print("binary checkerboard <=> rank 1 with 100% of cells at +/-1 <=> one number per position")


if __name__ == "__main__":
    main()
