"""
Representation similarity (RSA) of attention heads over few-shot prompt positions.

For each head, its output at every token position (z_h @ W_O[:, h]) is compared across positions
with cosine similarity, giving a [seq, seq] matrix per prompt. Block structure aligned with the
prompt's chunks (input / separator / output / query) suggests the head encodes chunk roles.

Heads: --heads, or by default the task's top --top_n senders (heads only) from
    <save_root>/<model>/across_tasks/Behavior/<prompt_type>_top<top_k>_sender_frequency.csv
Outputs: <save_root>/<model>/<d_name>/rsa/<n_shot>shot_<n_prompts>prompts_seed<seed>/
    L{l}H{h}_rsa_mean.html, L{l}H{h}_rsa_prompts.html, L{l}H{h}_rsa.pt

Usage:
    python src/rsa.py --d_name country-capital
    python src/rsa.py --d_name country-capital --heads 6,17 7,28 --n_prompts 12
"""
import argparse
import json
import os
import random

import pandas as pd
import torch
import torch.nn.functional as F
from nnsight import LanguageModel

from shared_utils.plot_utils import plot_rsa_grid, plot_rsa_matrix
from shared_utils.prompt_utils import build_chunk_labeled_prompts, filter_token_pairs_by_length
from shared_utils.wrapper import ModelAccessor, get_accessor_config, get_model_specs

torch.set_grad_enabled(False)

Head = tuple[int, int]


def _unwrap(saved):
    """nnsight < 0.4 returns a proxy with .value; newer versions return the tensor."""
    return saved.value if hasattr(saved, "value") else saved


def candidate_heads(freq_csv: str, d_name: str, top_n: int) -> list[Head]:
    """Top top_n heads (MLPs skipped) for d_name, in the CSV's count order."""
    df = pd.read_csv(freq_csv)
    heads = df[(df["task"] == d_name) & df["sender"].str.match(r"^L\d+H\d+$")].head(top_n)
    if heads.empty:
        raise ValueError(f"no heads for {d_name} in {freq_csv}")
    return [tuple(int(x) for x in s[1:].split("H")) for s in heads["sender"]]


def head_outputs(model: LanguageModel, input_ids: torch.Tensor, heads: list[Head]) -> dict[Head, torch.Tensor]:
    """{(layer, head): [n_prompts, seq, d_model]} -- each head's output at every position."""
    spec = get_model_specs(model)
    n_heads, d_head = spec["n_heads"], spec["d_head"]
    accessor = ModelAccessor(model, get_accessor_config(model))
    layers = sorted({layer for layer, _ in heads})

    saved = {}
    with model.trace(input_ids):
        for layer in layers:
            # input to o_proj = concatenated per-head z: [n, seq, n_heads * d_head]
            saved[layer] = accessor.layers[layer].attention.output.unwrap().input.save()

    outputs = {}
    for layer, head in heads:
        z = _unwrap(saved[layer]).float()
        w_o = model.state_dict()[f"model.layers.{layer}.self_attn.o_proj.weight"].float().to(z.device)
        cols = slice(head * d_head, (head + 1) * d_head)
        outputs[(layer, head)] = (z[..., cols] @ w_o[:, cols].T).cpu()
    assert n_heads * d_head == z.shape[-1]
    return outputs


def cosine_similarity_matrices(x: torch.Tensor) -> torch.Tensor:
    """[n, seq, d] -> [n, seq, seq] cosine similarity between positions."""
    normed = F.normalize(x, dim=-1)
    return normed @ normed.transpose(-1, -2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--d_name", type=str, required=True, help="task, e.g. country-capital")
    parser.add_argument("--n_shot", type=int, default=5)
    parser.add_argument("--n_prompts", type=int, default=12, help="prompts per head (rows of the grid plot)")
    parser.add_argument("--seed", type=int, default=42, help="seed for sampling prompts")
    parser.add_argument("--heads", type=str, nargs="+", default=None, metavar="LAYER,HEAD",
        help="heads to analyze, e.g. --heads 6,17 7,28; default: top candidates from the frequency CSV")
    parser.add_argument("--top_k", type=int, default=10, help="which frequency CSV (top-k edges) to read candidates from")
    parser.add_argument("--top_n", type=int, default=5, help="number of candidate heads to take from that CSV")
    parser.add_argument("--input_length", type=int, default=1,
        help="keep items whose input is this many tokens, so all prompts share one layout and can be averaged")
    parser.add_argument("--output_length", type=int, default=1)
    parser.add_argument("--prompt_type", type=str, default="EP", choices=["EP"])
    parser.add_argument("--save_root", type=str, default="output")
    parser.add_argument("--dataset_folder", type=str, default="datasets/abstractive")
    args = parser.parse_args()

    model_dir = os.path.join(args.save_root, args.model_name.split("/")[-1])
    if args.heads is not None:
        heads = [tuple(int(x) for x in h.split(",")) for h in args.heads]
    else:
        freq_csv = os.path.join(model_dir, "across_tasks", "Behavior",
                                f"{args.prompt_type}_top{args.top_k}_sender_frequency.csv")
        heads = candidate_heads(freq_csv, args.d_name, args.top_n)
    print("heads:", [f"L{l}H{h}" for l, h in heads])

    model = LanguageModel(args.model_name, device_map="auto" if torch.cuda.is_available() else "cpu",
                          dispatch=True, dtype=torch.float32)

    # prompts: fixed-length items so every prompt has the same chunk layout
    with open(os.path.join(args.dataset_folder, f"{args.d_name}.json"), encoding="utf-8") as f:
        dataset = filter_token_pairs_by_length(json.load(f), model,
                                               input_len=args.input_length, output_len=args.output_length)
    if len(dataset) < args.n_prompts:
        raise ValueError(f"only {len(dataset)} items with {args.input_length}/{args.output_length}-token "
                         f"input/output, need {args.n_prompts}")
    query_indices = sorted(random.Random(args.seed).sample(range(len(dataset)), args.n_prompts))
    prompts = build_chunk_labeled_prompts(dataset, model.tokenizer, args.n_shot, query_indices)
    lengths = {len(p["input_ids"]) for p in prompts}
    if len(lengths) != 1:
        raise ValueError(f"prompts differ in token length {sorted(lengths)}; matrices cannot be averaged")
    input_ids = torch.tensor([p["input_ids"] for p in prompts])
    labels = [p["labels"] for p in prompts]
    # mean plot: axis labels show the first prompt's tokens; hover lists every prompt's token at that position
    mean_labels = prompts[0]["labels"]
    roles = prompts[0]["roles"]
    position_hover = [" | ".join(sorted({lab[i][len(f"{i}_{roles[i]}_"):] for lab in labels}))
                      for i in range(len(mean_labels))]

    save_dir = os.path.join(model_dir, args.d_name, "rsa",
                            f"{args.n_shot}shot_{args.n_prompts}prompts_seed{args.seed}")
    os.makedirs(save_dir, exist_ok=True)

    for (layer, head), out in head_outputs(model, input_ids, heads).items():
        name = f"L{layer}H{head}"
        sims = cosine_similarity_matrices(out)          # [n_prompts, seq, seq]
        mean = sims.mean(dim=0)
        plot_rsa_matrix(mean, mean_labels,
            f"{name} representation similarity, mean over {args.n_prompts} prompts ({args.d_name}, {args.n_shot}-shot; "
            f"axis tokens from prompt 1, hover for all)",
            position_hover=position_hover,
        ).write_html(os.path.join(save_dir, f"{name}_rsa_mean.html"))
        plot_rsa_grid(sims, labels, [f"query: {p['prompt'].split(';')[-1].strip()}" for p in prompts],
            title=f"{name} representation similarity per prompt ({args.d_name}, {args.n_shot}-shot)",
        ).write_html(os.path.join(save_dir, f"{name}_rsa_prompts.html"))
        torch.save({"sims": sims, "mean": mean, "labels": labels, "roles": prompts[0]["roles"],
                    "prompts": [p["prompt"] for p in prompts], "query_indices": query_indices},
                   os.path.join(save_dir, f"{name}_rsa.pt"))
        print(f"{name}: saved to {save_dir}")


if __name__ == "__main__":
    main()
