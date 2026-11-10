"""
Screen the most important senders to lexical task heads (LTH) for symbolic-head signatures.

Candidates: the task's top --top_n senders (attention heads only) in
    <save_root>/<model>/across_tasks/Behavior/EP_top<top_k>_sender_frequency.csv
or --heads. Every head in the model is scored on the same prompts, so a candidate's score can be
read as a percentile against all heads.

Scores (attention, mean over prompts; 5-shot single-token prompts with chunk roles):
    induction   attention from the last token to demo output positions
    binding     from each demo-ending ";", attention to the same demo's input and output
    prev_token  attention from each position to the token right before it
    pair_binding          from each demo's output to that demo's own input
    pair_binding_control  the same distance back, from each later demo's input (offset heads score here too)
    *_shuffled_ratio  the score on shuffled pairs (real words, wrong partners) / clean: about 1 means
                the head responds to the category relation, not the specific fact
    *_ratio     the score on role-only prompts (every content word replaced by filler) divided by the
                clean score: about 1 means the head follows roles, not words (symbolic)
Labels (a score counts as high at or above the --percentile of all heads):
    symbolic_induction  high induction, induction_ratio >= --ratio
    content_induction   high induction, induction_ratio <  --ratio
    abstraction_like    high binding,   binding_ratio   >= --ratio
    previous_token      high prev_token
    symbolic_pair_binding / content_pair_binding  high pair_binding, at least 2x its control,
                        pair_binding_ratio >= / < --ratio
This is a screen: in the "A: B;" format, binding cannot be told apart from attending a fixed
distance back; causal ablation is the test of whether a head matters.

Usage:
    python src/classify_senders.py --d_name country-capital
    python src/classify_senders.py --d_name antonym --heads 7,28 6,14 5,6
"""
import argparse
import json
import os
import random

import numpy as np
import pandas as pd
import torch
from nnsight import LanguageModel

from lth_corruption import FILLER_WORDS, _unwrap
from route_utils import PromptMaker
from shared_utils.prompt_utils import build_chunk_labeled_prompts
from shared_utils.plot_utils import plot_rsa_matrix, plot_score_table
from shared_utils.wrapper import ModelAccessor, get_accessor_config, get_model_specs

torch.set_grad_enabled(False)


SCORES = ["induction", "binding", "pair_binding", "pair_binding_control", "prev_token"]


def demo_spans(roles: list[str]) -> list[dict]:
    """Positions of each demo: {"input": [...], "output": [...], "end": index of its closing ';'}."""
    demos, current, stage = [], {"input": [], "output": []}, None
    for i, role in enumerate(roles):
        if role == "input":
            if stage == "separator_after_output":
                current = {"input": [], "output": []}
            stage = "input"
            current["input"].append(i)
        elif role == "output":
            stage = "output"
            current["output"].append(i)
        elif role == "separator" and stage == "output":
            current["end"] = i
            demos.append(current)
            stage = "separator_after_output"
    return demos


def attention_all_layers(model, input_ids: torch.Tensor, batch_size: int) -> torch.Tensor:
    """[n_prompts, n_layers, n_heads, seq, seq] attention weights (eager attention)."""
    n_layers = get_model_specs(model)["n_layers"]
    accessor = ModelAccessor(model, get_accessor_config(model))
    out = []
    for start in range(0, len(input_ids), batch_size):
        saved = {}
        with model.trace(input_ids[start:start + batch_size]):
            for layer in range(n_layers):
                saved[layer] = accessor.layers[layer].attention.unwrap().output[1].save()
        out.append(torch.stack([_unwrap(saved[l]).float().cpu() for l in range(n_layers)], dim=1))
    return torch.cat(out)


def head_scores(attn: torch.Tensor, roles: list[str]) -> dict[str, torch.Tensor]:
    """Per head [n_layers, n_heads], mean over prompts. attn: [n, L, H, seq, seq]."""
    demos = demo_spans(roles)
    last = len(roles) - 1
    outputs = [i for d in demos for i in d["output"]]
    induction = attn[..., last, outputs].sum(-1)
    binding = torch.stack([attn[..., d["end"], d["input"] + d["output"]].sum(-1) for d in demos]).mean(0)
    # from each demo's (first) output token to that demo's own input
    pair_binding = torch.stack([attn[..., d["output"][0], d["input"]].sum(-1) for d in demos]).mean(0)
    # control at the same distance back, from each later demo's (first) input: an offset head scores here too
    pair_binding_control = torch.stack([
        attn[..., d["input"][0], d["input"][0] - (d["output"][0] - d["input"][0])] for d in demos[1:]]).mean(0)
    idx = torch.arange(1, len(roles))
    prev_token = attn[..., idx, idx - 1].mean(-1)
    return {"induction": induction.mean(0), "binding": binding.mean(0), "pair_binding": pair_binding.mean(0),
            "pair_binding_control": pair_binding_control.mean(0), "prev_token": prev_token.mean(0)}


def candidate_heads(freq_csv: str, d_name: str, top_n: int) -> list[tuple[int, int]]:
    """The task's top_n senders by count, keeping attention heads (MLPs cannot be symbolic heads)."""
    df = pd.read_csv(freq_csv)
    top = df[df["task"] == d_name].head(top_n)
    heads = top[top["sender"].str.match(r"^L\d+H\d+$")]["sender"]
    return [tuple(int(x) for x in s[1:].split("H")) for s in heads]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--d_name", type=str, required=True)
    parser.add_argument("--heads", type=str, nargs="+", default=None, metavar="LAYER,HEAD",
        help="heads to classify; default: the task's top --top_n senders from the frequency CSV")
    parser.add_argument("--top_n", type=int, default=15, help="senders to take from the frequency CSV (MLPs dropped)")
    parser.add_argument("--top_k", type=int, default=15, help="which frequency CSV (top-k edges)")
    parser.add_argument("--n_shot", type=int, default=5)
    parser.add_argument("--n_prompts", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--percentile", type=float, default=95, help="a score is high at or above this percentile of all heads")
    parser.add_argument("--ratio", type=float, default=0.75, help="role-only / clean score at or above which a head follows roles")
    parser.add_argument("--threshold", type=float, default=0.1, help="which LTH list to flag")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--save_root", type=str, default="output")
    parser.add_argument("--input_len", type=int, default=1,
        help="keep items whose input is exactly this many tokens (park-country: 5); role-only prompts then "
             "replace each input with that many filler words so lengths stay equal")
    parser.add_argument("--random_demos", action="store_true",
        help="demos sampled with distinct outputs instead of the items before the query (park-country)")
    args = parser.parse_args()

    model_dir = os.path.join(args.save_root, args.model_name.split("/")[-1])
    if args.heads:
        candidates = [tuple(int(x) for x in h.split(",")) for h in args.heads]
    else:
        candidates = candidate_heads(os.path.join(model_dir, "across_tasks", "Behavior",
                                                  f"EP_top{args.top_k}_sender_frequency.csv"), args.d_name, args.top_n)
    with open(os.path.join(model_dir, args.d_name, "Heads", "MAPS_nnsight",
                           f"shared_{args.d_name}_heads_p{args.threshold}_k{args.k}.json")) as f:
        lth = {tuple(h) for h in json.load(f)}
    print("candidates:", [f"L{l}H{h}" for l, h in candidates])

    model = LanguageModel(args.model_name, device_map="auto" if torch.cuda.is_available() else "cpu",
                          dispatch=True, dtype=torch.float32, attn_implementation="eager")
    # same prompts as lth_corruption.py / route tests (identical to before for single-token tasks)
    pm = PromptMaker(model.tokenizer, args.d_name, args.n_prompts, args.seed, n_shot=args.n_shot,
                     input_len=args.input_len, random_demos=args.random_demos)

    scores, roles, attn_clean = {}, None, None
    # random_both: same roles, every content word replaced; shuffled_pairs: real words, wrong partners
    for condition in ["clean", "random_both", "shuffled_pairs"]:
        if condition == "random_both" and args.input_len > 1:   # multi-token inputs -> as many filler words
            rng, prompts = random.Random(f"{args.seed}-random_both-multi"), []
            for q in pm.q_idx:
                items = [{"input": " ".join(rng.sample(FILLER_WORDS, args.input_len)), "output": rng.choice(FILLER_WORDS)}
                         for _ in range(args.n_shot)] + [pm.data[q]]
                prompts.append(build_chunk_labeled_prompts(items, model.tokenizer, args.n_shot, [args.n_shot])[0])
        else:
            prompts, _ = pm.make(condition)
        assert len({len(p["input_ids"]) for p in prompts}) == 1, "prompts differ in length"
        roles = prompts[0]["roles"]
        attn = attention_all_layers(model, torch.tensor([p["input_ids"] for p in prompts]), args.batch_size)
        scores[condition] = head_scores(attn, roles)
        if condition == "clean":
            attn_clean, labels_clean = attn, prompts[0]["labels"]
        print(f"{condition}: {len(prompts)} prompts, e.g. {prompts[0]['prompt']!r}")

    # one row per head in the model, with percentiles against all heads
    n_layers, n_heads = scores["clean"]["induction"].shape
    rows = []
    for layer in range(n_layers):
        for head in range(n_heads):
            row = {"head": f"L{layer}H{head}", "layer": layer, "is_candidate": (layer, head) in candidates,
                   "is_lth": (layer, head) in lth}
            for s in SCORES:
                clean, role_only = scores["clean"][s][layer, head].item(), scores["random_both"][s][layer, head].item()
                shuffled = scores["shuffled_pairs"][s][layer, head].item()
                row[s] = clean
                row[f"{s}_ratio"] = role_only / clean if clean > 1e-6 else np.nan
                row[f"{s}_shuffled_ratio"] = shuffled / clean if clean > 1e-6 else np.nan
            rows.append(row)
    df = pd.DataFrame(rows)
    for s in SCORES:
        df[f"{s}_pct"] = df[s].rank(pct=True) * 100

    def label(r) -> str:
        tags = []
        if r["induction_pct"] >= args.percentile:
            tags.append("symbolic_induction" if r["induction_ratio"] >= args.ratio else "content_induction")
        if r["binding_pct"] >= args.percentile and r["binding_ratio"] >= args.ratio:
            tags.append("abstraction_like")
        # output attends to its own input, and not just to whatever sits that far back
        if r["pair_binding_pct"] >= args.percentile and r["pair_binding"] >= 2 * r["pair_binding_control"]:
            tags.append("symbolic_pair_binding" if r["pair_binding_ratio"] >= args.ratio else "content_pair_binding")
        if r["prev_token_pct"] >= args.percentile:
            tags.append("previous_token")
        return "+".join(tags) or "other"
    df["label"] = df.apply(label, axis=1)

    save_dir = os.path.join(model_dir, args.d_name, "sender_classification")
    os.makedirs(os.path.join(save_dir, "attention"), exist_ok=True)
    df.to_csv(os.path.join(save_dir, "all_heads.csv"), index=False)
    cand = df[df["is_candidate"]].set_index("head").loc[[f"L{l}H{h}" for l, h in candidates]].reset_index()
    cand.to_csv(os.path.join(save_dir, "candidates.csv"), index=False)

    cols = ["head", "label", "is_lth", "induction", "induction_pct", "induction_ratio",
            "pair_binding", "pair_binding_pct", "pair_binding_ratio", "pair_binding_shuffled_ratio", "pair_binding_control",
            "binding", "binding_pct", "binding_ratio", "prev_token", "prev_token_pct"]
    print(f"\ncandidates (high = >= {args.percentile:g}th percentile of all {len(df)} heads; role-following ratio >= {args.ratio}):")
    print(cand[cols].to_string(index=False, float_format="%.2f"))
    print("\nlabel counts among candidates:", cand["label"].value_counts().to_dict())
    print("label counts among all heads:  ", df["label"].value_counts().to_dict())

    cand_plot = cand.assign(lth=np.where(cand["is_lth"], "(LTH)", ""))
    plot_score_table(cand_plot, "head", ["induction", "induction_ratio", "pair_binding", "pair_binding_ratio",
                                         "pair_binding_shuffled_ratio", "pair_binding_control", "binding", "binding_ratio", "prev_token"],
                     title=f"{args.d_name}: top senders' attention signatures", text_cols=["label", "lth"],
                     ).write_html(os.path.join(save_dir, "candidates_scores.html"))
    for layer, head in candidates:
        plot_rsa_matrix(attn_clean[:, layer, head].mean(0), labels_clean,
                        f"L{layer}H{head} attention, mean over {attn_clean.shape[0]} clean prompts ({args.d_name}); "
                        f"axis tokens from prompt 1 — rows: query position, columns: key position", zmin=0, zmax=1,
                        value_name="attention weight",
                        ).write_html(os.path.join(save_dir, "attention", f"L{layer}H{head}_attention.html"))
    print(f"\nsaved to {save_dir}")


if __name__ == "__main__":
    main()
