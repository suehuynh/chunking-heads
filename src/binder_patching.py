"""H-c test: do pair-binding heads write the relation into demo-output positions, where relation LTH
(L8H31 for antonym) read it?

Two prompts per query with IDENTICAL output tokens at identical positions:
    C (clean)       hot: cold; big: small; fast: slow; ...; up:
    M (mismatched)  big: cold; fast: small; ...; hot: slow; ...; up:   (inputs rotated by one)
so only the input each output belongs to differs. Interventions at demo-OUTPUT positions only:
    binder z patch    the binding heads' z (o_proj input slice) from the source run
    control z patch   same number of random same-layer heads (no head-type label, not LTH), --n_controls draws
    residual sweep    the whole residual stream entering layer L (L = 0..9), from the source run
in two directions: M -> C (necessity: does clean verbalization drop?) and C -> M (sufficiency: does it rise?).
Read-outs at the last token: each antonym LTH's decoded output (top-10 contains an antonym term; term score),
L8H31 attention to outputs, and model accuracy.

Usage:  python src/binder_patching.py
Output: output/<model>/antonym/binder_patching/{per_prompt.csv, summary.csv}
"""
import argparse
import json
import os
import random
import sys

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shared_utils.prompt_utils import build_chunk_labeled_prompts, filter_token_pairs_by_length  # noqa: E402

torch.set_grad_enabled(False)
BINDERS = {"binders": [(5, 22), (6, 14), (5, 6)]}   # antonym binders used in the ablation runs


class Shim:
    def __init__(self, tok):
        self.tokenizer = tok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model_name", default="meta-llama/Llama-3.2-1B-Instruct")
    ap.add_argument("--d_name", default="antonym")
    ap.add_argument("--n_prompts", type=int, default=100)
    ap.add_argument("--n_controls", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch", type=int, default=50)
    ap.add_argument("--save_root", default="output", help="where LTH lists / head labels are read from")
    ap.add_argument("--out_root", default=None, help="where results go (default: save_root)")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, dtype=torch.float32, attn_implementation="eager").eval()
    cfg = model.config
    d_head = cfg.hidden_size // cfg.num_attention_heads
    layers = model.model.layers
    W_U, final_norm = model.lm_head.weight, model.model.norm
    mdir = os.path.join(args.save_root, args.model_name.split("/")[-1])
    save_dir = os.path.join(args.out_root or args.save_root, args.model_name.split("/")[-1], args.d_name, "binder_patching")
    os.makedirs(save_dir, exist_ok=True)

    lth = [tuple(x) for x in json.load(open(os.path.join(
        mdir, args.d_name, "Heads", "MAPS_nnsight", f"shared_{args.d_name}_heads_p0.1_k10.json")))]
    labels = pd.read_csv(os.path.join(mdir, args.d_name, "sender_classification", "all_heads.csv"))
    # all pair-binding heads (content or symbolic) below the relation LTH's layer
    pb = labels[labels.label.str.contains("pair_binding") & (labels.layer < 8)]
    BINDERS["all_pair_binding"] = [(int(r.layer), int(r.head.split("H")[1])) for r in pb.itertuples()]
    labelled = {(int(r.layer), int(r.head.split("H")[1])) for r in labels[labels.label != "other"].itertuples()}

    terms = json.load(open("datasets/dataset_info/task_relation_dict.json", encoding="utf-8"))[args.d_name]
    ids = set()
    for t in terms:
        for s in {t, " " + t, t.capitalize(), " " + t.capitalize()}:
            e = tok(s, add_special_tokens=False)["input_ids"]
            if len(e) == 1:
                ids.add(e[0])
    term_ids = torch.tensor(sorted(ids))

    # ---------- prompts ----------
    data = filter_token_pairs_by_length(json.load(open(f"datasets/abstractive/{args.d_name}.json", encoding="utf-8")),
                                        Shim(tok), 1, 1)
    rng = random.Random(args.seed)
    q_idx = sorted(rng.sample(range(len(data)), min(args.n_prompts, len(data))))
    P = {"C": [], "M": []}
    answers = []
    for q in q_idx:
        demos = [data[(q - 5 + j) % len(data)] for j in range(5)]
        mism = [{"input": demos[(j + 1) % 5]["input"], "output": d["output"]} for j, d in enumerate(demos)]
        P["C"].append(build_chunk_labeled_prompts(demos + [data[q]], tok, 5, [5])[0])
        P["M"].append(build_chunk_labeled_prompts(mism + [data[q]], tok, 5, [5])[0])
        answers.append(tok(" " + data[q]["output"], add_special_tokens=False)["input_ids"][0])
    for k in P:
        assert len({len(p["input_ids"]) for p in P[k]}) == 1
    assert len(P["C"][0]["input_ids"]) == len(P["M"][0]["input_ids"])
    roles = P["C"][0]["roles"]
    out_pos = [i for i, r in enumerate(roles) if r == "output"]
    for a, b in zip(P["C"], P["M"]):   # identical output tokens at identical positions
        assert [a["input_ids"][i] for i in out_pos] == [b["input_ids"][i] for i in out_pos]
    print(f"{len(q_idx)} prompts; e.g.\n C {P['C'][0]['prompt']!r}\n M {P['M'][0]['prompt']!r}", flush=True)
    answers = torch.tensor(answers)
    ids_t = {k: torch.tensor([p["input_ids"] for p in v]) for k, v in P.items()}
    lth_layers = sorted({l for l, _ in lth})

    def run(key, z_patch=None, resid_patch=None, capture_layers=()):
        """z_patch: {layer: (heads, source z [n, seq, d_model])}; resid_patch: (layer, source hidden [n, seq, d]).
        Returns per-LTH decoded matches/scores, L8H31 attention to outputs, accuracy, captured z/hidden."""
        out = {"match": {h: [] for h in lth}, "score": {h: [] for h in lth}, "attn_out": [], "correct": [],
               "z": {l: [] for l in capture_layers}, "hid": []}
        X = ids_t[key]
        for s in range(0, len(X), args.batch):
            sl = slice(s, s + args.batch)
            hooks, zin = [], {}
            for l in sorted(set(lth_layers) | set(capture_layers) | set(z_patch or {})):
                def pre(m, a, l=l, sl=sl):
                    x = a[0]
                    if z_patch and l in z_patch:
                        heads, src = z_patch[l]
                        x = x.clone()
                        for h in heads:
                            x[:, out_pos, h * d_head:(h + 1) * d_head] = src[sl][:, out_pos, h * d_head:(h + 1) * d_head]
                    zin[l] = x
                    return (x,) + tuple(a[1:])
                hooks.append(layers[l].self_attn.o_proj.register_forward_pre_hook(pre))
            if resid_patch:
                L, src = resid_patch
                def lpre(m, a, kw, sl=sl, src=src):
                    h = a[0] if a else kw["hidden_states"]
                    h[:, out_pos] = src[sl][:, out_pos]
                hooks.append(layers[L].register_forward_pre_hook(lpre, with_kwargs=True))
            o = model(X[sl], output_hidden_states=True, output_attentions=True)
            for hk in hooks:
                hk.remove()
            for (l, h) in lth:
                z = zin[l][:, -1, h * d_head:(h + 1) * d_head] @ layers[l].self_attn.o_proj.weight[:, h * d_head:(h + 1) * d_head].T
                lg = final_norm(z) @ W_U.T
                out["match"][(l, h)].append(torch.isin(lg.topk(10, -1).indices, term_ids).any(-1).float())
                out["score"][(l, h)].append(lg[:, term_ids].max(-1).values - lg.mean(-1))
            out["attn_out"].append(o.attentions[8][:, 31, -1, out_pos].sum(-1))
            out["correct"].append((o.logits[:, -1].argmax(-1) == answers[sl]).float())
            for l in capture_layers:
                out["z"][l].append(zin[l])
            out["hid"].append(torch.stack(o.hidden_states[:11], 0))   # [L, b, seq, d]: inputs to layers 0..10
        for k in ["match", "score"]:
            out[k] = {h: torch.cat(v) for h, v in out[k].items()}
        out["attn_out"], out["correct"] = torch.cat(out["attn_out"]), torch.cat(out["correct"])
        out["z"] = {l: torch.cat(v) for l, v in out["z"].items()}
        out["hid"] = torch.cat(out["hid"], 1)
        return out

    # controls: random unlabeled non-LTH heads, same per-layer counts as each binder set
    def control_sets(heads, n):
        sets = []
        for r in range(n):
            crng = random.Random(f"{args.seed}-ctrl-{len(heads)}-{r}")
            per = {}
            for l, _ in heads:
                per[l] = per.get(l, 0) + 1
            s = []
            for l, c in per.items():
                pool = [h for h in range(cfg.num_attention_heads) if (l, h) not in labelled and (l, h) not in lth]
                s += [(l, h) for h in crng.sample(pool, c)]
            sets.append(s)
        return sets

    all_heads = sorted({h for v in BINDERS.values() for h in v})
    cap = sorted({l for l, _ in all_heads} | set(range(cfg.num_hidden_layers)))[:8]
    print("source runs ...", flush=True)
    base = {k: run(k, capture_layers=cap) for k in ["C", "M"]}

    rows = []
    def record(direction, intervention, kind, rep, r):
        for b in range(len(q_idx)):
            row = {"direction": direction, "intervention": intervention, "kind": kind, "rep": rep, "prompt": b,
                   "attn_out_L8H31": r["attn_out"][b].item(), "correct": r["correct"][b].item()}
            for h in lth:
                row[f"match_L{h[0]}H{h[1]}"] = r["match"][h][b].item()
                row[f"score_L{h[0]}H{h[1]}"] = r["score"][h][b].item()
            rows.append(row)

    record("C", "none", "baseline", 0, base["C"])
    record("M", "none", "baseline", 0, base["M"])
    for tgt, src in [("C", "M"), ("M", "C")]:
        direction = f"{src}->{tgt}"
        for name, heads in BINDERS.items():
            print(direction, name, [f"L{l}H{h}" for l, h in heads], flush=True)
            zp = {l: ([h for ll, h in heads if ll == l], base[src]["z"][l]) for l in {l for l, _ in heads}}
            record(direction, name, "binder", 0, run(tgt, z_patch=zp))
            for r, cs in enumerate(control_sets(heads, args.n_controls)):
                zp = {l: ([h for ll, h in cs if ll == l], base[src]["z"][l]) for l in {l for l, _ in cs}}
                record(direction, f"{name}_control", "control", r, run(tgt, z_patch=zp))
        for L in range(11):
            record(direction, f"resid_L{L}", "resid", 0, run(tgt, resid_patch=(L, base[src]["hid"][L])))
        print(direction, "done", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(save_dir, "per_prompt.csv"), index=False)
    metrics = ["match_L8H31", "score_L8H31", "attn_out_L8H31", "correct"]
    df["match_all_lth"] = df[[f"match_L{l}H{h}" for l, h in lth]].mean(1)
    summ = df.groupby(["direction", "intervention"], sort=False)[metrics + ["match_all_lth"]].mean()
    summ.to_csv(os.path.join(save_dir, "summary.csv"))
    json.dump({"binders": {k: [f"L{l}H{h}" for l, h in v] for k, v in BINDERS.items()}, "lth": [f"L{l}H{h}" for l, h in lth],
               "n_prompts": len(q_idx)}, open(os.path.join(save_dir, "run_info.json"), "w"), indent=2)
    pd.set_option("display.width", 200)
    print(summ.round(3).to_string())


if __name__ == "__main__":
    main()
