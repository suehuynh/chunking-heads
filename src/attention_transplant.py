"""H-a test a1/a3: do task heads choose WHERE to read by structural role, and does that choice matter for
verbalization? Attention-vs-content split: keep a task head's values from the clean prompt and replace only
its last-token attention row with the row it produces on a donor prompt:

    z_h(last) = sum_j alpha_donor[j] * v_clean[j]      (v from the head's kv group; then W_O as usual)

Donors (same token length as clean, so positions align):
    clean                 sanity check: must reproduce the baseline exactly
    random_both           apple: shoe; table: river; ...; China:      structure only, no task content
    unrelated_outputs     Vietnam: apple; ...; China:                  structure + inputs, outputs removed
    swapped_roles         Hanoi: Vietnam; ...; Beijing:                structure kept, content moved to inputs
    swapped_filler_query  Hanoi: Vietnam; ...; apple:                  as above, without the query confound
    scrambled_delimiters  Vietnam pillow Hanoi rocket ...; China:      content kept, ':'/';' template removed
    shuffled_pattern      clean row permuted over non-BOS positions    negative control
(donors whose length differs from clean, e.g. input-editing ones with multi-token inputs, are skipped)

Readouts per head and donor: the head's own decoded output (task term / animal term match and score), the
donor row's attention mass per role, and model accuracy (the transplant also feeds later layers). One head is
transplanted at a time.

H-a predicts: role-preserving donors (random_both, unrelated_outputs, swapped_*) keep verbalization because
they still point at output slots, which in the clean run hold the task content; scrambled_delimiters and
shuffled_pattern lose it. A content-driven head instead loses it under swapped_roles (its row follows the
content into the input slots) and keeps it under scrambled_delimiters.

Usage:  python src/attention_transplant.py --d_name antonym --device auto
        python src/attention_transplant.py --d_name park-country --input_len 5 --random_demos
Output: output/<model>/<task>/attention_transplant/{per_prompt.csv, summary.csv, run_info.json}
"""
import argparse
import json
import os
import random

import pandas as pd
import torch

from route_utils import (PromptMaker, decode_head, load_model, model_dir, pick_device, positions, task_heads,
                         task_terms)

torch.set_grad_enabled(False)
DONORS = ["clean", "random_both", "unrelated_outputs", "swapped_roles", "swapped_filler_query",
          "scrambled_delimiters", "shuffled_pattern"]
ROLES = ["bos", "input", "separator", "output", "query", "last_token"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model_name", default="meta-llama/Llama-3.2-1B-Instruct")
    ap.add_argument("--d_name", required=True)
    ap.add_argument("--heads", nargs="+", default=None, metavar="LAYER,HEAD", help="default: the task's LTH list")
    ap.add_argument("--donors", nargs="+", default=DONORS, choices=DONORS)
    ap.add_argument("--target", default="clean",
        help="prompt whose values are kept (default clean). For category tasks use animal_outputs: there, "
             "reading output slots gives 'animal' and reading inputs/query gives the task word, so verbalization "
             "shows where the head looks (in clean prompts every content word maps to the task word)")
    ap.add_argument("--n_prompts", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--input_len", type=int, default=1)
    ap.add_argument("--random_demos", action="store_true")
    ap.add_argument("--batch", type=int, default=50)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--save_root", default="output")
    args = ap.parse_args()

    device = pick_device(args.device)
    tok, model = load_model(args.model_name, device)
    cfg = model.config
    d_head = cfg.hidden_size // cfg.num_attention_heads
    group = cfg.num_attention_heads // cfg.num_key_value_heads
    layers = model.model.layers
    heads = ([tuple(int(x) for x in h.split(",")) for h in args.heads] if args.heads
             else task_heads(args.save_root, args.model_name, args.d_name))
    ids = task_terms(tok, args.d_name)
    save_dir = os.path.join(model_dir(args.save_root, args.model_name), args.d_name, "attention_transplant"
                            + ("" if args.target == "clean" else f"_target-{args.target}"))
    os.makedirs(save_dir, exist_ok=True)

    pm = PromptMaker(tok, args.d_name, args.n_prompts, args.seed, input_len=args.input_len, random_demos=args.random_demos)
    clean, answers = pm.make(args.target)   # "clean" below = the target prompt whose values are kept
    roles = clean[0]["roles"]
    L = len(clean[0]["input_ids"])
    assert all(len(p["input_ids"]) == L for p in clean)
    role_pos = {r: positions(roles, r) for r in ROLES}
    print(f"{len(clean)} prompts, length {L}, device {device}, heads {[f'L{l}H{h}' for l, h in heads]}", flush=True)

    def attention_rows(prompts):
        """last-token attention row of every target head: {head: [n, seq]} (float32, cpu)."""
        rows = {h: [] for h in heads}
        X = torch.tensor([p["input_ids"] for p in prompts])
        for s in range(0, len(X), args.batch):
            o = model(X[s:s + args.batch].to(device), output_attentions=True)
            for (l, h) in heads:
                rows[(l, h)].append(o.attentions[l][:, h, -1].float().cpu())
        return {k: torch.cat(v) for k, v in rows.items()}

    # donor attention rows
    donor_rows = {}
    for d in args.donors:
        if d == "shuffled_pattern":
            continue
        if d == "clean" and args.target != "clean":   # the real clean prompt, as a donor
            prompts, _ = pm.make("clean")
        elif d == "clean":
            prompts = clean
        elif d == "scrambled_delimiters":
            prompts = pm.scrambled_delimiters(clean)
        else:
            try:
                prompts, _ = pm.make(d)
            except Exception as e:   # e.g. condition not buildable for this task
                print(f"skip donor {d}: {e}")
                continue
        if any(len(p["input_ids"]) != L for p in prompts):
            print(f"skip donor {d}: token length differs from clean")
            continue
        print(f"donor {d}: {tok.decode(prompts[0]['input_ids'][1:])[:110]!r}", flush=True)
        donor_rows[d] = attention_rows(prompts)
    if args.target != "clean":
        donor_rows = {"target": attention_rows(clean), **donor_rows}
    self_key = "target" if args.target != "clean" else "clean"
    if "shuffled_pattern" in args.donors and self_key in donor_rows:
        g = torch.Generator().manual_seed(args.seed)
        sp = {}
        for h, r in donor_rows[self_key].items():
            r2 = r.clone()
            for b in range(len(r)):   # permute non-BOS positions, per prompt
                perm = torch.randperm(L - 1, generator=g) + 1
                r2[b, 1:] = r[b, perm]
            sp[h] = r2
        donor_rows["shuffled_pattern"] = sp

    X = torch.tensor([p["input_ids"] for p in clean])

    def run(head, alpha):
        """clean forward with `head`'s last-token z rebuilt from alpha [n, seq] and its clean values.
        alpha None -> unmodified. Returns the head's decoded output dict and model correctness."""
        l, h = head
        kv = h // group
        res, correct = [], []
        for s in range(0, len(X), args.batch):
            sl = slice(s, s + args.batch)
            store = {}
            hooks = [layers[l].self_attn.v_proj.register_forward_hook(lambda m, a, o: store.__setitem__("v", o))]
            def pre(m, a):
                z = a[0]
                if alpha is not None:
                    z = z.clone()
                    v = store["v"][:, :, kv * d_head:(kv + 1) * d_head]
                    z[:, -1, h * d_head:(h + 1) * d_head] = torch.einsum(
                        "bs,bsd->bd", alpha[sl].to(v.device, v.dtype), v)
                store["z"] = z[:, -1, h * d_head:(h + 1) * d_head]
                return (z,) + tuple(a[1:])
            hooks.append(layers[l].self_attn.o_proj.register_forward_pre_hook(pre))
            o = model(X[sl].to(device))
            for hk in hooks:
                hk.remove()
            w_o = layers[l].self_attn.o_proj.weight[:, h * d_head:(h + 1) * d_head]
            res.append({k: v.cpu() for k, v in decode_head(model, store["z"] @ w_o.T, ids).items()})
            correct.append((o.logits[:, -1].argmax(-1).cpu() == answers[sl]).float())
        out = {k: torch.cat([r[k] for r in res]) for k in res[0]}
        out["correct"] = torch.cat(correct)
        return out

    rows = []
    for head in heads:
        name = f"L{head[0]}H{head[1]}"
        base = run(head, None)
        for d, drows in [("none", None)] + list(donor_rows.items()):
            r = base if d == "none" else run(head, drows[head])
            if d == self_key:   # sanity: rebuilding z from the clean row must reproduce the baseline
                assert torch.allclose(r["score_target"], base["score_target"], atol=1e-2 if device == "cpu" else 0.3), \
                    f"{name}: clean transplant does not reproduce baseline"
            for b in range(len(X)):
                row = {"head": name, "donor": d, "prompt": b, "correct": r["correct"][b].item()}
                row.update({k: v[b].item() for k, v in r.items() if k != "correct"})
                if drows is not None:
                    for role, pos in role_pos.items():
                        row[f"donor_attn_{role}"] = drows[head][b, pos].sum().item()
                rows.append(row)
        print(name, "done", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(save_dir, "per_prompt.csv"), index=False)
    metrics = [c for c in df.columns if c.startswith(("match_", "score_", "donor_attn_"))] + ["correct"]
    summ = df.groupby(["head", "donor"], sort=False)[metrics].mean()
    summ.to_csv(os.path.join(save_dir, "summary.csv"))
    json.dump({"target": args.target, "heads": [f"L{l}H{h}" for l, h in heads], "donors": list(donor_rows), "n_prompts": len(X),
               "input_len": args.input_len, "random_demos": args.random_demos},
              open(os.path.join(save_dir, "run_info.json"), "w"), indent=2)
    pd.set_option("display.width", 220)
    show = ["match_target", "score_target", "match_animal", "donor_attn_output", "donor_attn_input",
            "donor_attn_query", "correct"]
    print(summ[[c for c in show if c in summ.columns]].round(3).to_string())


if __name__ == "__main__":
    main()
