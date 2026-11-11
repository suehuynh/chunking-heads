"""H-a test a2: do symbolic heads steer a task head's attention (through its query/key) or feed it content
(through its value)? Path-specific mean-ablation of sender heads into ONE receiver head.

For a receiver r at layer L and a sender set S (all in layers < L), let
    delta = sum_{s in S} (z_s - mean z_s) @ W_O^s          (each sender's deviation from its mean output)
The receiver's last-token output is recomputed from its layer-L input x with delta removed only where a path
is ablated, everything else untouched:
    Q path:  query  from x - delta at the last token
    K path:  keys   from x - delta at every position
    V path:  values from x - delta at every position
(paths: Q, K, V, QK, QKV) and written back, so later layers see it too. Also run, for comparison:
    global   the usual mean-ablation of S everywhere (direct + indirect effects, as in lth_corruption --ablate)
Sender sets come from sender_classification labels: symbolic_induction, abstraction_like, pair_binding
(layer < L, receiver excluded), plus --n_controls random unlabelled non-LTH sets with the same per-layer counts.

Readouts: the receiver's decoded output (task / animal term match and score), its attention to each prompt
role, model accuracy.

H-a predicts symbolic-induction effects through Q and/or K (attention to outputs drops, verbalization drops)
with little through V. Content carried by the senders (H3) predicts V effects. Keys at output positions are
written by heads acting AT those positions (abstraction-like / binding heads), so K effects are expected from
those sets rather than from last-token induction heads.

Usage:  python src/path_ablation.py --d_name antonym --receivers 8,31 --device auto
        python src/path_ablation.py --d_name country-capital --receivers 8,13 9,4 --conditions clean animal_outputs
Output: output/<model>/<task>/path_ablation/{per_prompt.csv, summary.csv, run_info.json}
"""
import argparse
import json
import os

import pandas as pd
import torch
from transformers.models.llama.modeling_llama import rotate_half

from route_utils import (PromptMaker, control_heads, decode_head, head_labels, load_model, model_dir, pick_device,
                         positions, task_heads, task_terms)

torch.set_grad_enabled(False)
PATHS = ["Q", "K", "V", "QK", "QKV", "global"]
SENDER_TYPES = {"induction": "symbolic_induction", "abstraction": "abstraction_like", "pair_binding": "pair_binding"}
ROLES = ["bos", "input", "separator", "output", "query", "last_token"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model_name", default="meta-llama/Llama-3.2-1B-Instruct")
    ap.add_argument("--d_name", required=True)
    ap.add_argument("--receivers", nargs="+", default=None, metavar="LAYER,HEAD",
                    help="default: the task's LTH list (layer >= 2)")
    ap.add_argument("--conditions", nargs="+", default=["clean"], help="lth_corruption conditions to run on")
    ap.add_argument("--sender_types", nargs="+", default=list(SENDER_TYPES), choices=list(SENDER_TYPES))
    ap.add_argument("--paths", nargs="+", default=PATHS, choices=PATHS)
    ap.add_argument("--n_controls", type=int, default=5)
    ap.add_argument("--n_prompts", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--input_len", type=int, default=1)
    ap.add_argument("--random_demos", action="store_true")
    ap.add_argument("--batch", type=int, default=50)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--save_root", default="output", help="where LTH lists / head labels are read from")
    ap.add_argument("--out_root", default=None, help="where results go (default: save_root)")
    args = ap.parse_args()

    device = pick_device(args.device)
    tok, model = load_model(args.model_name, device)
    cfg = model.config
    n_heads = cfg.num_attention_heads
    d_head = cfg.hidden_size // n_heads
    group = n_heads // cfg.num_key_value_heads
    layers = model.model.layers
    lth = task_heads(args.save_root, args.model_name, args.d_name)
    receivers = ([tuple(int(x) for x in h.split(",")) for h in args.receivers] if args.receivers
                 else [h for h in lth if h[0] >= 2])
    labels = head_labels(args.save_root, args.model_name, args.d_name)
    labelled = set(labels[labels.label != "other"].lh)
    ids = task_terms(tok, args.d_name)
    save_dir = os.path.join(model_dir(args.out_root or args.save_root, args.model_name), args.d_name, "path_ablation")
    os.makedirs(save_dir, exist_ok=True)

    pm = PromptMaker(tok, args.d_name, args.n_prompts, args.seed, input_len=args.input_len, random_demos=args.random_demos)
    prompts = {c: pm.make(c) for c in args.conditions}
    clean_X = torch.tensor([p["input_ids"] for p in pm.make("clean")[0]])

    def w_o(l, h):
        return layers[l].self_attn.o_proj.weight[:, h * d_head:(h + 1) * d_head]

    # each head's mean z over clean prompts and positions
    sums, count = [0] * len(layers), 0
    for s in range(0, len(clean_X), args.batch):
        zs = {}
        hooks = [layers[l].self_attn.o_proj.register_forward_pre_hook(lambda m, a, l=l: zs.__setitem__(l, a[0].float().sum((0, 1))))
                 for l in range(len(layers))]
        model(clean_X[s:s + args.batch].to(device))
        for hk in hooks:
            hk.remove()
        sums = [sums[l] + zs[l] for l in range(len(layers))]
        count += clean_X[s:s + args.batch].numel()
    means = [x / count for x in sums]   # [d_model] per layer (all heads' z concatenated)

    def run(X, answers, receiver, senders, path):
        """Returns receiver decoded output dict, its attention row, model correctness.
        senders None -> no intervention (baseline, with the manual recompute as a sanity check)."""
        lr, hr = receiver
        kv = hr // group
        attn_mod, ln = layers[lr].self_attn, layers[lr].input_layernorm
        sender_layers = sorted({l for l, _ in senders}) if senders else []
        res, rows_attn, correct = [], [], []
        for s in range(0, len(X), args.batch):
            st = {"z": {}}
            hooks = []
            for l in sender_layers:
                hs = [h for ll, h in senders if ll == l]
                def spre(m, a, l=l, hs=hs):
                    z = a[0]
                    st["z"][l] = z
                    if path == "global":   # replace the senders' z with their means everywhere
                        z = z.clone()
                        for h in hs:
                            z[..., h * d_head:(h + 1) * d_head] = means[l][h * d_head:(h + 1) * d_head].to(z.dtype)
                        return (z,) + tuple(a[1:])
                hooks.append(layers[l].self_attn.o_proj.register_forward_pre_hook(spre))
            def lpre(m, a, kw):
                st["x"] = a[0] if a else kw["hidden_states"]
                st["pe"] = kw["position_embeddings"]
            hooks.append(layers[lr].register_forward_pre_hook(lpre, with_kwargs=True))
            def rpre(m, a):
                x = st["x"]
                if senders and path != "global":
                    delta = 0
                    for l, h in senders:
                        dz = st["z"][l][..., h * d_head:(h + 1) * d_head] - means[l][h * d_head:(h + 1) * d_head].to(x.dtype)
                        delta = delta + dz @ w_o(l, h).T
                else:
                    delta = torch.zeros_like(x)
                xq = x[:, -1:] - (delta[:, -1:] if "Q" in path else 0)
                xk = x - (delta if "K" in path else 0)
                xv = x - (delta if "V" in path else 0)
                cos, sin = st["pe"]
                q = attn_mod.q_proj(ln(xq))[..., hr * d_head:(hr + 1) * d_head]           # [b, 1, d]
                k = attn_mod.k_proj(ln(xk))[..., kv * d_head:(kv + 1) * d_head]           # [b, seq, d]
                v = attn_mod.v_proj(ln(xv))[..., kv * d_head:(kv + 1) * d_head]
                q = q * cos[:, -1:] + rotate_half(q) * sin[:, -1:]
                k = k * cos + rotate_half(k) * sin
                alpha = torch.softmax((q @ k.transpose(1, 2)).float() * attn_mod.scaling, -1)[:, 0]   # [b, seq]
                z_new = torch.einsum("bs,bsd->bd", alpha.to(v.dtype), v)
                z = a[0].clone()
                if senders is None:   # sanity: manual recompute matches the model's own z
                    err = (z_new - z[:, -1, hr * d_head:(hr + 1) * d_head]).abs().max().item()
                    assert err < (1e-3 if z.dtype == torch.float32 else 5e-2), f"recompute mismatch {err}"
                if path != "global":
                    z[:, -1, hr * d_head:(hr + 1) * d_head] = z_new
                    st["alpha"] = alpha
                else:   # global ablation: the model's own computation already includes it
                    st["alpha"] = None
                st["zr"] = z[:, -1, hr * d_head:(hr + 1) * d_head]
                return (z,) + tuple(a[1:])
            hooks.append(attn_mod.o_proj.register_forward_pre_hook(rpre))
            o = model(X[s:s + args.batch].to(device), output_attentions=(path == "global"))
            for hk in hooks:
                hk.remove()
            alpha = st["alpha"] if st["alpha"] is not None else o.attentions[lr][:, hr, -1].float()
            res.append({k: v.cpu() for k, v in decode_head(model, st["zr"] @ w_o(lr, hr).T, ids).items()})
            rows_attn.append(alpha.float().cpu())
            correct.append((o.logits[:, -1].argmax(-1).cpu() == answers[s:s + args.batch]).float())
        out = {k: torch.cat([r[k] for r in res]) for k in res[0]}
        return out, torch.cat(rows_attn), torch.cat(correct)

    rows, info = [], {}
    for cond, (P, answers) in prompts.items():
        X = torch.tensor([p["input_ids"] for p in P])
        role_pos = {r: positions(P[0]["roles"], r) for r in ROLES}

        def record(receiver, sset, kind, rep, path, out, alpha, correct):
            for b in range(len(X)):
                row = {"condition": cond, "receiver": f"L{receiver[0]}H{receiver[1]}", "senders": sset, "kind": kind,
                       "rep": rep, "path": path, "prompt": b, "correct": correct[b].item()}
                row.update({k: v[b].item() for k, v in out.items()})
                for r, pos in role_pos.items():
                    row[f"attn_{r}"] = alpha[b, pos].sum().item()
                rows.append(row)

        for rec in receivers:
            name = f"L{rec[0]}H{rec[1]}"
            out, alpha, corr = run(X, answers, rec, None, "none")
            record(rec, "none", "baseline", 0, "none", out, alpha, corr)
            for stype in args.sender_types:
                sset = [lh for lh in labels[labels.label.str.contains(SENDER_TYPES[stype])].lh
                        if lh[0] < rec[0] and lh != rec]
                if not sset:
                    continue
                info[f"{name}/{stype}"] = [f"L{l}H{h}" for l, h in sset]
                sets = [("sender", 0, sset)] + [("control", r, control_heads(sset, labelled | set(lth), n_heads, args.seed, r))
                                                for r in range(args.n_controls)]
                for kind, rep, heads in sets:
                    for path in args.paths:
                        out, alpha, corr = run(X, answers, rec, heads, path)
                        record(rec, stype, kind, rep, path, out, alpha, corr)
                print(f"{cond} {name} {stype} ({len(sset)} heads) done", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(save_dir, "per_prompt.csv"), index=False)
    metrics = ["match_target", "score_target", "match_animal", "attn_output", "attn_query", "correct"]
    metrics += [m for m in ["match_competing", "score_competing"] if m in df.columns]
    per_rep = df.groupby(["condition", "receiver", "senders", "kind", "path", "rep"], sort=False)[metrics].mean()
    summ = per_rep.groupby(level=[0, 1, 2, 3, 4], sort=False).agg(["mean", "min", "max"])
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    summ.to_csv(os.path.join(save_dir, "summary.csv"))
    json.dump({"sender_sets": info, "receivers": [f"L{l}H{h}" for l, h in receivers], "conditions": args.conditions,
               "n_prompts": len(pm.q_idx), "n_controls": args.n_controls},
              open(os.path.join(save_dir, "run_info.json"), "w"), indent=2)
    pd.set_option("display.width", 250)
    show = summ[[f"{m}_mean" for m in ["match_target", "score_target", "attn_output", "correct"]]
                + ["match_target_min", "match_target_max"]]
    print(show.round(3).to_string())


if __name__ == "__main__":
    main()
