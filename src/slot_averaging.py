"""
H-b test: do lexical task heads verbalize the task by averaging the value vectors at demo-output slots?

A head's output is z_h = W_O^h sum_p a_p v_p, with v_p = W_V^{kv(h)} RMSNorm_L(x_p). If the slots share a
category, the shared part of v survives averaging and the item-specific parts cancel. Two parts:

Part A (offline): per-item head outputs z(x) = W_O^h W_V^{kv(h)} RMSNorm_L(x) for single words, averaged
    over k = 1, 2, 4, 8, 16 distinct items and decoded (final norm -> unembed). Item sources:
      *_iso  the word alone after <bos>;   *_ctx  the word at real demo-output positions in 5-shot prompts.
    Questions: does the average decode to the category word? does a single item (k=1) already? does the
    item's own pair (its country) fade with k? Controls: fillers (no category), city/animal mixtures,
    antonym outputs alone vs in context vs in shuffled-pair context (H-c: relation info in the slot).
    Also: cosine between each head's real in-context output and the synthetic mean of its output slots.

Part B (in context, causal): on clean country-capital prompts, overwrite the value vectors (the L9H4
    value-head slice, layer 9) at the 5 output positions, decompose v = mu_cat + r (mu over all in-context
    output slots of that category, r the item-specific residual), and read L9H4:
      mu_city + r_city (identity) | mu_city | mu_animal | mu_animal + r_city | mu_city + r_animal |
      mu_city + noise (orthogonal to mu, |r| and 3|r|) | v - projection on mu_city
    The same on the residual stream entering layer 9 (affects keys, other heads, later LTH).
    H-c check (antonym, L8H31): output-slot values clean -> shuffled-pairs prompt and back.

Usage: python src/slot_averaging.py
Outputs: output/<model>/slot_averaging/{partA_sets.csv, partA_summary.csv, partA_examples.json,
         partA_real_vs_synthetic.csv, partB.csv, partA_curves.html}
"""
import argparse
import json
import os
import random
import sys

import pandas as pd
import plotly.express as px
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lth_corruption import ANIMAL_TERMS, ANIMAL_WORDS, FILLER_WORDS, build_condition  # noqa: E402
from shared_utils.prompt_utils import build_chunk_labeled_prompts, filter_token_pairs_by_length  # noqa: E402

torch.set_grad_enabled(False)
HEADS = {"L9H4": (9, 4), "L7H5": (7, 5), "L12H27": (12, 27), "L8H31": (8, 31), "L11H22": (11, 22)}
KS = [1, 2, 4, 8, 16]


class Shim:   # filter_token_pairs_by_length expects an object with .tokenizer
    def __init__(self, tok):
        self.tokenizer = tok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model_name", default="meta-llama/Llama-3.2-1B-Instruct")
    ap.add_argument("--n_sets", type=int, default=200, help="random item sets per k")
    ap.add_argument("--n_prompts", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save_root", default="output")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, dtype=torch.float32, attn_implementation="eager").eval()
    cfg = model.config
    d_head = cfg.hidden_size // cfg.num_attention_heads
    group = cfg.num_attention_heads // cfg.num_key_value_heads
    layers = model.model.layers
    W_U, final_norm = model.lm_head.weight, model.model.norm
    save_dir = os.path.join(args.save_root, args.model_name.split("/")[-1], "slot_averaging")
    os.makedirs(save_dir, exist_ok=True)

    def term_ids(terms):
        ids = set()
        for t in terms:
            for s in {t, " " + t, t.capitalize(), " " + t.capitalize()}:
                e = tok(s, add_special_tokens=False)["input_ids"]
                if len(e) == 1:
                    ids.add(e[0])
        return torch.tensor(sorted(ids))
    rel = json.load(open("datasets/dataset_info/task_relation_dict.json", encoding="utf-8"))
    TERMS = {"city": term_ids(rel["country-capital"]), "animal": term_ids(ANIMAL_TERMS), "antonym": term_ids(rel["antonym"])}

    def v_slice(layer, head, x):              # residual [n, d] -> value-head vector [n, d_head]
        kv = head // group
        v = layers[layer].self_attn.v_proj(layers[layer].input_layernorm(x))
        return v[..., kv * d_head:(kv + 1) * d_head]

    def w_o(layer, head):
        return layers[layer].self_attn.o_proj.weight[:, head * d_head:(head + 1) * d_head]

    def head_out(layer, head, x):             # residual -> this head's output if it attended only to x
        return v_slice(layer, head, x) @ w_o(layer, head).T

    def decode(z, ids):                       # z [n, d_model] -> (top-10 match, best-term rank, score)
        logits = final_norm(z) @ W_U.T
        best = logits[:, ids].max(-1).values
        match = torch.isin(logits.topk(10, -1).indices, ids).any(-1).float()
        rank = (logits > best[:, None]).sum(-1).float()
        return match, rank, best - logits.mean(-1), logits

    # ---------- data ----------
    def load(task):
        data = json.load(open(f"datasets/abstractive/{task}.json", encoding="utf-8"))
        return filter_token_pairs_by_length(data, Shim(tok), input_len=1, output_len=1)
    data = {t: load(t) for t in ["country-capital", "antonym"]}

    def make_prompts(task, cond):
        d, rng = data[task], random.Random(args.seed)
        q_idx = sorted(rng.sample(range(len(d)), min(args.n_prompts, len(d))))
        crng = random.Random(f"{args.seed}-{cond}")
        prompts, outs = [], []
        for q in q_idx:
            demos = [d[(q - 5 + j) % len(d)] for j in range(5)]
            items, ans = build_condition(cond, demos, d[q], crng)
            p = build_chunk_labeled_prompts(items, tok, 5, [5])[0]
            p["answer"], p["out_words"] = ans, [it["output"] for it in items[:5]]
            p["in_words"] = [it["input"] for it in items[:5]]
            prompts.append(p)
        return prompts

    def forward(prompts, need_layers, value_patch=None, resid_patch=None, batch=50):
        """hidden states entering need_layers at output slots, real head outputs at the last token,
        last-token attention rows, last-token logits. value_patch / resid_patch: (layer, fn(t, out_pos, start))."""
        out_pos = [i for i, r in enumerate(prompts[0]["roles"]) if r == "output"]
        res = {"hid": {l: [] for l in need_layers}, "z": {h: [] for h in HEADS}, "attn": {h: [] for h in HEADS},
               "logits": []}
        for s in range(0, len(prompts), batch):
            ids = torch.tensor([p["input_ids"] for p in prompts[s:s + batch]])
            hooks, zin = [], {}
            for l in {l for l, _ in HEADS.values()}:
                hooks.append(layers[l].self_attn.o_proj.register_forward_pre_hook(
                    lambda m, a, l=l: zin.__setitem__(l, a[0][:, -1].clone())))
            if value_patch:
                l, fn = value_patch
                hooks.append(layers[l].self_attn.v_proj.register_forward_hook(lambda m, a, o, s=s: fn(o, out_pos, s)))
            if resid_patch:
                l, fn = resid_patch
                def pre(m, a, kw, s=s, fn=fn):
                    h = a[0] if a else kw["hidden_states"]
                    fn(h, out_pos, s)
                hooks.append(layers[l].register_forward_pre_hook(pre, with_kwargs=True))
            o = model(ids, output_hidden_states=True, output_attentions=True)
            for hk in hooks:
                hk.remove()
            for l in need_layers:
                res["hid"][l].append(o.hidden_states[l][:, out_pos])
            for name, (l, h) in HEADS.items():
                res["z"][name].append(zin[l][:, h * d_head:(h + 1) * d_head] @ w_o(l, h).T)
                res["attn"][name].append(o.attentions[l][:, h, -1])
            res["logits"].append(o.logits[:, -1])
        res["hid"] = {l: torch.cat(v) for l, v in res["hid"].items()}
        for k in ["z", "attn"]:
            res[k] = {n: torch.cat(v) for n, v in res[k].items()}
        res["logits"] = torch.cat(res["logits"])
        res["out_pos"] = out_pos
        return res

    need = sorted({l for l, _ in HEADS.values()})
    print("running in-context prompts ...", flush=True)
    P = {("country-capital", c): make_prompts("country-capital", c) for c in ["clean", "animal_outputs", "unrelated_outputs"]}
    P.update({("antonym", c): make_prompts("antonym", c) for c in ["clean", "shuffled_pairs"]})
    R = {k: forward(p, need) for k, p in P.items()}

    # ---------- Part A: pools of (item label, residual per layer) ----------
    def iso_pool(words):
        words = sorted(set(words))
        ids = torch.tensor([tok(" " + w)["input_ids"] for w in words])   # [<bos>, word]
        o = model(ids, output_hidden_states=True)
        return words, {l: o.hidden_states[l][:, -1] for l in need}

    def ctx_pool(key):
        words = [w for p in P[key] for w in p["out_words"]]
        return words, {l: R[key]["hid"][l].reshape(-1, cfg.hidden_size) for l in need}

    cc, ant = data["country-capital"], data["antonym"]
    pools = {
        "city_iso": iso_pool([d["output"] for d in cc]), "country_iso": iso_pool([d["input"] for d in cc]),
        "animal_iso": iso_pool(ANIMAL_WORDS), "filler_iso": iso_pool(FILLER_WORDS),
        "city_ctx": ctx_pool(("country-capital", "clean")), "animal_ctx": ctx_pool(("country-capital", "animal_outputs")),
        "filler_ctx": ctx_pool(("country-capital", "unrelated_outputs")),
        "antonym_out_iso": iso_pool([d["output"] for d in ant][:400]),
        "antonym_out_ctx": ctx_pool(("antonym", "clean")), "antonym_out_shuffled_ctx": ctx_pool(("antonym", "shuffled_pairs")),
    }
    city2country = {d["output"]: d["input"] for d in cc}
    country_tok = {c: tok(" " + c, add_special_tokens=False)["input_ids"][0] for c in city2country.values()}

    rng = random.Random(args.seed)
    rows, examples = [], []
    for pname, (words, hid) in pools.items():
        by_item = {}
        for i, w in enumerate(words):
            by_item.setdefault(w, []).append(i)
        items = sorted(by_item)
        for hname, (l, h) in HEADS.items():
            Z = head_out(l, h, hid[l])                                   # [n_vectors, d_model]
            for k in KS:
                if k > len(items):
                    continue
                sets = [[rng.choice(by_item[w]) for w in rng.sample(items, k)] for _ in range(args.n_sets)]
                zbar = torch.stack([Z[s].mean(0) for s in sets])
                rec = {}
                for tname, ids in TERMS.items():
                    m, r, sc, logits = decode(zbar, ids)
                    rec[tname] = (m, r, sc)
                own = []                                                  # rank of each item's own country
                if pname.startswith("city"):
                    for si, s in enumerate(sets):
                        ranks = [(logits[si] > logits[si, country_tok[city2country[words[i]]]]).sum().item() for i in s]
                        own.append(sum(ranks) / len(ranks))
                for si in range(args.n_sets):
                    row = {"pool": pname, "head": hname, "k": k, "set": si}
                    for tname, (m, r, sc) in rec.items():
                        row[f"{tname}_match"], row[f"{tname}_rank"], row[f"{tname}_score"] = m[si].item(), r[si].item(), sc[si].item()
                    if own:
                        row["own_country_rank"] = own[si]
                    rows.append(row)
                if hname in ("L9H4", "L8H31") and k in (1, 16):
                    for si in range(3):
                        top = logits[si].topk(10).indices                # logits of the last term list decoded
                        examples.append({"pool": pname, "head": hname, "k": k,
                                         "items": [words[i] for i in sets[si]][:8],
                                         "top10": [tok.decode(t) for t in (final_norm(zbar[si:si + 1]) @ W_U.T)[0].topk(10).indices]})
    # mixtures: 8 slots, j animals + (8 - j) cities, in context
    for j in [0, 2, 4, 6, 8]:
        cw, ch = pools["city_ctx"]; aw, ah = pools["animal_ctx"]
        for hname, (l, h) in HEADS.items():
            Zc, Za = head_out(l, h, ch[l]), head_out(l, h, ah[l])
            zbar = torch.stack([torch.cat([Za[rng.sample(range(len(aw)), j)], Zc[rng.sample(range(len(cw)), 8 - j)]]).mean(0)
                                for _ in range(args.n_sets)])
            mc, rc, sc_c, _ = decode(zbar, TERMS["city"]); ma, ra, sc_a, _ = decode(zbar, TERMS["animal"])
            for si in range(args.n_sets):
                rows.append({"pool": f"mix_{j}animal_{8 - j}city", "head": hname, "k": 8, "set": si,
                             "city_match": mc[si].item(), "city_rank": rc[si].item(), "city_score": sc_c[si].item(),
                             "animal_match": ma[si].item(), "animal_rank": ra[si].item(), "animal_score": sc_a[si].item()})
    A = pd.DataFrame(rows)
    A.to_csv(os.path.join(save_dir, "partA_sets.csv"), index=False)
    summ = A.groupby(["pool", "head", "k"]).mean(numeric_only=True).drop(columns="set").reset_index()
    summ.to_csv(os.path.join(save_dir, "partA_summary.csv"), index=False)
    json.dump(examples, open(os.path.join(save_dir, "partA_examples.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)

    # real head output vs the synthetic mean of its own output slots (clean prompts)
    rv = []
    for key in [("country-capital", "clean"), ("antonym", "clean")]:
        r = R[key]
        for hname, (l, h) in HEADS.items():
            Zs = head_out(l, h, r["hid"][l])                             # [n, 5, d]
            a = r["attn"][hname][:, r["out_pos"]]                        # [n, 5]
            real, uni, wtd = r["z"][hname], Zs.mean(1), (a[..., None] * Zs).sum(1)
            cos = torch.nn.functional.cosine_similarity
            rv.append({"task": key[0], "head": hname, "attn_on_outputs": a.sum(1).mean().item(),
                       "cos_real_vs_uniform_mean": cos(real, uni).mean().item(),
                       "cos_real_vs_attn_weighted_outputs": cos(real, wtd).mean().item()})
    pd.DataFrame(rv).to_csv(os.path.join(save_dir, "partA_real_vs_synthetic.csv"), index=False)

    # ---------- Part B: causal slot interventions on clean country-capital prompts ----------
    L, H = HEADS["L9H4"]
    kv = H // group
    cc_clean, cc_animal = P[("country-capital", "clean")], P[("country-capital", "animal_outputs")]
    xc, xa = R[("country-capital", "clean")]["hid"][L], R[("country-capital", "animal_outputs")]["hid"][L]   # [n, 5, d]
    spaces = {"value": (v_slice(L, H, xc), v_slice(L, H, xa)), "residual": (xc, xa)}
    g = torch.Generator().manual_seed(args.seed)
    brows = []
    answer_ids = torch.tensor([tok(" " + p["answer"], add_special_tokens=False)["input_ids"][0] for p in cc_clean])
    for space, (Vc, Va) in spaces.items():
        mu_c, mu_a = Vc.reshape(-1, Vc.shape[-1]).mean(0), Va.reshape(-1, Va.shape[-1]).mean(0)
        r_c, r_a = Vc - mu_c, Va - mu_a
        u = mu_c / mu_c.norm()
        noise = torch.randn(Vc.shape, generator=g)
        noise = noise - (noise @ u)[..., None] * u
        noise = noise / noise.norm(dim=-1, keepdim=True) * r_c.norm(dim=-1, keepdim=True)
        conds = {"none": None, "mu_city + r_city (identity)": mu_c + r_c, "mu_city only": mu_c.expand_as(Vc),
                 "mu_animal only": mu_a.expand_as(Vc), "mu_animal + r_city": mu_a + r_c, "mu_city + r_animal": mu_c + r_a,
                 "mu_city + noise (|r|)": mu_c + noise, "mu_city + noise (3|r|)": mu_c + 3 * noise,
                 "remove mu_city direction": Vc - (Vc @ u)[..., None] * u}
        for cname, new in conds.items():
            vp = rp = None
            if new is not None and space == "value":
                def fn(o, pos, s, new=new):
                    n = o.shape[0]
                    o[:, pos, kv * d_head:(kv + 1) * d_head] = new[s:s + n]
                vp = (L, fn)
            elif new is not None:
                def fn(hs, pos, s, new=new):
                    n = hs.shape[0]
                    hs[:, pos] = new[s:s + n]
                rp = (L, fn)
            r = forward(cc_clean, [L], value_patch=vp, resid_patch=rp)
            acc = (r["logits"].argmax(-1) == answer_ids).float().mean().item()
            for hname in ["L9H4", "L11H22", "L12H27"]:
                row = {"task": "country-capital", "space": space, "intervention": cname, "head": hname, "accuracy": acc,
                       "head_out_norm": r["z"][hname].norm(dim=-1).mean().item()}
                for tname in ["city", "animal"]:
                    m, rk, sc, _ = decode(r["z"][hname], TERMS[tname])
                    row[f"{tname}_match"], row[f"{tname}_score"] = m.mean().item(), sc.mean().item()
                brows.append(row)
        print(f"part B {space} done", flush=True)

    # H-c: antonym L8H31, output-slot values swapped between clean and shuffled-pairs prompts
    L8, H8 = HEADS["L8H31"]
    kv8 = H8 // group
    ac, ash = P[("antonym", "clean")], P[("antonym", "shuffled_pairs")]
    v_clean = v_slice(L8, H8, R[("antonym", "clean")]["hid"][L8])
    v_shuf = v_slice(L8, H8, R[("antonym", "shuffled_pairs")]["hid"][L8])
    for cname, prompts, new in [("clean, none", ac, None), ("shuffled, none", ash, None),
                                ("shuffled prompt + clean output-slot values", ash, v_clean),
                                ("clean prompt + shuffled output-slot values", ac, v_shuf)]:
        vp = None
        if new is not None:
            def fn(o, pos, s, new=new):
                o[:, pos, kv8 * d_head:(kv8 + 1) * d_head] = new[s:s + o.shape[0]]
            vp = (L8, fn)
        r = forward(prompts, [L8], value_patch=vp)
        m, rk, sc, _ = decode(r["z"]["L8H31"], TERMS["antonym"])
        brows.append({"task": "antonym", "space": "value", "intervention": cname, "head": "L8H31",
                      "antonym_match": m.mean().item(), "antonym_score": sc.mean().item(),
                      "head_out_norm": r["z"]["L8H31"].norm(dim=-1).mean().item()})
    B = pd.DataFrame(brows)
    B.to_csv(os.path.join(save_dir, "partB.csv"), index=False)

    # ---------- plots + printout ----------
    pd.set_option("display.width", 250)
    cur = summ[summ["pool"].isin(["city_iso", "city_ctx", "country_iso", "animal_iso", "animal_ctx", "filler_iso", "filler_ctx"])]
    long = pd.concat([cur.assign(term="city", match=cur["city_match"]), cur.assign(term="animal", match=cur["animal_match"])])
    px.line(long, x="k", y="match", color="pool", line_dash="term", facet_col="head", markers=True, log_x=True,
            title="Part A: category-term top-10 match of the k-item average of head outputs").write_html(
        os.path.join(save_dir, "partA_curves.html"))
    show = summ[summ["head"].isin(["L9H4", "L7H5", "L12H27"]) & ~summ["pool"].str.startswith("antonym")]
    print("\n=== Part A (country-capital pools): top-10 match / median-ish rank")
    print(show.pivot_table(index=["pool", "head"], columns="k", values=["city_match", "animal_match"]).round(2).to_string())
    print(show[show.pool.str.startswith("city")].pivot_table(index=["pool", "head"], columns="k", values="own_country_rank").round(0).to_string())
    print("\n=== Part A (antonym pools): antonym-term match")
    sa = summ[summ.pool.str.startswith("antonym") & summ["head"].isin(["L8H31", "L9H4"])]
    print(sa.pivot_table(index=["pool", "head"], columns="k", values=["antonym_match", "antonym_score"]).round(2).to_string())
    print("\n=== real vs synthetic"); print(pd.DataFrame(rv).round(2).to_string())
    print("\n=== Part B"); print(B.round(2).to_string())
    print(f"\nsaved to {save_dir}")


if __name__ == "__main__":
    main()
