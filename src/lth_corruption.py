"""
H1 steps 1-2: which demo features do lexical task heads (LTH) need to verbalize the task,
and where in the prompt do they read from?

For each prompt condition (same token length as clean):
    clean              Vietnam: Hanoi; England: London; China:
    random_inputs      apple: Hanoi; shoe: London; China:          (outputs kept, pairing broken)
    shuffled_pairs     Vietnam: London; England: Hanoi; China:     (outputs kept, pairing broken)
    unrelated_outputs  Vietnam: apple; England: shoe; China:       (output content removed)
    random_both        apple: shoe; table: river; China:           (only the A:B; structure left)
    swapped_roles      Hanoi: Vietnam; London: England; Beijing:   (roles flipped)
    outputs_at_inputs  Hanoi: apple; London: shoe; China:          (output words, but at input positions)
    inputs_at_outputs  apple: Vietnam; shoe: England; China:       (input words, but at output positions)
    animal_outputs     Vietnam: dog; England: cat; China:          (consistent outputs, wrong category)
    relation_swap      loud: quiet; humble: arrogant; ...; fast:   (synonym task, --swap_task antonym: same
                                                                    inputs, outputs related by another relation)
    zero_shot          China:                                       (no demos; different length, a baseline)
every LTH is scored at the last token by decoding its output (z_h @ W_O^h -> final norm -> unembed):
    maps_target       top-k contains a target term (e.g. "capital"): the identification criterion
    score_target      max logit over target-term tokens minus the mean logit
    score_competing   the same for a competing term list (e.g. "country", for the swapped task)
plus each LTH's attention from the last token to every prompt role (bos/input/separator/output/query),
and the model's accuracy on the condition's correct answer.

Usage:
    python src/lth_corruption.py --d_name country-capital
    python src/lth_corruption.py --d_name antonym --n_prompts 100
    python src/lth_corruption.py --d_name synonym --swap_task antonym --conditions clean shuffled_pairs relation_swap
        (restricts items to those whose input also has a one-token --swap_task output; saved under *_swap-antonym)
New tasks (not in LEGACY_TASKS) drop items whose input or output is itself a target/competing term word
(e.g. "similar" in synonym), so a head cannot match the term list by copying content.
"""
import argparse
import json
import os
import random

import pandas as pd
import torch
from nnsight import LanguageModel

from shared_utils.plot_utils import plot_grouped_bars
from shared_utils.prompt_utils import build_chunk_labeled_prompts, filter_token_pairs_by_length
from shared_utils.wrapper import ModelAccessor, get_accessor_config, get_model_specs
from submodules.identify_heads_early_decode import convert_expand_candidate_strs_to_token_ids_nnsight

torch.set_grad_enabled(False)

CONDITIONS = ["clean", "random_inputs", "shuffled_pairs", "unrelated_outputs", "random_both", "swapped_roles",
              "outputs_at_inputs", "inputs_at_outputs", "animal_outputs", "zero_shot"]
# coherence dose-response: k of the 5 demo outputs are animals (random positions), the rest fillers
COHERENCE_CONDITIONS = [f"animal_k{k}" for k in range(6)]
# anomaly checks: swapped demos with a filler query (no city query); animals at input positions;
# k of the 5 pairs mismatched (outputs rotated among k random demos; k=1 impossible)
EXTRA_CONDITIONS = ["relation_swap", "swapped_filler_query", "animal_inputs", "animal_inputs_filler_outputs"] +                    [f"mismatch_k{k}" for k in (0, 2, 3, 4, 5)]
# single-token animal words: a consistent output category that is not the task's
ANIMAL_WORDS = ["dog", "cat", "horse", "cow", "pig", "sheep", "goat", "lion", "tiger", "bear", "wolf", "fox", "deer",
                "mouse", "duck", "goose", "eagle", "owl", "snake", "frog", "whale", "shark", "monkey", "camel", "turtle"]
ANIMAL_TERMS = ["animal", "animals", "creature", "creatures", "species", "pet", "pets"]
ROLES = ["bos", "input", "separator", "output", "query", "last_token"]
# single-token words with no task meaning, used as random inputs / unrelated outputs
FILLER_WORDS = ["apple", "shoe", "table", "river", "pencil", "cloud", "garden", "window", "candle", "bottle",
                "chair", "guitar", "rabbit", "blanket", "mirror", "ladder", "basket", "kettle", "button", "carpet",
                "hammer", "pillow", "spoon", "wallet", "bucket", "helmet", "rocket", "sock", "tunnel", "violin"]
# terms for the task the swapped prompt asks for (capital -> country); empty -> reuse the target terms
COMPETING_TERMS = {"country-capital": ["country", "countries", "nation", "nations"],
                   "present-past": ["present", "current", "now", "base"],
                   "park-country": ["park", "parks", "national"],
                   "synonym": ["opposite", "reverse", "antonym", "opposing", "contrary"],
                   "antonym": ["synonym", "equivalent", "alike", "similar", "analogous", "interchangeable"]}
# tasks whose results predate the term-word filter in load_items (kept unfiltered so old prompts reproduce)
LEGACY_TASKS = {"country-capital", "antonym", "park-country", "present-past"}


class _TokShim:   # filter_token_pairs_by_length expects an object with .tokenizer
    def __init__(self, tok):
        self.tokenizer = tok


def load_items(task: str, tok, input_len: int = 1, swap_task: str | None = None,
               dataset_folder: str = "datasets/abstractive", project_root: str = "") -> list[dict]:
    """The task's items with a 1-token output and an input_len-token input. Shared by every script so
    prompts match. New tasks: drop items whose input/output is a target or competing term word.
    swap_task: keep only items whose input also has a 1-token output in swap_task (stored as "swap",
    used by the relation_swap condition), different from the item's own output and not a term word."""
    def raw(t):
        with open(os.path.join(project_root, dataset_folder, f"{t}.json"), encoding="utf-8") as f:
            return json.load(f)
    data = filter_token_pairs_by_length(raw(task), _TokShim(tok), input_len=input_len, output_len=1)
    with open(os.path.join(project_root, "datasets", "dataset_info", "task_relation_dict.json"), encoding="utf-8") as f:
        rel = json.load(f)
    bad = {w.lower() for w in rel.get(task, []) + COMPETING_TERMS.get(task, [])}
    if task not in LEGACY_TASKS:
        data = [d for d in data if d["input"].lower() not in bad and d["output"].lower() not in bad]
    if swap_task:
        other = {d["input"]: d["output"] for d in
                 filter_token_pairs_by_length(raw(swap_task), _TokShim(tok), input_len=input_len, output_len=1)}
        data = [{**d, "swap": other[d["input"]]} for d in data
                if d["input"] in other and other[d["input"]] != d["output"] and other[d["input"]].lower() not in bad]
    return data


def _unwrap(saved):
    """nnsight < 0.4 returns a proxy with .value; newer versions return the tensor."""
    return saved.value if hasattr(saved, "value") else saved


def build_condition(condition: str, demos: list[dict], query: dict, rng: random.Random) -> tuple[list[dict], str]:
    """(demos + [query] as {"input", "output"} items, correct answer) for one condition."""
    fillers = rng.sample(FILLER_WORDS, 2 * len(demos))
    if condition == "clean":
        items = demos
    elif condition == "random_inputs":
        items = [{"input": f, "output": d["output"]} for f, d in zip(fillers, demos)]
    elif condition == "shuffled_pairs":   # rotate outputs by one so no demo keeps its own output
        items = [{"input": d["input"], "output": demos[(i + 1) % len(demos)]["output"]} for i, d in enumerate(demos)]
    elif condition == "unrelated_outputs":
        items = [{"input": d["input"], "output": f} for f, d in zip(fillers, demos)]
    elif condition == "random_both":
        items = [{"input": fillers[2 * i], "output": fillers[2 * i + 1]} for i in range(len(demos))]
    elif condition == "outputs_at_inputs":   # demo outputs moved to input positions, fillers as outputs
        items = [{"input": d["output"], "output": f} for f, d in zip(fillers, demos)]
    elif condition == "inputs_at_outputs":   # demo inputs moved to output positions, fillers as inputs
        items = [{"input": f, "output": d["input"]} for f, d in zip(fillers, demos)]
    elif condition == "animal_outputs":      # outputs consistent with each other, but in the wrong category
        items = [{"input": d["input"], "output": a} for a, d in zip(rng.sample(ANIMAL_WORDS, len(demos)), demos)]
    elif condition.startswith("animal_k"):   # k animal outputs at random positions, fillers elsewhere
        k = min(int(condition[len("animal_k"):]), len(demos))
        animal_pos = set(rng.sample(range(len(demos)), k))
        animals = iter(rng.sample(ANIMAL_WORDS, k))
        items = [{"input": d["input"], "output": next(animals) if i in animal_pos else fillers[i]}
                 for i, d in enumerate(demos)]
    elif condition == "swapped_filler_query":   # Hanoi: Vietnam; ...; apple:  (answer is meaningless)
        items = [{"input": d["output"], "output": d["input"]} for d in demos]
        return items + [{"input": fillers[0], "output": fillers[1]}], fillers[1]
    elif condition == "animal_inputs":          # dog: Vietnam; cat: England; ...; China:
        items = [{"input": a, "output": d["input"]} for a, d in zip(rng.sample(ANIMAL_WORDS, len(demos)), demos)]
    elif condition == "animal_inputs_filler_outputs":   # dog: apple; cat: shoe; ...; China:
        items = [{"input": a, "output": f} for a, f in zip(rng.sample(ANIMAL_WORDS, len(demos)), fillers)]
    elif condition.startswith("mismatch_k"):    # rotate outputs among k random demos
        k = min(int(condition[len("mismatch_k"):]), len(demos))
        pos = sorted(rng.sample(range(len(demos)), k)) if k >= 2 else []
        new_out = {p: demos[pos[(j + 1) % len(pos)]]["output"] for j, p in enumerate(pos)}
        items = [{"input": d["input"], "output": new_out.get(i, d["output"])} for i, d in enumerate(demos)]
    elif condition == "relation_swap":   # same inputs, outputs from another relation (needs items with "swap")
        if any("swap" not in d for d in demos):
            raise ValueError("relation_swap needs --swap_task (items without a 'swap' output)")
        items = [{"input": d["input"], "output": d["swap"]} for d in demos]
    elif condition == "zero_shot":           # query only, no demos
        return [query], query["output"]
    elif condition == "swapped_roles":
        items = [{"input": d["output"], "output": d["input"]} for d in demos]
        return items + [{"input": query["output"], "output": query["input"]}], query["input"]
    else:
        raise ValueError(condition)
    return items + [query], query["output"]


def term_ids(model, terms: list[str]) -> torch.Tensor:
    ids, _ = convert_expand_candidate_strs_to_token_ids_nnsight(model, candidate_strs=terms, output_str_tokens=True)
    return torch.tensor(ids)


Head = tuple[int, int]


def head_means(model, input_ids: torch.Tensor, heads: list[Head], batch_size: int) -> dict[Head, torch.Tensor]:
    """Each head's mean z (its slice of the o_proj input) over prompts and positions: the value used
    for mean-ablation. {head: [d_head]}"""
    d_head = get_model_specs(model)["d_head"]
    accessor = ModelAccessor(model, get_accessor_config(model))
    layers = sorted({l for l, _ in heads})
    sums, count = {l: 0 for l in layers}, 0
    for start in range(0, len(input_ids), batch_size):
        saved = {}
        with model.trace(input_ids[start:start + batch_size]):
            for layer in layers:
                saved[layer] = accessor.layers[layer].attention.output.unwrap().input.save()
        for layer in layers:
            z = _unwrap(saved[layer]).float()
            sums[layer] = sums[layer] + z.sum(dim=(0, 1))
        count += z.shape[0] * z.shape[1]
    return {(l, h): (sums[l] / count)[h * d_head:(h + 1) * d_head] for l, h in heads}


def run_condition(model, input_ids: torch.Tensor, lth: list[Head],
                  ablate: dict[Head, torch.Tensor] | None = None) -> tuple[dict, dict, torch.Tensor]:
    """LTH outputs at the last token {head: [n, d_model]}, LTH attention rows {head: [n, seq]},
    and the model's last-token logits [n, vocab]. ablate: {head: mean z}, written over that head's
    z at every position (mean-ablation) before anything downstream reads it."""
    spec = get_model_specs(model)
    d_head = spec["d_head"]
    accessor = ModelAccessor(model, get_accessor_config(model))
    ablate = ablate or {}
    lth_layers = {l for l, _ in lth}
    z_saved, attn_saved = {}, {}
    with model.trace(input_ids):
        for layer in sorted(lth_layers | {l for l, _ in ablate}):   # layer order, as nnsight requires
            z_all = accessor.layers[layer].attention.output.unwrap().input
            for (l, h), mean in ablate.items():
                if l == layer:
                    z_all[:, :, h * d_head:(h + 1) * d_head] = mean.to(z_all.device)
            if layer in lth_layers:
                z_saved[layer] = z_all[:, -1].save()
                attn_saved[layer] = accessor.layers[layer].attention.unwrap().output[1][:, :, -1].save()  # [n, heads, seq]
        logits = model.lm_head.output[:, -1].save()

    outputs, attention = {}, {}
    for layer, head in lth:
        z = _unwrap(z_saved[layer]).float()
        w_o = model.state_dict()[f"model.layers.{layer}.self_attn.o_proj.weight"].float().to(z.device)
        cols = slice(head * d_head, (head + 1) * d_head)
        outputs[(layer, head)] = z[:, cols] @ w_o[:, cols].T
        attention[(layer, head)] = _unwrap(attn_saved[layer])[:, head].float().cpu()
    return outputs, attention, _unwrap(logits).float().cpu()


def verbalization(model, head_out: torch.Tensor, ids: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Per prompt: top-k match (0/1) and max term logit minus mean logit, decoding with the final norm."""
    logits = model.lm_head(model.model.norm(head_out)).float().cpu()       # [n, vocab]
    topk = logits.topk(k, dim=-1).indices
    match = torch.isin(topk, ids).any(dim=-1).float()
    score = logits[:, ids].max(dim=-1).values - logits.mean(dim=-1)
    return match, score


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--d_name", type=str, required=True)
    parser.add_argument("--n_shot", type=int, default=5)
    parser.add_argument("--n_prompts", type=int, default=100, help="max prompts per condition (fewer if the task has fewer)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k", type=int, default=10, help="top-k decoded tokens, as in LTH identification")
    parser.add_argument("--threshold", type=float, default=0.1, help="which LTH list to load")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--save_root", type=str, default="output")
    parser.add_argument("--dataset_folder", type=str, default="datasets/abstractive")
    parser.add_argument("--project_root", type=str, default="")
    parser.add_argument("--conditions", type=str, nargs="+", default=CONDITIONS, choices=CONDITIONS + COHERENCE_CONDITIONS + EXTRA_CONDITIONS,
        help="subset of conditions to run (clean is always run, first)")
    parser.add_argument("--ablate", type=str, nargs="+", default=None, metavar="LAYER,HEAD",
        help="mean-ablate these heads (their z set to its mean over clean prompts and positions) in every condition")
    parser.add_argument("--control", action="store_true",
        help="with --ablate: ablate the same number of random heads from the same layers instead "
             "(excluding the listed heads and the LTH), seeded by --seed")
    parser.add_argument("--control_rep", type=int, default=0,
        help="with --control: which random draw of control heads (0, 1, 2, ...); prompts stay the same")
    parser.add_argument("--input_len", type=int, default=1,
        help="keep items whose input is exactly this many tokens (outputs are always 1 token); park names "
             "are multi-token, so park-country uses e.g. 5. Only conditions that edit outputs keep lengths then")
    parser.add_argument("--random_demos", action="store_true",
        help="demos sampled at random (distinct outputs, none equal to the query's) instead of the 5 items "
             "before the query; needed when the dataset is sorted by output, e.g. park-country")
    parser.add_argument("--swap_task", type=str, default=None,
        help="for relation_swap: task giving each input's alternative output (e.g. antonym for synonym); "
             "restricts ALL conditions to items that have one, and adds _swap-<task> to the run name")
    parser.add_argument("--run_tag", type=str, default=None,
        help="suffix for the output folder, so a run with other conditions does not overwrite the base run")
    parser.add_argument("--ablation_tag", type=str, default=None,
        help="name for the ablated set in the output folder, e.g. binders; default: the head list")
    args = parser.parse_args()

    model_short = args.model_name.split("/")[-1]
    task_dir = os.path.join(args.save_root, model_short, args.d_name)
    with open(os.path.join(task_dir, "Heads", "MAPS_nnsight", f"shared_{args.d_name}_heads_p{args.threshold}_k{args.k}.json")) as f:
        lth = [tuple(h) for h in json.load(f)]
    print("LTH:", [f"L{l}H{h}" for l, h in lth])
    ablate_heads = [tuple(int(x) for x in h.split(",")) for h in args.ablate] if args.ablate else []

    model = LanguageModel(args.model_name, device_map="auto" if torch.cuda.is_available() else "cpu",
                          dispatch=True, dtype=torch.float32, attn_implementation="eager")  # eager: attention weights
    if ablate_heads and args.control:   # same number of random other heads in each listed layer
        rng_ctrl = random.Random(f"{args.seed}-control" + (f"{args.control_rep}" if args.control_rep else ""))
        n_heads = get_model_specs(model)["n_heads"]
        excluded = set(ablate_heads) | set(lth)
        per_layer = {l: sum(1 for x, _ in ablate_heads if x == l) for l, _ in ablate_heads}
        ablate_heads = [(l, h) for l, n in per_layer.items()
                        for h in rng_ctrl.sample([h for h in range(n_heads) if (l, h) not in excluded], n)]
    if ablate_heads:
        print(("control-" if args.control else "") + "ablating:", [f"L{l}H{h}" for l, h in ablate_heads])

    with open(os.path.join(args.project_root, "datasets", "dataset_info", "task_relation_dict.json"), encoding="utf-8") as f:
        target_terms = json.load(f)[args.d_name]
    competing_terms = COMPETING_TERMS.get(args.d_name, target_terms)
    target_ids, competing_ids = term_ids(model, target_terms), term_ids(model, competing_terms)
    animal_ids = term_ids(model, ANIMAL_TERMS)
    print(f"target terms {target_terms} | competing terms {competing_terms} | animal terms {ANIMAL_TERMS}")

    dataset = load_items(args.d_name, model.tokenizer, args.input_len, args.swap_task,
                         args.dataset_folder, args.project_root)
    rng = random.Random(args.seed)
    query_indices = sorted(rng.sample(range(len(dataset)), min(args.n_prompts, len(dataset))))
    print(f"{len(query_indices)} prompts per condition ({len(dataset)} items, input {args.input_len} token(s))")
    demo_sets = {}
    for q in query_indices:
        if args.random_demos:   # fixed per query, so every condition sees the same demos
            pool, seen = [], {dataset[q]["output"]}
            for i in random.Random(f"{args.seed}-order-{q}").sample(range(len(dataset)), len(dataset)):
                if dataset[i]["output"] not in seen and len(pool) < args.n_shot:
                    pool.append(dataset[i]); seen.add(dataset[i]["output"])
            demo_sets[q] = pool
        else:
            demo_sets[q] = [dataset[(q - args.n_shot + j) % len(dataset)] for j in range(args.n_shot)]

    rows, clean_lengths = [], None
    ablate_means = None
    conditions = ["clean"] + [c for c in args.conditions if c != "clean"]
    for condition in conditions:
        cond_rng = random.Random(f"{args.seed}-{condition}")
        prompts, answers = [], []
        n_shot = 0 if condition == "zero_shot" else args.n_shot
        for q in query_indices:
            demos = demo_sets[q]
            items, answer = build_condition(condition, demos, dataset[q], cond_rng)
            prompts.append(build_chunk_labeled_prompts(items, model.tokenizer, n_shot, [n_shot])[0])
            answers.append(answer)
        lengths = [len(p["input_ids"]) for p in prompts]
        if condition != "zero_shot":   # zero-shot has no demos, so it cannot match the clean length
            clean_lengths = clean_lengths or lengths
            assert lengths == clean_lengths, f"{condition}: token lengths differ from clean"
        assert len(set(lengths)) == 1, f"{condition}: prompts differ in length, cannot batch"
        print(f"\n{condition}: e.g. {prompts[0]['prompt']!r} -> {answers[0]!r}")

        answer_ids = torch.tensor([model.tokenizer(" " + a, add_special_tokens=False)["input_ids"][0] for a in answers])
        roles = prompts[0]["roles"]
        for start in range(0, len(prompts), args.batch_size):
            batch = prompts[start:start + args.batch_size]
            input_ids = torch.tensor([p["input_ids"] for p in batch])
            if ablate_heads and ablate_means is None:   # means from the clean prompts (clean runs first)
                ablate_means = head_means(model, torch.tensor([p["input_ids"] for p in prompts]),
                                          ablate_heads, args.batch_size)
            outputs, attention, logits = run_condition(model, input_ids, lth, ablate_means)
            correct = (logits.argmax(dim=-1) == answer_ids[start:start + len(batch)]).float()
            for head in lth:
                match_t, score_t = verbalization(model, outputs[head], target_ids, args.k)
                match_c, score_c = verbalization(model, outputs[head], competing_ids, args.k)
                match_a, score_a = verbalization(model, outputs[head], animal_ids, args.k)
                for b in range(len(batch)):
                    row = {"condition": condition, "lth": f"L{head[0]}H{head[1]}", "prompt": start + b,
                           "maps_target": match_t[b].item(), "score_target": score_t[b].item(),
                           "maps_competing": match_c[b].item(), "score_competing": score_c[b].item(), "maps_animal": match_a[b].item(),
                           "score_animal": score_a[b].item(), "correct": correct[b].item()}
                    for role in ROLES:
                        row[f"attn_{role}"] = attention[head][b, [i for i, r in enumerate(roles) if r == role]].sum().item()
                    rows.append(row)

    df = pd.DataFrame(rows)
    run_name = (f"{args.n_shot}shot_seed{args.seed}" + (f"_swap-{args.swap_task}" if args.swap_task else "")
                + (f"_{args.run_tag}" if args.run_tag else ""))
    if ablate_heads:
        tag = args.ablation_tag or "-".join(args.ablate).replace(",", "_")
        run_name += f"_ablate-{tag}" + ("_control" + (str(args.control_rep) if args.control_rep else "") if args.control else "")
    save_dir = os.path.join(task_dir, "lth_corruption", run_name)
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "run_info.json"), "w") as f:
        json.dump({"conditions": conditions, "ablated": [f"L{l}H{h}" for l, h in ablate_heads],
                   "control": args.control, "n_prompts": len(query_indices), "lth": [f"L{l}H{h}" for l, h in lth]}, f, indent=2)
    df.to_csv(os.path.join(save_dir, "per_prompt.csv"), index=False)

    metrics = (["maps_target", "score_target", "maps_competing", "score_competing", "maps_animal", "score_animal", "correct"]
               + [f"attn_{r}" for r in ROLES])
    per_lth = df.groupby(["condition", "lth"], sort=False)[metrics].mean().reset_index()
    per_lth.to_csv(os.path.join(save_dir, "per_lth.csv"), index=False)
    summary = per_lth.groupby("condition", sort=False)[metrics].mean()
    summary.to_csv(os.path.join(save_dir, "summary.csv"))
    print("\nmean over LTH:")
    print(summary.to_string(float_format="%.3f"))

    long = summary.reset_index().melt(id_vars="condition", value_vars=["maps_target", "correct"],
                                      var_name="metric", value_name="value")
    plot_grouped_bars(long, "condition", "value", "metric",
                      title=f"{args.d_name}: LTH verbalization (top-{args.k} match) and model accuracy by condition",
                      y_title="fraction").write_html(os.path.join(save_dir, "verbalization.html"))
    long = summary.reset_index().melt(id_vars="condition", value_vars=["score_target", "score_competing"],
                                      var_name="metric", value_name="value")
    plot_grouped_bars(long, "condition", "value", "metric",
                      title=f"{args.d_name}: LTH term logit (max - mean) for target vs competing terms",
                      y_title="logit").write_html(os.path.join(save_dir, "term_scores.html"))
    long = summary.reset_index().melt(id_vars="condition", value_vars=[f"attn_{r}" for r in ROLES],
                                      var_name="role", value_name="attention")
    plot_grouped_bars(long, "condition", "attention", "role",
                      title=f"{args.d_name}: LTH attention from the last token, by prompt role",
                      y_title="attention mass").write_html(os.path.join(save_dir, "attention.html"))
    print(f"\nsaved to {save_dir}")


if __name__ == "__main__":
    main()
