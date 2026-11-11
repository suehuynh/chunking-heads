"""Shared helpers for the H-a route tests (attention_transplant.py, path_ablation.py): HF model loading,
prompts identical to lth_corruption.py's (same seeds, demo selection and condition builders), task-head
lists, head-type labels, and decoding a head's output into task-term matches."""
import json
import os
import random
import sys

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lth_corruption import ANIMAL_TERMS, COMPETING_TERMS, FILLER_WORDS, build_condition, load_items  # noqa: E402
from shared_utils.prompt_utils import build_chunk_labeled_prompts  # noqa: E402

Head = tuple[int, int]


def pick_device(name: str) -> str:
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return name


def load_model(model_name: str, device: str):
    tok = AutoTokenizer.from_pretrained(model_name)
    dtype = torch.float32 if device == "cpu" else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype, attn_implementation="eager").to(device).eval()
    return tok, model


def model_dir(save_root: str, model_name: str) -> str:
    return os.path.join(save_root, model_name.split("/")[-1])


def task_heads(save_root: str, model_name: str, task: str, threshold=0.1, k=10) -> list[Head]:
    path = os.path.join(model_dir(save_root, model_name), task, "Heads", "MAPS_nnsight",
                        f"shared_{task}_heads_p{threshold}_k{k}.json")
    with open(path) as f:
        return [tuple(h) for h in json.load(f)]


def head_labels(save_root: str, model_name: str, task: str) -> pd.DataFrame:
    """sender_classification/all_heads.csv with a (layer, head) tuple column `lh`."""
    df = pd.read_csv(os.path.join(model_dir(save_root, model_name), task, "sender_classification", "all_heads.csv"))
    df["lh"] = [(int(r.layer), int(r.head.split("H")[1])) for r in df.itertuples()]
    return df


def term_ids(tok, terms: list[str]) -> torch.Tensor:
    ids = set()
    for t in terms:
        for s in {t, " " + t, t.capitalize(), " " + t.capitalize()}:
            e = tok(s, add_special_tokens=False)["input_ids"]
            if len(e) == 1:
                ids.add(e[0])
    return torch.tensor(sorted(ids))


def task_terms(tok, task: str) -> dict[str, torch.Tensor]:
    with open(os.path.join("datasets", "dataset_info", "task_relation_dict.json"), encoding="utf-8") as f:
        rel = json.load(f)
    out = {"target": term_ids(tok, rel[task]), "animal": term_ids(tok, ANIMAL_TERMS)}
    if task in COMPETING_TERMS:   # e.g. "opposite" for synonym: reported as match_/score_competing
        out["competing"] = term_ids(tok, COMPETING_TERMS[task])
    return out


class PromptMaker:
    """Builds the same prompts as lth_corruption.py for any of its conditions."""

    def __init__(self, tok, task: str, n_prompts=100, seed=42, n_shot=5, input_len=1, random_demos=False,
                 swap_task=None):
        self.tok, self.seed, self.n_shot = tok, seed, n_shot
        self.data = load_items(task, tok, input_len, swap_task)
        if not self.data:
            raise ValueError(f"{task}: no items with a {input_len}-token input and a 1-token output; "
                             f"multi-token inputs need --input_len (park-country: --input_len 5 --random_demos)")
        rng = random.Random(seed)
        self.q_idx = sorted(rng.sample(range(len(self.data)), min(n_prompts, len(self.data))))
        self.demos = {}
        for q in self.q_idx:
            if random_demos:
                pool, seen = [], {self.data[q]["output"]}
                for i in random.Random(f"{seed}-order-{q}").sample(range(len(self.data)), len(self.data)):
                    if self.data[i]["output"] not in seen and len(pool) < n_shot:
                        pool.append(self.data[i]); seen.add(self.data[i]["output"])
                self.demos[q] = pool
            else:
                self.demos[q] = [self.data[(q - n_shot + j) % len(self.data)] for j in range(n_shot)]

    def make(self, condition: str) -> tuple[list[dict], torch.Tensor]:
        """(prompt dicts with input_ids/roles, answer token ids)."""
        crng = random.Random(f"{self.seed}-{condition}")
        prompts, answers = [], []
        for q in self.q_idx:
            items, ans = build_condition(condition, self.demos[q], self.data[q], crng)
            prompts.append(build_chunk_labeled_prompts(items, self.tok, self.n_shot, [self.n_shot])[0])
            answers.append(self.tok(" " + ans, add_special_tokens=False)["input_ids"][0])
        return prompts, torch.tensor(answers)

    def scrambled_delimiters(self, prompts: list[dict]) -> list[dict]:
        """Same prompts with every ':'/';' separator token replaced by a random single-token filler word
        (different per position), so the repeated X:Y; template is gone but all content tokens stay put."""
        fill = [self.tok(" " + w, add_special_tokens=False)["input_ids"] for w in FILLER_WORDS]
        fill = [f[0] for f in fill if len(f) == 1]
        rng = random.Random(f"{self.seed}-scramble")
        out = []
        for p in prompts:
            ids = list(p["input_ids"])
            for i, r in enumerate(p["roles"]):
                if r == "separator":
                    ids[i] = rng.choice(fill)
            out.append({**p, "input_ids": ids})
        return out


def positions(roles: list[str], role: str) -> list[int]:
    return [i for i, r in enumerate(roles) if r == role]


def decode_head(model, z_out: torch.Tensor, ids: dict[str, torch.Tensor], k=10) -> dict[str, torch.Tensor]:
    """z_out [n, d_model] (a head's output) -> final norm -> unembed; per term list: top-k match and
    score (best term logit minus mean logit)."""
    logits = (model.model.norm(z_out) @ model.lm_head.weight.T).float()
    top = logits.topk(k, -1).indices
    res = {}
    for name, t in ids.items():
        t = t.to(logits.device)
        res[f"match_{name}"] = torch.isin(top, t).any(-1).float()
        res[f"score_{name}"] = logits[:, t].max(-1).values - logits.mean(-1)
    return res


def control_heads(heads: list[Head], exclude: set, n_heads: int, seed, rep: int) -> list[Head]:
    """Random heads with the same per-layer counts as `heads`, none in `exclude`."""
    rng = random.Random(f"{seed}-ctrl-{len(heads)}-{rep}")
    per = {}
    for l, _ in heads:
        per[l] = per.get(l, 0) + 1
    out = []
    for l, c in sorted(per.items()):
        pool = [h for h in range(n_heads) if (l, h) not in exclude]
        out += [(l, h) for h in rng.sample(pool, min(c, len(pool)))]
    return out
