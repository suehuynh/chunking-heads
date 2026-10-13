"""
Chunking-head candidates: how often each sender is in the top-k path-patching senders
of a task's lexical task heads (LTH), across LTH and corruption types.

Reads the JSON top_edges.py builds:
    <save_root>/<model>/across_tasks/Behavior/EP_top10_edges_per_receiver.json
    {"<task> <run>": {LTH: [top-k senders]}}

Per (task, sender):
    count             times in a top-k list, summed over LTH x runs
    count_<run>       the same, for one run (corruption type)
    important_to      LTH whose top-k it is in (either run)
    n_important_to    number of those LTH
    n_downstream_lth  LTH in a higher layer than the sender (all of them were patched)

Cross-task sharing, separately per corruption: a sender is shared within a task when it is in the
top-k of >= --min_lth LTH of that task; n_tasks_shared = number of tasks where it is shared.

Usage:
    python src/submodules/shared_task_recognition_heads.py
    python src/submodules/shared_task_recognition_heads.py --tasks country-capital park-country --top_n 15
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src/, for shared_utils
from shared_utils.plot_utils import plot_stacked_counts

TopEdges = dict[tuple[str, str], dict[str, list[str]]]

DEFAULT_RUNS = ["shuffle_input_seed42_allpos", "shuffle_output_seed42_allpos"]
_COMPONENT_RE = re.compile(r"^L(\d+)(H\d+|MLP)$")


def component_layer(label: str) -> int:
    """'L6H17' or 'L6MLP' -> 6."""
    m = _COMPONENT_RE.match(label)
    if m is None:
        raise ValueError(f"unexpected component label {label!r}")
    return int(m[1])


def load_top_edges(json_path: str, runs: list[str], tasks: list[str] | None = None) -> TopEdges:
    """{(task, run): {LTH: [senders]}} for the requested runs (and tasks)."""
    with open(json_path) as f:
        raw = json.load(f)
    top_edges = {}
    for key, receivers in raw.items():
        task, run = key.split(" ", 1)
        if run in runs and (tasks is None or task in tasks):
            top_edges[(task, run)] = receivers
    if not top_edges:
        raise ValueError(f"no entries for runs {runs} (tasks {tasks}) in {json_path}")

    for task in sorted({t for t, _ in top_edges}):
        missing = [r for r in runs if (task, r) not in top_edges]
        print(f"{task}: {len(runs) - len(missing)}/{len(runs)} runs" + (f", missing {missing}" if missing else ""))
    return top_edges


def sender_frequency(top_edges: TopEdges, runs: list[str]) -> pd.DataFrame:
    """One row per (task, sender): counts, LTH it is important to, and downstream LTH."""
    rows = []
    for task in sorted({t for t, _ in top_edges}):
        task_runs = {run: rec for (t, run), rec in top_edges.items() if t == task}
        lth = sorted({r for rec in task_runs.values() for r in rec}, key=lambda r: (component_layer(r), r))
        senders = {s for rec in task_runs.values() for lst in rec.values() for s in lst}
        for sender in senders:
            layer = component_layer(sender)
            row = {"task": task, "sender": sender, "sender_layer": layer, "n_lth": len(lth), "count": 0}
            for run in runs:
                receivers = task_runs.get(run, {})
                row[f"important_to_{run}"] = [r for r in lth if sender in receivers.get(r, [])]
                row[f"count_{run}"] = len(row[f"important_to_{run}"])  # one top-k list per LTH per run
                row["count"] += row[f"count_{run}"]
            important_to = [r for r in lth if any(sender in rec.get(r, []) for rec in task_runs.values())]
            row["important_to"] = important_to
            row["n_important_to"] = len(important_to)
            row["n_downstream_lth"] = sum(layer < component_layer(r) for r in lth)
            rows.append(row)
    df = pd.DataFrame(rows)
    return df.sort_values(["task", "count", "sender_layer"], ascending=[True, False, True]).reset_index(drop=True)


def shared_candidates(df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """Senders in the top_n (by count) of at least two tasks; one column of counts per task."""
    top = df.groupby("task", sort=False).head(top_n)
    n_tasks = top.groupby("sender")["task"].nunique()
    shared = n_tasks[n_tasks >= 2].index
    table = top[top["sender"].isin(shared)].pivot(index="sender", columns="task", values="count")
    table.insert(0, "n_tasks", n_tasks[table.index])
    return table.sort_values("n_tasks", ascending=False)


def cross_task_sharing(df: pd.DataFrame, runs: list[str], min_lth: int = 2) -> pd.DataFrame:
    """One row per (corruption, sender): in which tasks the sender is shared, i.e. in the
    top-k of >= min_lth LTH of that task under that corruption, and how many tasks that is."""
    tasks = sorted(df["task"].unique())
    rows = []
    for run in runs:
        shared = df[df[f"count_{run}"] >= min_lth]
        for sender, g in shared.groupby("sender"):
            shared_in = sorted(g["task"])
            rows.append({"corruption": run.split("_seed")[0], "sender": sender,
                         "sender_layer": component_layer(sender),
                         **{t: int(t in shared_in) for t in tasks},
                         "n_tasks_shared": len(shared_in), "shared_in": shared_in})
    out = pd.DataFrame(rows)
    return out.sort_values(["corruption", "n_tasks_shared", "sender_layer"],
                           ascending=[True, False, True]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--save_root", type=str, default="output")
    parser.add_argument("--prompt_type", type=str, default="EP", choices=["EP", "IP"])
    parser.add_argument("--top_k", type=int, default=15, help="top-k of the edges JSON to read")
    parser.add_argument("--runs", type=str, nargs="+", default=DEFAULT_RUNS,
        help="run folders to count as corruption types")
    parser.add_argument("--tasks", type=str, nargs="+", default=None, help="subset of tasks (default: all)")
    parser.add_argument("--top_n", type=int, default=20, help="senders per task to print and plot")
    parser.add_argument("--min_lth", type=int, default=2,
        help="a sender is shared within a task when it is in the top-k of at least this many LTH")
    args = parser.parse_args()

    behavior_dir = os.path.join(args.save_root, args.model_name.split("/")[-1], "across_tasks", "Behavior")
    json_path = os.path.join(behavior_dir, f"{args.prompt_type}_top{args.top_k}_edges_per_receiver.json")
    out_prefix = os.path.join(behavior_dir, f"{args.prompt_type}_top{args.top_k}_sender_frequency")

    df = sender_frequency(load_top_edges(json_path, args.runs, args.tasks), args.runs)
    list_cols = ["important_to"] + [f"important_to_{r}" for r in args.runs]
    df.assign(**{c: df[c].str.join(" ") for c in list_cols}).to_csv(out_prefix + ".csv", index=False)
    value_cols = (["count"] + [f"count_{r}" for r in args.runs] + list_cols
                  + ["n_important_to", "n_downstream_lth"])
    nested = {task: g.set_index("sender")[value_cols].to_dict(orient="index") for task, g in df.groupby("task", sort=False)}
    with open(out_prefix + ".json", "w") as f:
        json.dump(nested, f, indent=2)

    cols = ["sender", "count"] + [f"count_{r}" for r in args.runs] + ["n_important_to", "n_downstream_lth"]
    for task, g in df.groupby("task", sort=False):
        print(f"\n{task} ({g['n_lth'].iloc[0]} LTH): top {args.top_n} senders")
        print(g[cols].head(args.top_n).to_string(index=False))

    shared = shared_candidates(df, args.top_n)
    shared_path = os.path.join(behavior_dir, f"{args.prompt_type}_top{args.top_k}_shared_sender_candidates.csv")
    shared.to_csv(shared_path)
    print(f"\nsenders in the top {args.top_n} of >= 2 tasks (count per task):")
    print(shared.to_string())

    html_path = os.path.join(behavior_dir, f"{args.prompt_type}_top{args.top_k}_sender_counts.html")
    fig = plot_stacked_counts(
        df, label_col="sender", stack_cols=[f"count_{r}" for r in args.runs], facet_col="task",
        top_n=args.top_n, stack_names=[r.split("_seed")[0] for r in args.runs],
        title=f"Times each sender is in an LTH's top-{args.top_k} senders",
    )
    fig.write_html(html_path)
    print(f"\nsaved {out_prefix}.csv\nsaved {out_prefix}.json\nsaved {shared_path}\nsaved {html_path}")

    # cross-task sharing, per corruption
    sharing = cross_task_sharing(df, args.runs, args.min_lth)
    tasks = sorted(df["task"].unique())
    sharing_prefix = os.path.join(behavior_dir, f"{args.prompt_type}_top{args.top_k}_cross_task_sharing")
    sharing.assign(shared_in=sharing["shared_in"].str.join(" ")).to_csv(sharing_prefix + ".csv", index=False)
    with open(sharing_prefix + ".json", "w") as f:
        json.dump({c: g.set_index("sender")[["n_tasks_shared", "shared_in"]].to_dict(orient="index")
                   for c, g in sharing.groupby("corruption", sort=False)}, f, indent=2)
    plot_stacked_counts(
        sharing, label_col="sender", stack_cols=tasks, facet_col="corruption", top_n=args.top_n,
        y_title="tasks where shared",
        title=f"Number of tasks where a sender is in the top-{args.top_k} of >= {args.min_lth} LTH (per corruption)",
    ).write_html(sharing_prefix + ".html")

    print(f"\nsenders shared (top-{args.top_k} of >= {args.min_lth} LTH) in >= 2 tasks:")
    multi = sharing[sharing["n_tasks_shared"] >= 2]
    for corruption, g in multi.groupby("corruption", sort=False):
        print(f"  {corruption}: " + ", ".join(f"{s} ({n}: {' '.join(t)})"
                                             for s, n, t in zip(g["sender"], g["n_tasks_shared"], g["shared_in"])))
    both = set.intersection(*(set(g["sender"]) for _, g in multi.groupby("corruption"))) \
        if multi["corruption"].nunique() == len(args.runs) else set()
    print(f"  under every corruption: {sorted(both, key=lambda s: (component_layer(s), s)) or 'none'}")
    print(f"saved {sharing_prefix}.csv\nsaved {sharing_prefix}.json\nsaved {sharing_prefix}.html")
    n_lth = df.groupby("task")["n_lth"].first().to_dict()
    print(f"note: tasks have different numbers of LTH {n_lth}; being shared (>= {args.min_lth} LTH) is easier "
          "in tasks with more LTH.")
    print("note: in all-position runs, deep LTH (L11+) have many senders tied near 0.011, so their top-k "
          "is filled in index order (mostly L0/L1); check per-run counts before trusting early heads.")


if __name__ == "__main__":
    main()
