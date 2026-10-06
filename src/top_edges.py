"""
Top-k senders (by path-patching effect) for each receiver of one task x corruption,
added to a shared JSON under <save_root>/<model>/across_tasks/Behavior/.

Reads the summary CSV plot_utils.py writes to
    <save_root>/<model>/<d_name>/path_patching_per_receiver/<corruption>[_seed<N>]_<allpos|lastpos>/<prompt_type>_path_patching_summary.csv

Usage:
    python src/top_edges.py --d_name country-capital --corruption_type shuffle_input --seed 42 --positions all
    python src/top_edges.py --d_name country-capital --corruption_type zs --positions last
"""
import argparse
import json
import os

import pandas as pd


def results_subdir(corruption_type: str, seed: int, positions: str) -> str:
    """Folder name path_patching.py writes to (the seed only applies to shuffles)."""
    name = corruption_type if corruption_type == "zs" else f"{corruption_type}_seed{seed}"
    return name + ("_allpos" if positions == "all" else "_lastpos")


def top_edges_per_receiver(df: pd.DataFrame, top_k: int) -> dict[str, list[str]]:
    """{receiver: [top_k senders by effect, highest first]}, receivers in layer order."""
    df = df.assign(component="L" + df["sender_layer"].astype(str) + df["sender"])
    top = (
        df.sort_values(["receiver_layer", "receiver_comp", "effect"], ascending=[True, True, False])
        .groupby("receiver", sort=False)
        .head(top_k)
    )
    return top.groupby("receiver", sort=False)["component"].apply(list).to_dict()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--d_name", type=str, required=True, help="task, e.g. country-capital")
    parser.add_argument("--corruption_type", type=str, required=True,
                        choices=["zs", "shuffle_input", "shuffle_output"])
    parser.add_argument("--seed", type=int, default=42, help="shuffle seed of the run (ignored for zs)")
    parser.add_argument("--positions", type=str, default="all", choices=["last", "all"])
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--prompt_type", type=str, default="EP", choices=["EP", "IP"])
    parser.add_argument("--save_root", type=str, default="output")
    args = parser.parse_args()

    model_dir = os.path.join(args.save_root, args.model_name.split("/")[-1])
    runs_dir = os.path.join(model_dir, args.d_name, "path_patching_per_receiver")
    subdir = results_subdir(args.corruption_type, args.seed, args.positions)
    if subdir == "zs_lastpos" and not os.path.isdir(os.path.join(runs_dir, subdir)):
        subdir = "zs"  # runs from before the _lastpos suffix existed
    csv_path = os.path.join(runs_dir, subdir, f"{args.prompt_type}_path_patching_summary.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} not found; run plot_utils.py --subdir path_patching_per_receiver/{subdir} first")

    key = f"{args.d_name} {subdir}"
    result = {key: top_edges_per_receiver(pd.read_csv(csv_path), args.top_k)}
    print(result)

    out_dir = os.path.join(model_dir, "across_tasks", "Behavior")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{args.prompt_type}_top{args.top_k}_edges_per_receiver.json")
    all_results = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            all_results = json.load(f)
    all_results.update(result)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"saved '{key}' to {out_path} ({len(all_results)} entries)")


if __name__ == "__main__":
    main()
