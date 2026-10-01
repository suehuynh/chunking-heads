"""
Plot path-patching results (LOGIT_DIFF) written by path_patching.py.

path_patching.py saves one tensor per receiver to
    <save_root>/<model>/<d_name>/<subdir>/<prompt_type>_L{layer}H{head}_batch_results_tensor.pt
(older runs used L{layer}C{comp}; both load). Current subdirs:
    path_patching                                    Experiment 1: senders above the earliest LTH
    path_patching_per_receiver/zs                    senders above each receiver, zero-shot corrupt
    path_patching_per_receiver/<corruption>_seed<N>  e.g. shuffle_input_seed42
Each tensor has shape [n_sender_layers, n_heads + 1]: rows are sender layers, columns
0..n_heads-1 are attention heads and the last column is that layer's MLP.
Values are the normalized recovery of the clean answer's logit.

Usage:
    python src/shared_utils/plot_utils.py --model_name meta-llama/Llama-3.2-1B-Instruct \
        --d_name country-capital --prompt_type EP --subdir path_patching_per_receiver/shuffle_input_seed42
"""
import argparse
import math
import re
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots

Receiver = tuple[int, int]

_FILENAME_RE = re.compile(r"^(?P<ptype>\w+?)_L(?P<layer>-?\d+)[CH](?P<comp>-?\d+)_batch_results_tensor\.pt$")


def path_patching_dir(save_root: str, model_name: str, d_name: str, subdir: str) -> Path:
    """Directory path_patching.py wrote to; accepts a full or short model name."""
    return Path(save_root) / model_name.split("/")[-1] / d_name / subdir


def load_path_patching_results(pp_dir: Path, prompt_type: str) -> dict[Receiver, torch.Tensor]:
    """{(receiver_layer, receiver_comp): [n_sender_layers, n_heads + 1] tensor}."""
    tensors = {}
    for f in sorted(pp_dir.glob(f"{prompt_type}_L*_batch_results_tensor.pt")):
        m = _FILENAME_RE.match(f.name)
        if m is None or m["ptype"] != prompt_type:
            continue
        tensors[(int(m["layer"]), int(m["comp"]))] = torch.load(f, map_location="cpu")
    if not tensors:
        raise FileNotFoundError(f"no {prompt_type}_L*_batch_results_tensor.pt files in {pp_dir}")
    return dict(sorted(tensors.items()))


def receiver_label(receiver: Receiver) -> str:
    layer, comp = receiver
    return f"L{layer}MLP" if comp == -1 else f"L{layer}H{comp}"


def sender_labels(n_cols: int) -> list[str]:
    """Column labels: H0..H{n_heads-1}, then MLP (the last column)."""
    return [f"H{h}" for h in range(n_cols - 1)] + ["MLP"]


def results_to_dataframe(tensors: dict[Receiver, torch.Tensor]) -> pd.DataFrame:
    """Long format: one row per (receiver, sender)."""
    rows = []
    for (rl, rc), t in tensors.items():
        labels = sender_labels(t.shape[1])
        for sl in range(t.shape[0]):
            for sc, sender in enumerate(labels):
                rows.append({
                    "receiver": receiver_label((rl, rc)),
                    "receiver_layer": rl,
                    "receiver_comp": rc,
                    "sender_layer": sl,
                    "sender": sender,
                    "effect": t[sl, sc].item(),
                })
    return pd.DataFrame(rows)


def top_senders(df: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    """Top-k senders per receiver by effect, receivers in layer order."""
    top = df.sort_values("effect", ascending=False).groupby("receiver", sort=False).head(k)
    return top.sort_values(["receiver_layer", "receiver_comp", "effect"], ascending=[True, True, False])


def plot_receiver_grid(tensors: dict[Receiver, torch.Tensor], title: str, n_cols: int = 2) -> go.Figure:
    """One heatmap per receiver, each on its OWN symmetric color scale and
    colorbar. Shows each receiver's internal structure, but colors are not
    comparable across receivers -- read the colorbar values."""
    receivers = list(tensors)
    n_rows = math.ceil(len(receivers) / n_cols)
    fig = make_subplots(rows=n_rows, cols=n_cols, horizontal_spacing=0.12,
                        subplot_titles=[f"receiver {receiver_label(r)}" for r in receivers])
    for i, receiver in enumerate(receivers):
        t = tensors[receiver].numpy()
        vmax = float(abs(t).max()) or 1.0
        # place this subplot's colorbar just right of its own axes
        suffix = "" if i == 0 else str(i + 1)
        x_dom = fig.layout[f"xaxis{suffix}"].domain
        y_dom = fig.layout[f"yaxis{suffix}"].domain
        fig.add_trace(
            go.Heatmap(
                z=t,
                x=sender_labels(t.shape[1]),
                y=[f"L{l}" for l in range(t.shape[0])],
                colorscale="RdBu", zmid=0, zmin=-vmax, zmax=vmax,
                colorbar=dict(x=x_dom[1] + 0.01, y=(y_dom[0] + y_dom[1]) / 2,
                              len=y_dom[1] - y_dom[0], thickness=12),
            ),
            row=i // n_cols + 1, col=i % n_cols + 1,
        )
    fig.update_layout(title=title, height=500 * n_rows, width=650 * n_cols)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model_name", type=str, required=True,
        help="full or short model name, e.g. meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--d_name", type=str, required=True, help="task, e.g. country-capital")
    parser.add_argument("--prompt_type", type=str, default="EP", choices=["EP", "IP"])
    parser.add_argument("--save_root", type=str, default="output",
        help="root that path_patching.py wrote to (same as its --save_root)")
    parser.add_argument("--subdir", type=str, required=True,
        help="results folder under <save_root>/<model>/<d_name>/, e.g. path_patching, "
             "path_patching_per_receiver/zs, path_patching_per_receiver/shuffle_input_seed42")
    parser.add_argument("--top_k", type=int, default=5, help="senders to print per receiver")
    parser.add_argument("--n_cols", type=int, default=2, help="heatmaps per row")
    args = parser.parse_args()

    pp_dir = path_patching_dir(args.save_root, args.model_name, args.d_name, args.subdir)
    tensors = load_path_patching_results(pp_dir, args.prompt_type)
    print(f"loaded {len(tensors)} receivers from {pp_dir}: {[receiver_label(r) for r in tensors]}")

    df = results_to_dataframe(tensors)
    # outputs live inside the experiment's own folder, so one neutral name is enough
    csv_path = pp_dir / f"{args.prompt_type}_path_patching_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"saved {len(df)} rows to {csv_path}")

    print(f"\ntop {args.top_k} senders per receiver:")
    print(top_senders(df, args.top_k).to_string(index=False))

    model_short = args.model_name.split("/")[-1]
    title = f"Path patching: sender -> each receiver ({model_short}, {args.d_name}, {args.prompt_type}, {args.subdir})"
    html_path = pp_dir / f"{args.prompt_type}_path_patching_summary.html"
    plot_receiver_grid(tensors, title, n_cols=args.n_cols).write_html(html_path)
    print(f"\nsaved heatmaps to {html_path}")


if __name__ == "__main__":
    main()
