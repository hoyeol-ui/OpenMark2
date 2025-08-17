# Tools/plot_report.py
from __future__ import annotations
import argparse, json, os
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

# Plotly (PNG 저장엔 kaleido가 필요)
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# -------------------------
# IO utils
# -------------------------
def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def load_log_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows

def collect_test_keys(logs_by_name: Dict[str, List[dict]]) -> List[str]:
    keys = set()
    for rows in logs_by_name.values():
        for r in rows:
            keys.update(r.get("acc", {}).keys())
    return sorted(list(keys))

def best_last_tables(logs_by_name: Dict[str, List[dict]]) -> (pd.DataFrame, pd.DataFrame):
    """return (best_df, last_df) with columns=test_keys, index=run_name"""
    test_keys = collect_test_keys(logs_by_name)
    best = {}
    last = {}
    for name, rows in logs_by_name.items():
        # best
        best_row = {}
        cur_best = {k: -1 for k in test_keys}
        for r in rows:
            for k, v in r.get("acc", {}).items():
                v = float(v)
                if v > cur_best[k]:
                    cur_best[k] = v
        for k in test_keys:
            best_row[k] = cur_best[k]
        best[name] = best_row

        # last
        last_row = {k: float(rows[-1].get("acc", {}).get(k, np.nan)) for k in test_keys}
        last[name] = last_row

    return pd.DataFrame(best), pd.DataFrame(last)

def epoch_curve_df(name: str, rows: List[dict], test_key: str) -> pd.DataFrame:
    xs, ys = [], []
    for r in rows:
        if "acc" in r and test_key in r["acc"]:
            xs.append(int(r.get("epoch", len(xs) + 1)))
            ys.append(float(r["acc"][test_key]))
    return pd.DataFrame({"run": name, "epoch": xs, "acc": ys})


def class_counts(root: Path) -> pd.DataFrame:
    rows = []
    for d in sorted([p for p in root.iterdir() if p.is_dir()]):
        n = sum(1 for _ in d.rglob("*") if _.suffix.lower() in {".jpg",".jpeg",".png",".webp",".bmp"})
        rows.append({"class": d.name, "count": n})
    return pd.DataFrame(rows)


def load_clip_eval(path: Path) -> float:
    # 파일은 {"acc":0.XXX} 포맷이 아니라 print만 했을 수도 있어서
    # 안전하게 파싱: json이면 열고, 아니면 마지막 줄에서 퍼센트 추출 시도
    try:
        with path.open("r") as f:
            obj = json.load(f)
            # 다양한 키 지원
            for k in ["acc","accuracy","top1","top1_acc"]:
                if k in obj:
                    return float(obj[k])
            # 혹시 다른 구조면 첫 값 반환 시도
            vals = [float(v) for v in obj.values() if isinstance(v,(int,float))]
            if vals:
                return float(vals[0])
    except Exception:
        pass
    return np.nan


def load_linprobe(path: Path) -> Dict[str, float]:
    with Path(path).open("r") as f:
        obj = json.load(f)
    # {"train_dir": "...", "eval": {"test_clean": 0.86, ...}}
    return {k: float(v) for k, v in obj.get("eval", {}).items()}


# -------------------------
# Plot helpers (Plotly)
# -------------------------
def save(fig, png_path: Path, html_path: Path):
    fig.write_html(str(html_path), include_plotlyjs="cdn")
    try:
        # PNG 내보내기 (kaleido 필요: pip install -U kaleido)
        fig.write_image(str(png_path), scale=2, width=png_path.name.endswith(".wide.png") and 1400 or None)
    except Exception as e:
        print(f"[WARN] PNG export failed (install kaleido?): {e}")

def bars_from_df(df: pd.DataFrame, title: str) -> go.Figure:
    # df: rows=runs, cols=testsets
    df = df.copy()
    df.index.name = "run"
    df = df.reset_index().melt(id_vars="run", var_name="testset", value_name="acc")
    fig = px.bar(
        df, x="testset", y="acc", color="run", barmode="group",
        text="acc", title=title, range_y=[0,1.0]
    )
    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig.update_layout(yaxis=dict(tickformat=".0%"), uniformtext_minsize=10, uniformtext_mode="hide")
    return fig

def lines_epoch_curves(dfs: List[pd.DataFrame], title: str) -> go.Figure:
    fig = go.Figure()
    for df in dfs:
        fig.add_trace(go.Scatter(x=df["epoch"], y=df["acc"], mode="lines+markers", name=df["run"].iloc[0]))
    fig.update_layout(title=title, xaxis_title="Epoch", yaxis_title="Accuracy", yaxis=dict(range=[0,1], tickformat=".0%"))
    return fig

def bars_class_counts(df_a: pd.DataFrame, name_a: str, df_b: pd.DataFrame, name_b: str) -> go.Figure:
    df = df_a.merge(df_b, on="class", how="outer", suffixes=(f"_{name_a}", f"_{name_b}")).fillna(0)
    df = df.melt(id_vars="class", var_name="src", value_name="count")
    fig = px.bar(df, x="class", y="count", color="src", barmode="group", title="Class Distribution")
    fig.update_layout(xaxis_tickangle=-20)
    return fig

def bars_simple(mapping: Dict[str, float], title: str, y_to_percent=True) -> go.Figure:
    xs = list(mapping.keys()); ys = list(mapping.values())
    fig = px.bar(x=xs, y=ys, text=[f"{v:.2f}" for v in ys], title=title)
    fig.update_traces(textposition="outside")
    if y_to_percent:
        fig.update_layout(yaxis=dict(range=[0,1], tickformat=".0%"))
    return fig


# -------------------------
# Defaults tailored to your repo
# -------------------------
def default_args() -> dict:
    return dict(
        runs = [
            ("wm_const_500", "outputs/cifar10__resnet__wmconst_subset500_matched/log.jsonl"),
            ("disrupt_500",  "outputs/cifar10_wmconst_disrupt_subset/log.jsonl"),
        ],
        class_dirs = [
            ("const_500",   "data/cifar10/train_wm_const_local__subset500_matched"),
            ("disrupt_500", "data/cifar10/train-wm-const+disrupt"),
        ],
        clip_jsons = [
            ("const",   "outputs/clip_eval__wm_const__subset500_matched.json"),
            ("disrupt", "outputs/clip_eval__disrupt__subset500.json"),
        ],
        linprobe_jsons = [
            ("trainClean",   "outputs/linprobe__trainClean_evalClean+Disrupt.json"),
            ("trainDisrupt", "outputs/linprobe__trainDisrupt_evalClean+Disrupt.json"),
        ],
        out = "outputs/report_plots"
    )


# -------------------------
# Main
# -------------------------
def main():
    dflt = default_args()

    ap = argparse.ArgumentParser(description="Make clean, publication-ready report plots (Plotly).")
    ap.add_argument("--out", default=dflt["out"])
    ap.add_argument("--runs", nargs=2, action="append", metavar=("NAME","LOGJSONL"),
                    help="e.g., --runs wm_const_500 outputs/.../log.jsonl --runs disrupt_500 outputs/.../log.jsonl")
    ap.add_argument("--class_dirs", nargs=2, action="append", metavar=("NAME","DIR"))
    ap.add_argument("--clip_jsons", nargs=2, action="append", metavar=("NAME","JSON"))
    ap.add_argument("--linprobe_jsons", nargs=2, action="append", metavar=("NAME","JSON"))
    args = ap.parse_args()

    out_dir = ensure_dir(Path(args.out))

    # fill defaults if omitted
    runs         = args.runs if args.runs else dflt["runs"]
    class_dirs   = args.class_dirs if args.class_dirs else dflt["class_dirs"]
    clip_jsons   = args.clip_jsons if args.clip_jsons else dflt["clip_jsons"]
    linprobe_js  = args.linprobe_jsons if args.linprobe_jsons else dflt["linprobe_jsons"]

    # 1) Load logs
    logs_by_name = {}
    for name, p in runs:
        pth = Path(p)
        assert pth.exists(), f"not found: {pth}"
        logs_by_name[name] = load_log_jsonl(pth)

    test_keys = collect_test_keys(logs_by_name) or ["test_clean"]

    # 2) Best/Last bars
    best_df, last_df = best_last_tables(logs_by_name)
    fig_best = bars_from_df(best_df, "Best Accuracy Summary")
    save(fig_best, out_dir / "bars__best.png", out_dir / "bars__best.html")

    fig_last = bars_from_df(last_df, "Last Epoch Accuracy Summary")
    save(fig_last, out_dir / "bars__last.png", out_dir / "bars__last.html")

    # 3) Epoch curves (각 test_key 별로 생성)
    for tk in test_keys:
        dfs = [epoch_curve_df(name, rows, tk) for name, rows in logs_by_name.items()]
        fig = lines_epoch_curves(dfs, f"Accuracy per Epoch — {tk}")
        save(fig, out_dir / f"epoch_curve__{tk}.png", out_dir / f"epoch_curve__{tk}.html")

    # 4) Class distribution (const vs disrupt)
    if len(class_dirs) >= 2:
        (n1, d1), (n2, d2) = class_dirs[0], class_dirs[1]
        df1 = class_counts(Path(d1)); df2 = class_counts(Path(d2))
        fig = bars_class_counts(df1, n1, df2, n2)
        save(fig, out_dir / "class_counts.png", out_dir / "class_counts.html")

    # 5) CLIP zero-shot bars
    clip_map = {}
    for name, j in clip_jsons:
        p = Path(j)
        if p.exists():
            # clip_eval 스크립트 출력 JSON은 {"data":..., "acc":...} 형태면 acc 사용
            try:
                with p.open("r") as f:
                    obj = json.load(f)
                # 다양한 키 케이스 지원
                if isinstance(obj, dict):
                    v = obj.get("acc")
                    if v is None and "accuracy" in obj: v = obj["accuracy"]
                    if v is None and "top1" in obj: v = obj["top1"]
                    if v is None and "results" in obj and "acc" in obj["results"]:
                        v = obj["results"]["acc"]
                    clip_map[name] = float(v) if v is not None else np.nan
            except Exception:
                clip_map[name] = np.nan
        else:
            clip_map[name] = np.nan

    if clip_map:
        fig = bars_simple(clip_map, "CLIP Zero-shot Accuracy")
        save(fig, out_dir / "clip_zeroshot_bars.png", out_dir / "clip_zeroshot_bars.html")

    # 6) Linear probe bars
    lp_map_all = {}  # {series_name: {dataset: acc}}
    for series_name, jp in linprobe_js:
        p = Path(jp)
        if p.exists():
            vals = load_linprobe(p)
            lp_map_all[series_name] = vals

    if lp_map_all:
        # tidy
        rows = []
        for series, kv in lp_map_all.items():
            for ds, acc in kv.items():
                rows.append({"series": series, "dataset": ds, "acc": float(acc)})
        df = pd.DataFrame(rows)
        fig = px.bar(df, x="dataset", y="acc", color="series", barmode="group",
                     title="Linear-Probe Accuracy", text="acc")
        fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
        fig.update_layout(yaxis=dict(range=[0,1], tickformat=".0%"))
        save(fig, out_dir / "linprobe_bars.png", out_dir / "linprobe_bars.html")

    # 7) Mini dashboard: 한 페이지로 묶은 HTML
    htmls = [
        "bars__best.html",
        "bars__last.html",
        *(f"epoch_curve__{tk}.html" for tk in test_keys),
        "class_counts.html",
        "clip_zeroshot_bars.html",
        "linprobe_bars.html",
    ]
    htmls = [h for h in htmls if (out_dir / h).exists()]
    dash = ["<h1>OpenMark Report Plots</h1>"]
    for h in htmls:
        dash.append(f"<h2>{h.replace('.html','')}</h2>")
        dash.append(f'<iframe src="{h}" width="100%" height="520" frameborder="0"></iframe>')
        dash.append("<hr/>")
    (out_dir / "dashboard.html").write_text("\n".join(dash), encoding="utf-8")
    print("[DONE] All figures saved to:", out_dir)
    print("      - Open the dashboard:", out_dir / "dashboard.html")


if __name__ == "__main__":
    main()