# research_tools/plot_runs.py
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt

def load_log(path: Path) -> List[dict]:
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

def collect_testsets(logs_by_name: Dict[str, List[dict]]) -> List[str]:
    # 로그들에서 등장하는 모든 test셋 키(acc 딕셔너리의 키)를 모음
    keys = set()
    for rows in logs_by_name.values():
        for r in rows:
            if "acc" in r:
                keys.update(r["acc"].keys())
    return sorted(list(keys))

def extract_curve(rows: List[dict], test_key: str) -> Tuple[List[int], List[float]]:
    xs, ys = [], []
    for r in rows:
        if "acc" not in r or test_key not in r["acc"]:
            continue
        xs.append(int(r.get("epoch", len(xs)+1)))
        ys.append(float(r["acc"][test_key]))
    return xs, ys

def final_values(rows: List[dict]) -> Dict[str, float]:
    # 마지막 epoch의 acc 딕셔너리 반환 (없으면 빈 dict)
    if not rows:
        return {}
    last = rows[-1]
    return {k: float(v) for k, v in last.get("acc", {}).items()}

def best_values(rows: List[dict]) -> Dict[str, float]:
    # 각 test셋별 epoch 중 최댓값
    best: Dict[str, float] = {}
    for r in rows:
        for k, v in r.get("acc", {}).items():
            v = float(v)
            if (k not in best) or (v > best[k]):
                best[k] = v
    return best

def plot_epoch_curves(logs_by_name: Dict[str, List[dict]], test_keys: List[str], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for tk in test_keys:
        plt.figure(figsize=(7,5))
        for name, rows in logs_by_name.items():
            xs, ys = extract_curve(rows, tk)
            if xs and ys:
                plt.plot(xs, ys, label=name)
        plt.title(f"Accuracy per Epoch — {tk}")
        plt.xlabel("Epoch"); plt.ylabel("Accuracy")
        plt.grid(True, alpha=0.3)
        plt.legend()
        fig_path = out_dir / f"epoch_curves__{tk}.png"
        plt.tight_layout(); plt.savefig(fig_path, dpi=150); plt.close()
        print(f"[PLOT] {fig_path}")

def plot_bar_summary(summary: Dict[str, Dict[str, float]], title: str, out_path: Path):
    """
    summary: {run_name: {test_key: value}}
    """
    runs = list(summary.keys())
    # 모든 test_key의 합집합
    test_keys = sorted({k for d in summary.values() for k in d.keys()})
    import numpy as np
    x = np.arange(len(test_keys))
    width = 0.8 / max(1, len(runs))  # 막대 폭

    plt.figure(figsize=(9,5))
    for i, run in enumerate(runs):
        vals = [summary[run].get(tk, float("nan")) for tk in test_keys]
        plt.bar(x + i*width, vals, width=width, label=run)
    plt.xticks(x + (len(runs)-1)*width/2, test_keys, rotation=20)
    plt.ylim(0, 1.0)
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[PLOT] {out_path}")

def save_table_csv(best_last: Dict[str, Dict[str, Dict[str, float]]], out_path: Path):
    """
    best_last 구조:
    {
      "best": {run: {test_key: acc}},
      "last": {run: {test_key: acc}},
    }
    """
    rows = []
    all_runs = sorted(best_last["best"].keys())
    all_tests = sorted({tk for run in all_runs for tk in best_last["best"][run].keys() | best_last["last"][run].keys()})
    # CSV 작성
    header = ["metric","run"] + all_tests
    rows.append(",".join(header))
    for metric in ["best", "last"]:
        for run in all_runs:
            vals = [f"{best_last[metric][run].get(tk, float('nan')):.4f}" if tk in best_last[metric][run] else "" for tk in all_tests]
            rows.append(",".join([metric, run] + vals))
    out_path.write_text("\n".join(rows), encoding="utf-8")
    print(f"[CSV]  {out_path}")

def main():
    ap = argparse.ArgumentParser(description="Plot curves and bars from multiple log.jsonl runs.")
    ap.add_argument("--run", nargs=2, action="append", metavar=("NAME","LOG_PATH"),
                    required=True, help="예: --run clean outputs/cifar_clean/log.jsonl")
    ap.add_argument("--out", default="outputs/plots", help="그래프/표 저장 폴더")
    args = ap.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    # 로그 로드
    logs_by_name: Dict[str, List[dict]] = {}
    for name, path in args.run:
        p = Path(path)
        assert p.exists(), f"not found: {p}"
        logs_by_name[name] = load_log(p)
        if not logs_by_name[name]:
            print(f"[WARN] empty log: {p}")

    # 테스트셋 키 수집
    test_keys = collect_testsets(logs_by_name)
    if not test_keys:
        print("[ERR] no test keys found in logs (check 'acc' field).")
        return

    # 에폭 곡선
    plot_epoch_curves(logs_by_name, test_keys, out_dir)

    # 요약(최고/마지막)
    best_last = {"best":{}, "last":{}}
    for name, rows in logs_by_name.items():
        best_last["best"][name] = best_values(rows)
        best_last["last"][name] = final_values(rows)

    # 막대그래프(최고값 기준 / 마지막값 기준)
    plot_bar_summary(best_last["best"], "Best Accuracy Summary", out_dir / "bars__best.png")
    plot_bar_summary(best_last["last"], "Last Epoch Accuracy Summary", out_dir / "bars__last.png")

    # CSV 저장
    save_table_csv(best_last, out_dir / "summary__best_last.csv")

    print("[DONE] plots & tables saved in", out_dir)

if __name__ == "__main__":
    main()