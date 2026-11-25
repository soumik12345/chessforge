"""Compile all evals, diagnostics, logs, and exact param/corpus stats into a Tier-1 paper report."""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from pathlib import Path

from chessforge.models import build_model_config, count_params


def _fmt_pct(v, digits=2):
    return "—" if v is None else f"{100.0 * v:.{digits}f}"


def _fmt_f(v, digits=4):
    return "—" if v is None else f"{v:.{digits}f}"


def _row(label: str, params_m: str, ov: dict) -> str:
    acc = _fmt_pct(ov.get("move_acc"))
    ci = _fmt_pct(ov.get("move_acc_ci95"))
    top3 = _fmt_pct(ov.get("move_acc_top3"))
    ppl = _fmt_f(ov.get("perplexity"), 3)
    ent = _fmt_f(ov.get("policy_entropy"), 3)
    vacc = _fmt_pct(ov.get("value_acc"))
    vnll = _fmt_f(ov.get("value_nll"), 4)
    ahum = _fmt_pct(ov.get("anti_sf_human_acc"))
    aeng = _fmt_pct(ov.get("anti_sf_engine_pick_rate"))
    aprf = _fmt_pct(ov.get("anti_sf_human_pref_prob"))
    return (
        f"| `{label}` | {params_m} | {acc} ± {ci} | {top3} | {ppl} | {ent} | "
        f"{vacc} | {vnll} | {ahum} | {aeng} | {aprf} |"
    )


def build_markdown_report(bundle: dict) -> str:
    lines: list[str] = ["# ChessForge & ChessformerV2 — Final Paper Results Report\n"]

    # 1. Exact Parameter Counts
    lines.append("## 1. Verified Parameter Counts (Use True Counts in Paper)\n")
    lines.append("| Internal Preset | Paper Name | Exact Parameters |")
    lines.append("| :--- | :--- | ---: |")
    paper_names = {
        "maia3-3m": "MAIA-3-3M",
        "maia3-5m": "MAIA-3-5M",
        "maia3-23m": "MAIA-3-23M",
        "maia3-79m": "MAIA-3-79M",
        "chessforge-v2-5m": "ChessformerV2-9M (8.83M)",
        "chessforge-v2-12m": "ChessformerV2-18M (18.17M)",
        "chessforge-v2-28m": "ChessformerV2-35M (35.23M)",
        "chessforge-v2-28m-joint": "ChessformerV2-35M-Joint (35.24M)",
        "chessforge-v2-60m": "ChessformerV2-73M (73.25M)",
        "chessforge-v2-60m-joint": "ChessformerV2-73M-Joint (73.27M)",
    }
    for k, v in bundle["param_counts"].items():
        lines.append(f"| `{k}` | {paper_names.get(k, k)} | {v:,} |")
    lines.append("")

    # 2. Corpus Audit
    diag = bundle.get("diagnostics", {})
    if "corpus_audit" in diag:
        lines.append("## 2. Exact Verified Corpus & Benchmark Audit\n")
        lines.append("```json")
        lines.append(json.dumps(diag["corpus_audit"], indent=2))
        lines.append("```\n")

    # 3. Main Evaluation Tables by Domain
    evals = bundle["evals"]
    header = (
        "| Eval File | Params | Move Acc (%) | Top-3 (%) | PPL ↓ | Ent | "
        "Val Acc (%) | Val NLL ↓ | AntiSF Hum (%) | AntiSF Eng ↓ | AntiSF Pref (%) |"
    )
    sep = "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"

    for title, suffix in [
        ("3. Main Results — `Eval-Titled-Online`", "_online.json"),
        ("4. Main Results — `Eval-FIDE-OTB`", "_otb.json"),
        ("5. Standard Benchmark — `ALLIE`", "_allie.json"),
    ]:
        matched = [(k, v) for k, v in sorted(evals.items()) if k.endswith(suffix)]
        if not matched:
            continue
        lines.append(f"## {title}\n")
        lines.append(header)
        lines.append(sep)
        for k, v in matched:
            ov = v.get("overall", {})
            p_str = "—"
            for tag, pm in [
                ("maia3-3m", "3.15M"), ("maia3-5m", "5.23M"), ("maia3-23m", "22.94M"),
                ("maia3-79m", "78.90M"), ("5m", "8.83M"), ("28m", "35.2M"), ("60m", "73.3M"),
            ]:
                if tag in k:
                    p_str = pm
                    break
            lines.append(_row(k, p_str, ov))
        lines.append("")

    # 6. Joint Move & Thinking-Time Calibration + Mutual Information
    jt_rows = [
        (k, v["overall"])
        for k, v in sorted(evals.items())
        if "marginal_effort_nll" in v.get("overall", {}) or "joint_nll" in v.get("overall", {})
    ]
    if jt_rows:
        lines.append("## 6. Cognitive Effort Prediction & Joint Move–Time Mutual Information\n")
        lines.append(
            "| Eval File | Move Acc (%) | Marginal Effort NLL $H(U\\mid S)$ ↓ | "
            "Posterior Effort NLL $H(U\\mid S,A)$ ↓ | MI $I(A;U\\mid S)$ (nats) ↑ | "
            "Joint NLL ↓ | Effort MAE ↓ | Effort Pearson $r$ ↑ |"
        )
        lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
        for k, ov in jt_rows:
            lines.append(
                f"| `{k}` | {_fmt_pct(ov.get('move_acc'))} | "
                f"{_fmt_f(ov.get('marginal_effort_nll'), 4)} | "
                f"{_fmt_f(ov.get('effort_given_move_nll'), 4)} | "
                f"{_fmt_f(ov.get('effort_move_mi_nats'), 4)} | "
                f"{_fmt_f(ov.get('joint_nll'), 4)} | "
                f"{_fmt_f(ov.get('ponder_effort_mae'), 4)} | "
                f"{_fmt_f(ov.get('ponder_effort_r'), 4)} |"
            )
        lines.append("")

    # 7. Paired McNemar's Significance Tests
    if "mcnemar" in diag:
        lines.append("## 7. Paired McNemar's Statistical Significance Tests\n")
        for dom in ("online", "otb"):
            dm = diag["mcnemar"].get(dom, {})
            if not dm:
                continue
            lines.append(f"### Domain: `{dom}`\n")
            lines.append("| Pair (`A vs B`) | $\\Delta$ Acc (%) | A Right / B Wrong | A Wrong / B Right | $\\chi^2$ | Two-Sided $p$-value |")
            lines.append("| :--- | :---: | ---: | ---: | ---: | ---: |")
            for pair, st in sorted(dm.items()):
                lines.append(
                    f"| `{pair}` | {100.0 * st['delta_acc']:+.3f} | {st['n10']:,} | "
                    f"{st['n01']:,} | {st['chi2']:.2f} | {st['p_value']:.3e} |"
                )
            lines.append("")

    # 8. OTB Value Head 5-Fold Oracle Recalibration
    if "otb_value_recalibration" in diag and diag["otb_value_recalibration"]:
        lines.append("## 8. MAIA-3 FIDE OTB Value Head 5-Fold Cross-Validated Recalibration\n")
        lines.append("| Baseline | Raw OTB Val Acc (%) | Raw OTB Val NLL ↓ | 5-Fold Calibrated Val Acc (%) | 5-Fold Calibrated Val NLL ↓ |")
        lines.append("| :--- | :---: | :---: | :---: | :---: |")
        for m, st in sorted(diag["otb_value_recalibration"].items()):
            lines.append(
                f"| `{m}` | {_fmt_pct(st['raw_value_acc'])} | {_fmt_f(st['raw_value_nll'], 4)} | "
                f"{_fmt_pct(st['cv5_recalibrated_value_acc'])} | {_fmt_f(st['cv5_recalibrated_value_nll'], 4)} |"
            )
        lines.append("")

    # 9. Qualitative Joint-Time Case Studies
    cases = diag.get("joint_time_case_studies", [])
    if cases:
        lines.append("## 9. Qualitative Grandmaster Joint Move–Time Case Studies (Top 5)\n")
        for idx, c in enumerate(cases[:5], 1):
            lines.append(
                f"**Case {idx}** (`{c['self_title']} {c['self_elo']}` vs `{c['oppo_title']} {c['oppo_elo']}`, "
                f"Ply {c['ply']}, Clock `{c['clk_left_sec']}s`, Observed `{c['human_move']}` after `{c['human_ponder_sec']}s`, "
                f"SF `{c['sf_move']}`, $\\Delta_{{cp}}={c['sf_cp_loss']}$, JSD=`{c['jsd_nats']}`):\n"
                f"- FEN: `{c['fen']}`\n"
                f"- Fast Effort Top-3: `{c['fast_top3']}`\n"
                f"- Slow Effort Top-3: `{c['slow_top3']}`\n"
            )

    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="/mnt/maia3data")
    ap.add_argument("--out_json", default="/mnt/maia3data/FINAL_PAPER_BUNDLE_V2.json")
    ap.add_argument("--out_md", default="/mnt/maia3data/FINAL_PAPER_REPORT.md")
    a = ap.parse_args()

    root = Path(a.data_root)
    bundle: dict = {"param_counts": {}, "evals": {}, "diagnostics": {}, "train_logs": {}}

    for p in [
        "maia3-3m", "maia3-5m", "maia3-23m", "maia3-79m",
        "chessforge-v2-5m", "chessforge-v2-12m", "chessforge-v2-28m", "chessforge-v2-60m",
    ]:
        bundle["param_counts"][p] = count_params(build_model_config(p))
    bundle["param_counts"]["chessforge-v2-28m-joint"] = count_params(
        build_model_config("chessforge-v2-28m", {"use_joint_move_time": True, "joint_move_time_rank": 16})
    )
    bundle["param_counts"]["chessforge-v2-60m-joint"] = count_params(
        build_model_config("chessforge-v2-60m", {"use_joint_move_time": True, "joint_move_time_rank": 16})
    )

    diag_path = root / "paper_results" / "diagnostics" / "paper_diagnostics.json"
    if diag_path.exists():
        bundle["diagnostics"] = json.loads(diag_path.read_text())

    for jf in sorted(glob.glob(str(root / "paper_results" / "**" / "*.json"), recursive=True)):
        if os.path.abspath(jf) == os.path.abspath(str(diag_path)):
            continue
        rel = os.path.relpath(jf, str(root / "paper_results"))
        bundle["evals"][rel] = json.loads(Path(jf).read_text())

    for lf in sorted(glob.glob(str(root / "logs" / "*.log"))):
        name = os.path.basename(lf)
        kept = []
        for line in Path(lf).read_text(errors="replace").splitlines():
            if any(tag in line for tag in ("model=", "active positions", "[curriculum]", "[val]", "[test]")):
                kept.append(line)
            elif line.startswith("step "):
                m = re.match(r"step (\d+)\s+", line)
                if m:
                    s = int(m.group(1))
                    if "dpo" in name or "bench" in name or "ablate" in name or s in (100, 1000, 2000) or s % 10000 == 0:
                        kept.append(line)
        bundle["train_logs"][name] = kept

    Path(a.out_json).write_text(json.dumps(bundle, indent=2))
    md = build_markdown_report(bundle)
    Path(a.out_md).write_text(md)
    print(f"Wrote {a.out_json} ({Path(a.out_json).stat().st_size / 1024:.1f} KB)")
    print(f"Wrote {a.out_md} ({Path(a.out_md).stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()