import csv
import json
import re
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter

BASE_DIR = Path(__file__).resolve().parent

PARSHIFT_DIR = BASE_DIR / "data" / "processed" / "parshift"
RESULTS_DIR = PARSHIFT_DIR / "results"
AGGREGATED_DIR = PARSHIFT_DIR / "aggregated"
REPORTS_DIR = PARSHIFT_DIR / "reports"

EXPECTED_SHIFTS = [
    "A0-XA",
    "A0-X0",
    "A0-XY",
    "A0-AY",
    "AB-BA",
    "AB-B0",
    "AB-X0",
    "AB-XA",
    "AB-XB",
    "AB-A0",
    "AB-BY",
    "AB-XY",
    "AB-AY",
]


def extract_legislature(path: Path):
    match = re.search(r"legislature_(\d+)", str(path))
    return int(match.group(1)) if match else None


def extract_segment(path: Path):
    match = re.search(r"segment_(\d+)_stats", path.name)
    return int(match.group(1)) if match else None


def extract_wave(path: Path):
    text = str(path)
    if "wave_1" in text:
        return "wave_1"
    if "wave_2" in text:
        return "wave_2"
    return "unknown_wave"


def safe_float(value):
    if value is None or value == "":
        return 0.0
    try:
        return float(value)
    except ValueError:
        return 0.0


def safe_int(value):
    if value is None or value == "":
        return 0
    try:
        return int(float(value))
    except ValueError:
        return 0


def load_stats_file(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"{path.name} não contém uma lista JSON.")

    rows = []

    wave = extract_wave(path)
    legislature = extract_legislature(path)
    segment = extract_segment(path)

    for item in data:
        if not isinstance(item, dict):
            continue

        shift = item.get("Pshift")

        row = {
            "wave": wave,
            "legislature": legislature,
            "segment": segment,
            "pshift": shift,
            "frequency": safe_int(item.get("Frequency")),
            "probability": safe_float(item.get("Probability")),
            "p_s_given_d": safe_float(item.get("P(S|D)")),
            "p_s_given_d_c": safe_float(item.get("P(S|D,C)")),
            "change_of_speaker": bool(item.get("Change of Speaker (C)")),
            "directed_remark": bool(item.get("Directed Remark (D)")),
            "source_file": path.name
        }

        rows.append(row)

    return rows


def find_stats_files():
    files = sorted(
        RESULTS_DIR.glob("wave_*/*/*segment_*_stats.json"),
        key=lambda p: (
            extract_wave(p),
            extract_legislature(p) or 999,
            extract_segment(p) or 999,
            p.name
        )
    )

    return files


def write_csv(path: Path, rows):
    if not rows:
        return

    fieldnames = sorted({key for row in rows for key in row.keys()})

    preferred_order = [
        "wave",
        "legislature",
        "segment",
        "pshift",
        "frequency",
        "probability",
        "p_s_given_d",
        "p_s_given_d_c",
        "total_frequency",
        "avg_probability",
        "avg_p_s_given_d",
        "avg_p_s_given_d_c",
        "dominant_pshift",
        "frequency_wave_1",
        "frequency_wave_2",
        "frequency_diff",
        "probability_wave_1",
        "probability_wave_2",
        "probability_diff",
        "source_file"
    ]

    ordered = [f for f in preferred_order if f in fieldnames]
    remaining = [f for f in fieldnames if f not in ordered]
    fieldnames = ordered + remaining

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)


def aggregate_by_wave_legislature(rows):
    grouped = defaultdict(list)

    for row in rows:
        grouped[(row["wave"], row["legislature"], row["pshift"])].append(row)

    output = []

    for (wave, legislature, pshift), values in grouped.items():
        total_frequency = sum(v["frequency"] for v in values)
        avg_probability = sum(v["probability"] for v in values) / len(values)
        avg_p_s_given_d = sum(v["p_s_given_d"] for v in values) / len(values)
        avg_p_s_given_d_c = sum(v["p_s_given_d_c"] for v in values) / len(values)

        output.append({
            "wave": wave,
            "legislature": legislature,
            "pshift": pshift,
            "total_frequency": total_frequency,
            "avg_probability": avg_probability,
            "avg_p_s_given_d": avg_p_s_given_d,
            "avg_p_s_given_d_c": avg_p_s_given_d_c
        })

    return sorted(
        output,
        key=lambda r: (r["wave"], r["legislature"], r["pshift"])
    )


def aggregate_by_wave(rows):
    grouped = defaultdict(list)

    for row in rows:
        grouped[(row["wave"], row["pshift"])].append(row)

    output = []

    for (wave, pshift), values in grouped.items():
        total_frequency = sum(v["frequency"] for v in values)
        avg_probability = sum(v["probability"] for v in values) / len(values)
        avg_p_s_given_d = sum(v["p_s_given_d"] for v in values) / len(values)
        avg_p_s_given_d_c = sum(v["p_s_given_d_c"] for v in values) / len(values)

        output.append({
            "wave": wave,
            "pshift": pshift,
            "total_frequency": total_frequency,
            "avg_probability": avg_probability,
            "avg_p_s_given_d": avg_p_s_given_d,
            "avg_p_s_given_d_c": avg_p_s_given_d_c
        })

    return sorted(output, key=lambda r: (r["wave"], r["pshift"]))


def dominant_shift_by_legislature(rows):
    grouped = defaultdict(list)

    for row in rows:
        grouped[(row["wave"], row["legislature"], row["segment"])].append(row)

    output = []

    for (wave, legislature, segment), values in grouped.items():
        dominant = max(values, key=lambda r: r["frequency"])

        output.append({
            "wave": wave,
            "legislature": legislature,
            "segment": segment,
            "dominant_pshift": dominant["pshift"],
            "frequency": dominant["frequency"],
            "probability": dominant["probability"]
        })

    return sorted(
        output,
        key=lambda r: (r["wave"], r["legislature"], r["segment"])
    )


def compare_waves_by_legislature(rows):
    grouped = defaultdict(dict)

    for row in rows:
        key = (row["legislature"], row["segment"], row["pshift"])
        grouped[key][row["wave"]] = row

    output = []

    for (legislature, segment, pshift), values in grouped.items():
        w1 = values.get("wave_1")
        w2 = values.get("wave_2")

        if not w1 or not w2:
            continue

        output.append({
            "legislature": legislature,
            "segment": segment,
            "pshift": pshift,
            "frequency_wave_1": w1["frequency"],
            "frequency_wave_2": w2["frequency"],
            "frequency_diff": w2["frequency"] - w1["frequency"],
            "probability_wave_1": w1["probability"],
            "probability_wave_2": w2["probability"],
            "probability_diff": w2["probability"] - w1["probability"],
            "p_s_given_d_wave_1": w1["p_s_given_d"],
            "p_s_given_d_wave_2": w2["p_s_given_d"],
            "p_s_given_d_diff": w2["p_s_given_d"] - w1["p_s_given_d"],
            "p_s_given_d_c_wave_1": w1["p_s_given_d_c"],
            "p_s_given_d_c_wave_2": w2["p_s_given_d_c"],
            "p_s_given_d_c_diff": w2["p_s_given_d_c"] - w1["p_s_given_d_c"]
        })

    return sorted(
        output,
        key=lambda r: (r["legislature"], r["segment"], r["pshift"])
    )


def write_report(path: Path, rows, agg_wave, agg_leg, dominant, comparison):
    lines = []

    lines.append("AGGREGATED PARSHIFT RESULTS REPORT")
    lines.append(f"Generated at: {datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
    lines.append("")

    lines.append("=" * 80)
    lines.append("INPUT SUMMARY")
    lines.append("=" * 80)
    lines.append(f"Total rows: {len(rows)}")
    lines.append(f"Expected stats files: 84")
    lines.append(f"Detected stats files: {len(set(r['source_file'] for r in rows))}")
    lines.append(f"Waves: {sorted(set(r['wave'] for r in rows))}")
    lines.append(f"Legislatures: {sorted(set(r['legislature'] for r in rows))}")
    lines.append(f"Segments: {sorted(set(r['segment'] for r in rows))}")
    lines.append(f"Pshifts: {sorted(set(r['pshift'] for r in rows))}")
    lines.append("")

    lines.append("=" * 80)
    lines.append("DOMINANT SHIFTS BY WAVE / LEGISLATURE / SEGMENT")
    lines.append("=" * 80)

    for row in dominant:
        lines.append(
            f"{row['wave']} | Legislature {row['legislature']} | Segment {row['segment']} "
            f"-> {row['dominant_pshift']} "
            f"(freq={row['frequency']}, prob={row['probability']:.4f})"
        )

    lines.append("")

    lines.append("=" * 80)
    lines.append("GLOBAL AVERAGE PROBABILITY BY WAVE")
    lines.append("=" * 80)

    for wave in sorted(set(r["wave"] for r in agg_wave)):
        lines.append("")
        lines.append(wave.upper())

        wave_rows = [r for r in agg_wave if r["wave"] == wave]
        wave_rows = sorted(wave_rows, key=lambda r: r["avg_probability"], reverse=True)

        for r in wave_rows:
            lines.append(
                f"  {r['pshift']}: "
                f"total_freq={r['total_frequency']}, "
                f"avg_prob={r['avg_probability']:.4f}, "
                f"avg_P(S|D)={r['avg_p_s_given_d']:.4f}, "
                f"avg_P(S|D,C)={r['avg_p_s_given_d_c']:.4f}"
            )

    lines.append("")

    lines.append("=" * 80)
    lines.append("BIGGEST WAVE DIFFERENCES BY PROBABILITY")
    lines.append("=" * 80)

    comparison_sorted = sorted(
        comparison,
        key=lambda r: abs(r["probability_diff"]),
        reverse=True
    )

    for r in comparison_sorted[:30]:
        lines.append(
            f"Leg {r['legislature']} | Segment {r['segment']} | {r['pshift']} "
            f"prob_w1={r['probability_wave_1']:.4f} "
            f"prob_w2={r['probability_wave_2']:.4f} "
            f"diff={r['probability_diff']:.4f}"
        )

    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    AGGREGATED_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    stats_files = find_stats_files()

    all_rows = []

    for path in stats_files:
        try:
            all_rows.extend(load_stats_file(path))
        except Exception as e:
            print(f"ERRO ao ler {path}: {e}")

    agg_leg = aggregate_by_wave_legislature(all_rows)
    agg_wave = aggregate_by_wave(all_rows)
    dominant = dominant_shift_by_legislature(all_rows)
    comparison = compare_waves_by_legislature(all_rows)

    all_csv = AGGREGATED_DIR / f"all_parshift_results_{timestamp}.csv"
    agg_leg_csv = AGGREGATED_DIR / f"parshift_by_wave_legislature_{timestamp}.csv"
    agg_wave_csv = AGGREGATED_DIR / f"parshift_by_wave_{timestamp}.csv"
    dominant_csv = AGGREGATED_DIR / f"dominant_shifts_{timestamp}.csv"
    comparison_csv = AGGREGATED_DIR / f"wave_comparison_{timestamp}.csv"

    write_csv(all_csv, all_rows)

    write_csv(agg_leg_csv, agg_leg)
    write_csv(agg_wave_csv, agg_wave)
    write_csv(dominant_csv, dominant)
    write_csv(comparison_csv, comparison)

    json_path = AGGREGATED_DIR / f"all_parshift_results_{timestamp}.json"
    report_path = REPORTS_DIR / f"aggregated_parshift_report_{timestamp}.txt"

    json_path.write_text(json.dumps(all_rows, indent=2, ensure_ascii=False), encoding="utf-8")

    write_report(
        report_path,
        all_rows,
        agg_wave,
        agg_leg,
        dominant,
        comparison
    )

    print(f"Stats files detected: {len(stats_files)}")
    print(f"Rows aggregated: {len(all_rows)}")
    print("")
    print(f"All results CSV: {all_csv}")
    print(f"By wave/legislature CSV: {agg_leg_csv}")
    print(f"By wave CSV: {agg_wave_csv}")
    print(f"Dominant shifts CSV: {dominant_csv}")
    print(f"Wave comparison CSV: {comparison_csv}")
    print(f"JSON: {json_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()