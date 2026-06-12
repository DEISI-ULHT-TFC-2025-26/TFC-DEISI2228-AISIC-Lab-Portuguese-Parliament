import csv
import json
import re
from pathlib import Path
from datetime import datetime
from collections import defaultdict

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

SHIFT_CLASS = {
    "A0-X0": "Turn Claiming",
    "A0-XA": "Turn Claiming",
    "A0-XY": "Turn Claiming",
    "AB-X0": "Turn Usurping",
    "AB-XA": "Turn Usurping",
    "AB-XB": "Turn Usurping",
    "AB-XY": "Turn Usurping",
    "AB-BA": "Turn Receiving",
    "AB-B0": "Turn Receiving",
    "AB-BY": "Turn Receiving",
    "A0-AY": "Turn Continuing",
    "AB-A0": "Turn Continuing",
    "AB-AY": "Turn Continuing",
}


def extract_legislature(path: Path):
    match = re.search(r"legislature_(\d+)", str(path))
    return int(match.group(1)) if match else None


def extract_sample(path: Path):
    match = re.search(r"sample_(\d+)", path.name)
    return int(match.group(1)) if match else None


def extract_segment(path: Path):
    match = re.search(r"segment_(\d+)_stats", path.name)
    return int(match.group(1)) if match else None


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


def safe_bool(value):
    if isinstance(value, bool):
        return value

    if value is None:
        return False

    if isinstance(value, str):
        return value.strip().lower() in ["true", "1", "yes", "sim"]

    return bool(value)


def load_stats_file(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"{path.name} não contém uma lista JSON.")

    rows = []

    legislature = extract_legislature(path)
    sample = extract_sample(path)
    segment = extract_segment(path)

    for item in data:
        if not isinstance(item, dict):
            continue

        shift = item.get("Pshift")

        if not shift:
            continue

        row = {
            "dataset": "final_dataset",
            "legislature": legislature,
            "sample": sample,
            "segment": segment,
            "pshift": shift,
            "shift_class": SHIFT_CLASS.get(shift, "Unknown"),
            "frequency": safe_int(item.get("Frequency")),
            "probability": safe_float(item.get("Probability")),
            "p_s_given_d": safe_float(item.get("P(S|D)")),
            "p_s_given_d_c": safe_float(item.get("P(S|D,C)")),
            "change_of_speaker": safe_bool(item.get("Change of Speaker (C)")),
            "directed_remark": safe_bool(item.get("Directed Remark (D)")),
            "source_file": path.name,
        }

        rows.append(row)

    return rows


def find_stats_files():
    final_files = sorted(
        RESULTS_DIR.glob("final_dataset/**/*segment_*_stats.json"),
        key=lambda p: (
            extract_legislature(p) or 999,
            extract_sample(p) or 999,
            extract_segment(p) or 999,
            p.name
        )
    )

    if final_files:
        return final_files

    legacy_files = sorted(
        RESULTS_DIR.glob("wave_*/*/*segment_*_stats.json"),
        key=lambda p: (
            extract_legislature(p) or 999,
            extract_segment(p) or 999,
            p.name
        )
    )

    return legacy_files


def write_csv(path: Path, rows):
    if not rows:
        return

    fieldnames = sorted({key for row in rows for key in row.keys()})

    preferred_order = [
        "dataset",
        "legislature",
        "sample",
        "segment",
        "pshift",
        "shift_class",
        "frequency",
        "probability",
        "global_probability",
        "total_frequency",
        "total_transitions",
        "avg_probability",
        "avg_p_s_given_d",
        "avg_p_s_given_d_c",
        "dominant_pshift",
        "source_file",
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


def aggregate_global(rows):
    grouped = defaultdict(list)

    for row in rows:
        grouped[row["pshift"]].append(row)

    total_transitions = sum(row["frequency"] for row in rows)
    output = []

    for pshift in EXPECTED_SHIFTS:
        values = grouped.get(pshift, [])
        total_frequency = sum(v["frequency"] for v in values)

        if values:
            avg_probability = sum(v["probability"] for v in values) / len(values)
            avg_p_s_given_d = sum(v["p_s_given_d"] for v in values) / len(values)
            avg_p_s_given_d_c = sum(v["p_s_given_d_c"] for v in values) / len(values)
        else:
            avg_probability = 0.0
            avg_p_s_given_d = 0.0
            avg_p_s_given_d_c = 0.0

        output.append({
            "dataset": "final_dataset",
            "pshift": pshift,
            "shift_class": SHIFT_CLASS.get(pshift, "Unknown"),
            "total_frequency": total_frequency,
            "total_transitions": total_transitions,
            "global_probability": total_frequency / total_transitions if total_transitions else 0.0,
            "avg_probability": avg_probability,
            "avg_p_s_given_d": avg_p_s_given_d,
            "avg_p_s_given_d_c": avg_p_s_given_d_c,
        })

    return sorted(output, key=lambda r: r["total_frequency"], reverse=True)


def aggregate_by_legislature(rows):
    grouped = defaultdict(list)

    for row in rows:
        grouped[(row["legislature"], row["pshift"])].append(row)

    totals_by_legislature = defaultdict(int)

    for row in rows:
        totals_by_legislature[row["legislature"]] += row["frequency"]

    output = []

    legislatures = sorted(set(row["legislature"] for row in rows if row["legislature"] is not None))

    for legislature in legislatures:
        for pshift in EXPECTED_SHIFTS:
            values = grouped.get((legislature, pshift), [])
            total_frequency = sum(v["frequency"] for v in values)
            total_transitions = totals_by_legislature[legislature]

            if values:
                avg_probability = sum(v["probability"] for v in values) / len(values)
                avg_p_s_given_d = sum(v["p_s_given_d"] for v in values) / len(values)
                avg_p_s_given_d_c = sum(v["p_s_given_d_c"] for v in values) / len(values)
            else:
                avg_probability = 0.0
                avg_p_s_given_d = 0.0
                avg_p_s_given_d_c = 0.0

            output.append({
                "dataset": "final_dataset",
                "legislature": legislature,
                "pshift": pshift,
                "shift_class": SHIFT_CLASS.get(pshift, "Unknown"),
                "total_frequency": total_frequency,
                "total_transitions": total_transitions,
                "global_probability": total_frequency / total_transitions if total_transitions else 0.0,
                "avg_probability": avg_probability,
                "avg_p_s_given_d": avg_p_s_given_d,
                "avg_p_s_given_d_c": avg_p_s_given_d_c,
            })

    return sorted(output, key=lambda r: (r["legislature"], -r["total_frequency"], r["pshift"]))


def aggregate_by_segment(rows):
    grouped = defaultdict(list)

    for row in rows:
        grouped[(row["segment"], row["pshift"])].append(row)

    totals_by_segment = defaultdict(int)

    for row in rows:
        totals_by_segment[row["segment"]] += row["frequency"]

    output = []

    segments = sorted(set(row["segment"] for row in rows if row["segment"] is not None))

    for segment in segments:
        for pshift in EXPECTED_SHIFTS:
            values = grouped.get((segment, pshift), [])
            total_frequency = sum(v["frequency"] for v in values)
            total_transitions = totals_by_segment[segment]

            output.append({
                "dataset": "final_dataset",
                "segment": segment,
                "pshift": pshift,
                "shift_class": SHIFT_CLASS.get(pshift, "Unknown"),
                "total_frequency": total_frequency,
                "total_transitions": total_transitions,
                "global_probability": total_frequency / total_transitions if total_transitions else 0.0,
            })

    return sorted(output, key=lambda r: (r["segment"], -r["total_frequency"], r["pshift"]))


def aggregate_by_legislature_segment(rows):
    grouped = defaultdict(list)

    for row in rows:
        grouped[(row["legislature"], row["segment"], row["pshift"])].append(row)

    totals = defaultdict(int)

    for row in rows:
        totals[(row["legislature"], row["segment"])] += row["frequency"]

    output = []

    legislatures = sorted(set(row["legislature"] for row in rows if row["legislature"] is not None))
    segments = sorted(set(row["segment"] for row in rows if row["segment"] is not None))

    for legislature in legislatures:
        for segment in segments:
            for pshift in EXPECTED_SHIFTS:
                values = grouped.get((legislature, segment, pshift), [])
                total_frequency = sum(v["frequency"] for v in values)
                total_transitions = totals[(legislature, segment)]

                output.append({
                    "dataset": "final_dataset",
                    "legislature": legislature,
                    "segment": segment,
                    "pshift": pshift,
                    "shift_class": SHIFT_CLASS.get(pshift, "Unknown"),
                    "total_frequency": total_frequency,
                    "total_transitions": total_transitions,
                    "global_probability": total_frequency / total_transitions if total_transitions else 0.0,
                })

    return sorted(output, key=lambda r: (r["legislature"], r["segment"], -r["total_frequency"], r["pshift"]))


def aggregate_by_class(rows):
    grouped = defaultdict(list)

    for row in rows:
        grouped[row["shift_class"]].append(row)

    total_transitions = sum(row["frequency"] for row in rows)
    output = []

    for shift_class, values in grouped.items():
        total_frequency = sum(v["frequency"] for v in values)

        output.append({
            "dataset": "final_dataset",
            "shift_class": shift_class,
            "total_frequency": total_frequency,
            "total_transitions": total_transitions,
            "global_probability": total_frequency / total_transitions if total_transitions else 0.0,
        })

    return sorted(output, key=lambda r: r["total_frequency"], reverse=True)


def aggregate_class_by_legislature(rows):
    grouped = defaultdict(list)

    for row in rows:
        grouped[(row["legislature"], row["shift_class"])].append(row)

    totals_by_legislature = defaultdict(int)

    for row in rows:
        totals_by_legislature[row["legislature"]] += row["frequency"]

    output = []

    for (legislature, shift_class), values in grouped.items():
        total_frequency = sum(v["frequency"] for v in values)
        total_transitions = totals_by_legislature[legislature]

        output.append({
            "dataset": "final_dataset",
            "legislature": legislature,
            "shift_class": shift_class,
            "total_frequency": total_frequency,
            "total_transitions": total_transitions,
            "global_probability": total_frequency / total_transitions if total_transitions else 0.0,
        })

    return sorted(output, key=lambda r: (r["legislature"], -r["total_frequency"], r["shift_class"]))


def dominant_shift_by_legislature_segment(rows):
    grouped = defaultdict(list)

    for row in rows:
        grouped[(row["legislature"], row["segment"])].append(row)

    output = []

    for (legislature, segment), values in grouped.items():
        dominant = max(values, key=lambda r: r["frequency"])

        output.append({
            "dataset": "final_dataset",
            "legislature": legislature,
            "segment": segment,
            "dominant_pshift": dominant["pshift"],
            "shift_class": dominant["shift_class"],
            "frequency": dominant["frequency"],
            "probability": dominant["probability"],
        })

    return sorted(output, key=lambda r: (r["legislature"], r["segment"]))


def write_report(path: Path, rows, global_rows, by_legislature, by_segment, by_class, dominant):
    lines = []

    lines.append("AGGREGATED PARSHIFT RESULTS REPORT")
    lines.append("Dataset: final_dataset")
    lines.append(f"Generated at: {datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
    lines.append("")

    lines.append("=" * 80)
    lines.append("INPUT SUMMARY")
    lines.append("=" * 80)
    lines.append(f"Total rows: {len(rows)}")
    lines.append(f"Detected stats files: {len(set(r['source_file'] for r in rows))}")
    lines.append(f"Legislatures: {sorted(set(r['legislature'] for r in rows))}")
    lines.append(f"Samples per legislature: {sorted(set(r['sample'] for r in rows))}")
    lines.append(f"Segments: {sorted(set(r['segment'] for r in rows))}")
    lines.append(f"Pshifts: {sorted(set(r['pshift'] for r in rows))}")
    lines.append(f"Total transitions: {sum(r['frequency'] for r in rows)}")
    lines.append("")

    lines.append("=" * 80)
    lines.append("GLOBAL PARTICIPATION SHIFTS")
    lines.append("=" * 80)

    for r in global_rows:
        lines.append(
            f"{r['pshift']}: "
            f"freq={r['total_frequency']}, "
            f"prob={r['global_probability']:.4f}, "
            f"class={r['shift_class']}"
        )

    lines.append("")

    lines.append("=" * 80)
    lines.append("PARTICIPATION SHIFT CLASSES")
    lines.append("=" * 80)

    for r in by_class:
        lines.append(
            f"{r['shift_class']}: "
            f"freq={r['total_frequency']}, "
            f"prob={r['global_probability']:.4f}"
        )

    lines.append("")

    lines.append("=" * 80)
    lines.append("DOMINANT SHIFTS BY LEGISLATURE AND SEGMENT")
    lines.append("=" * 80)

    for row in dominant:
        lines.append(
            f"Legislature {row['legislature']} | Segment {row['segment']} "
            f"-> {row['dominant_pshift']} "
            f"(freq={row['frequency']}, prob={row['probability']:.4f})"
        )

    lines.append("")

    lines.append("=" * 80)
    lines.append("TOP SHIFTS BY LEGISLATURE")
    lines.append("=" * 80)

    legislatures = sorted(set(r["legislature"] for r in by_legislature))

    for legislature in legislatures:
        lines.append("")
        lines.append(f"LEGISLATURE {legislature}")

        leg_rows = [r for r in by_legislature if r["legislature"] == legislature]
        leg_rows = sorted(leg_rows, key=lambda r: r["total_frequency"], reverse=True)

        for r in leg_rows[:6]:
            lines.append(
                f"  {r['pshift']}: "
                f"freq={r['total_frequency']}, "
                f"prob={r['global_probability']:.4f}, "
                f"class={r['shift_class']}"
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

    global_rows = aggregate_global(all_rows)
    by_legislature = aggregate_by_legislature(all_rows)
    by_segment = aggregate_by_segment(all_rows)
    by_legislature_segment = aggregate_by_legislature_segment(all_rows)
    by_class = aggregate_by_class(all_rows)
    class_by_legislature = aggregate_class_by_legislature(all_rows)
    dominant = dominant_shift_by_legislature_segment(all_rows)

    all_csv = AGGREGATED_DIR / f"all_parshift_results_final_{timestamp}.csv"
    global_csv = AGGREGATED_DIR / f"parshift_global_{timestamp}.csv"
    by_legislature_csv = AGGREGATED_DIR / f"parshift_by_legislature_{timestamp}.csv"
    by_segment_csv = AGGREGATED_DIR / f"parshift_by_segment_{timestamp}.csv"
    by_legislature_segment_csv = AGGREGATED_DIR / f"parshift_by_legislature_segment_{timestamp}.csv"
    by_class_csv = AGGREGATED_DIR / f"parshift_by_class_{timestamp}.csv"
    class_by_legislature_csv = AGGREGATED_DIR / f"parshift_class_by_legislature_{timestamp}.csv"
    dominant_csv = AGGREGATED_DIR / f"dominant_shifts_final_{timestamp}.csv"

    write_csv(all_csv, all_rows)
    write_csv(global_csv, global_rows)
    write_csv(by_legislature_csv, by_legislature)
    write_csv(by_segment_csv, by_segment)
    write_csv(by_legislature_segment_csv, by_legislature_segment)
    write_csv(by_class_csv, by_class)
    write_csv(class_by_legislature_csv, class_by_legislature)
    write_csv(dominant_csv, dominant)

    json_path = AGGREGATED_DIR / f"all_parshift_results_final_{timestamp}.json"
    report_path = REPORTS_DIR / f"aggregated_parshift_report_final_{timestamp}.txt"

    json_path.write_text(
        json.dumps(all_rows, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    write_report(
        report_path,
        all_rows,
        global_rows,
        by_legislature,
        by_segment,
        by_class,
        dominant
    )

    print(f"Stats files detected: {len(stats_files)}")
    print(f"Rows aggregated: {len(all_rows)}")
    print("")
    print(f"All results CSV: {all_csv}")
    print(f"Global CSV: {global_csv}")
    print(f"By legislature CSV: {by_legislature_csv}")
    print(f"By segment CSV: {by_segment_csv}")
    print(f"By legislature/segment CSV: {by_legislature_segment_csv}")
    print(f"By class CSV: {by_class_csv}")
    print(f"Class by legislature CSV: {class_by_legislature_csv}")
    print(f"Dominant shifts CSV: {dominant_csv}")
    print(f"JSON: {json_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()