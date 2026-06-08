import csv
import json
import re
from pathlib import Path
from datetime import datetime
from collections import Counter, defaultdict

BASE_DIR = Path(__file__).resolve().parent

PARSHIFT_DIR = BASE_DIR / "data" / "processed" / "parshift"
INPUT_CSV_DIR = PARSHIFT_DIR / "input_csv"
WAVE_1_DIR = INPUT_CSV_DIR / "wave_1"
WAVE_2_DIR = INPUT_CSV_DIR / "wave_2"
MAPPINGS_DIR = PARSHIFT_DIR / "mappings"
REPORTS_DIR = PARSHIFT_DIR / "reports"

ASSEMBLY_LABEL = "Assembleia"
PRESIDENT_LABEL = "Presidente"


def load_latest_mapping():
    latest_path = MAPPINGS_DIR / "participant_mapping_latest.json"

    if not latest_path.exists():
        return {}, {}

    with open(latest_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    id_to_label = data.get("id_to_label", {})
    label_to_id = data.get("label_to_id", {})

    return id_to_label, label_to_id


def extract_legislature_from_filename(file_name):
    match = re.search(r"legislature_(\d+)", file_name)

    if not match:
        return "UNKNOWN"

    return int(match.group(1))


def load_csv_rows(path):
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")

        for row in reader:
            rows.append(row)

    return rows


def word_count(text):
    if not text:
        return 0
    return len(str(text).split())


def safe_div(a, b):
    return a / b if b else 0


def calculate_density(nodes_count, edges_count):
    if nodes_count <= 1:
        return 0

    possible_directed_edges = nodes_count * (nodes_count - 1)

    return edges_count / possible_directed_edges if possible_directed_edges else 0


def characterize_file(path, wave_name, id_to_label):
    rows = load_csv_rows(path)

    speaker_counter = Counter()
    target_counter = Counter()
    dyad_counter = Counter()

    words = 0

    for row in rows:
        speaker_id = row.get("speaker_id", "").strip()
        target_id = row.get("target_id", "").strip()
        text = row.get("utterance_text", "")

        speaker_label = id_to_label.get(speaker_id, speaker_id)
        target_label = id_to_label.get(target_id, target_id)

        speaker_counter[speaker_label] += 1
        target_counter[target_label] += 1
        dyad_counter[(speaker_label, target_label)] += 1

        words += word_count(text)

    speakers = set(speaker_counter.keys())
    targets = set(target_counter.keys())
    participants = speakers | targets

    unique_dyads = len(dyad_counter)

    assembly_as_target = target_counter.get(ASSEMBLY_LABEL, 0)
    president_as_speaker = speaker_counter.get(PRESIDENT_LABEL, 0)

    nodes_count = len(participants)
    edges_count = unique_dyads

    density = calculate_density(nodes_count, edges_count)

    top_speakers = dict(speaker_counter.most_common(10))
    top_targets = dict(target_counter.most_common(10))

    top_dyads = {
        f"{speaker} -> {target}": count
        for (speaker, target), count in dyad_counter.most_common(10)
    }

    legislature = extract_legislature_from_filename(path.name)

    return {
        "wave": wave_name,
        "legislature": legislature,
        "file": path.name,
        "utterances": len(rows),
        "words": words,
        "unique_speakers": len(speakers),
        "unique_targets": len(targets),
        "unique_participants": len(participants),
        "unique_dyads": unique_dyads,
        "avg_interactions_per_dyad": safe_div(len(rows), unique_dyads),
        "assembly_as_target_count": assembly_as_target,
        "assembly_as_target_rate": safe_div(assembly_as_target, len(rows)),
        "president_as_speaker_count": president_as_speaker,
        "president_as_speaker_rate": safe_div(president_as_speaker, len(rows)),
        "network_nodes": nodes_count,
        "network_edges": edges_count,
        "network_density": density,
        "top_speakers": top_speakers,
        "top_targets": top_targets,
        "top_dyads": top_dyads
    }


def characterize_wave(wave_name, wave_dir, id_to_label):
    results = []

    files = sorted(
        wave_dir.glob("*.csv"),
        key=lambda p: extract_legislature_from_filename(p.name)
    )

    for path in files:
        result = characterize_file(path, wave_name, id_to_label)
        results.append(result)

    return results


def mean(values):
    return sum(values) / len(values) if values else 0


def aggregate_wave(results):
    return {
        "files": len(results),
        "total_utterances": sum(r["utterances"] for r in results),
        "total_words": sum(r["words"] for r in results),
        "avg_utterances": mean([r["utterances"] for r in results]),
        "avg_words": mean([r["words"] for r in results]),
        "avg_unique_speakers": mean([r["unique_speakers"] for r in results]),
        "avg_unique_targets": mean([r["unique_targets"] for r in results]),
        "avg_unique_participants": mean([r["unique_participants"] for r in results]),
        "avg_unique_dyads": mean([r["unique_dyads"] for r in results]),
        "avg_interactions_per_dyad": mean([r["avg_interactions_per_dyad"] for r in results]),
        "avg_assembly_as_target_rate": mean([r["assembly_as_target_rate"] for r in results]),
        "avg_president_as_speaker_rate": mean([r["president_as_speaker_rate"] for r in results]),
        "avg_network_density": mean([r["network_density"] for r in results])
    }


def compare_waves(wave1_results, wave2_results):
    by_leg_wave1 = {r["legislature"]: r for r in wave1_results}
    by_leg_wave2 = {r["legislature"]: r for r in wave2_results}

    common_legs = sorted(set(by_leg_wave1.keys()) & set(by_leg_wave2.keys()))

    comparisons = []

    numeric_fields = [
        "utterances",
        "words",
        "unique_speakers",
        "unique_targets",
        "unique_participants",
        "unique_dyads",
        "avg_interactions_per_dyad",
        "assembly_as_target_rate",
        "president_as_speaker_rate",
        "network_density"
    ]

    for leg in common_legs:
        w1 = by_leg_wave1[leg]
        w2 = by_leg_wave2[leg]

        diff = {
            "legislature": leg,
            "wave_1_file": w1["file"],
            "wave_2_file": w2["file"]
        }

        for field in numeric_fields:
            diff[f"{field}_wave_1"] = w1[field]
            diff[f"{field}_wave_2"] = w2[field]
            diff[f"{field}_diff"] = w2[field] - w1[field]

        comparisons.append(diff)

    return comparisons


def write_report(report_path, output):
    lines = []

    lines.append("PARSHIFT NETWORK CHARACTERIZATION REPORT")
    lines.append(f"Generated at: {output['generated_at']}")
    lines.append("")

    lines.append("=" * 80)
    lines.append("GLOBAL SUMMARY")
    lines.append("=" * 80)
    lines.append("")

    for wave_name in ["wave_1", "wave_2"]:
        summary = output["summary_by_wave"].get(wave_name, {})

        lines.append(f"{wave_name.upper()}")
        lines.append(f"  Files: {summary.get('files', 0)}")
        lines.append(f"  Total utterances: {summary.get('total_utterances', 0)}")
        lines.append(f"  Total words: {summary.get('total_words', 0)}")
        lines.append(f"  Avg utterances: {summary.get('avg_utterances', 0):.4f}")
        lines.append(f"  Avg unique speakers: {summary.get('avg_unique_speakers', 0):.4f}")
        lines.append(f"  Avg unique targets: {summary.get('avg_unique_targets', 0):.4f}")
        lines.append(f"  Avg unique participants: {summary.get('avg_unique_participants', 0):.4f}")
        lines.append(f"  Avg unique dyads: {summary.get('avg_unique_dyads', 0):.4f}")
        lines.append(f"  Avg interactions per dyad: {summary.get('avg_interactions_per_dyad', 0):.4f}")
        lines.append(f"  Avg Assembly as target rate: {summary.get('avg_assembly_as_target_rate', 0):.4f}")
        lines.append(f"  Avg President as speaker rate: {summary.get('avg_president_as_speaker_rate', 0):.4f}")
        lines.append(f"  Avg network density: {summary.get('avg_network_density', 0):.6f}")
        lines.append("")

    lines.append("=" * 80)
    lines.append("PER FILE RESULTS")
    lines.append("=" * 80)
    lines.append("")

    for wave_name in ["wave_1", "wave_2"]:
        lines.append(f"{wave_name.upper()}")
        lines.append("")

        for result in output["results_by_wave"].get(wave_name, []):
            lines.append(f"Legislature {result['legislature']} | {result['file']}")
            lines.append(f"  Utterances: {result['utterances']}")
            lines.append(f"  Words: {result['words']}")
            lines.append(f"  Unique speakers: {result['unique_speakers']}")
            lines.append(f"  Unique targets: {result['unique_targets']}")
            lines.append(f"  Unique participants: {result['unique_participants']}")
            lines.append(f"  Unique dyads: {result['unique_dyads']}")
            lines.append(f"  Avg interactions per dyad: {result['avg_interactions_per_dyad']:.4f}")
            lines.append(f"  Assembly as target: {result['assembly_as_target_count']} ({result['assembly_as_target_rate']:.4f})")
            lines.append(f"  President as speaker: {result['president_as_speaker_count']} ({result['president_as_speaker_rate']:.4f})")
            lines.append(f"  Network density: {result['network_density']:.6f}")
            lines.append("  Top dyads:")

            for dyad, count in result["top_dyads"].items():
                lines.append(f"    - {dyad}: {count}")

            lines.append("")

    lines.append("=" * 80)
    lines.append("WAVE 2 - WAVE 1 COMPARISON")
    lines.append("=" * 80)
    lines.append("")

    for comparison in output["wave_comparison"]:
        lines.append(f"Legislature {comparison['legislature']}")
        lines.append(f"  Wave 1 file: {comparison['wave_1_file']}")
        lines.append(f"  Wave 2 file: {comparison['wave_2_file']}")
        lines.append(f"  Utterances diff: {comparison['utterances_diff']}")
        lines.append(f"  Unique speakers diff: {comparison['unique_speakers_diff']}")
        lines.append(f"  Unique targets diff: {comparison['unique_targets_diff']}")
        lines.append(f"  Unique dyads diff: {comparison['unique_dyads_diff']}")
        lines.append(f"  Assembly target rate diff: {comparison['assembly_as_target_rate_diff']:.4f}")
        lines.append(f"  President speaker rate diff: {comparison['president_as_speaker_rate_diff']:.4f}")
        lines.append(f"  Network density diff: {comparison['network_density_diff']:.6f}")
        lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    id_to_label, _ = load_latest_mapping()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    wave1_results = characterize_wave("wave_1", WAVE_1_DIR, id_to_label)
    wave2_results = characterize_wave("wave_2", WAVE_2_DIR, id_to_label)

    output = {
        "generated_at": timestamp,
        "results_by_wave": {
            "wave_1": wave1_results,
            "wave_2": wave2_results
        },
        "summary_by_wave": {
            "wave_1": aggregate_wave(wave1_results),
            "wave_2": aggregate_wave(wave2_results)
        },
        "wave_comparison": compare_waves(wave1_results, wave2_results)
    }

    json_path = REPORTS_DIR / f"network_metrics_{timestamp}.json"
    txt_path = REPORTS_DIR / f"network_report_{timestamp}.txt"

    json_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")

    write_report(txt_path, output)

    print(f"JSON metrics criado: {json_path}")
    print(f"Report criado: {txt_path}")


if __name__ == "__main__":
    main()