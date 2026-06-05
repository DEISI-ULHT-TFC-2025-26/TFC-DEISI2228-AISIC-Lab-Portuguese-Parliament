import json
import shutil
import statistics
from pathlib import Path
from datetime import datetime
from collections import defaultdict

JSON_DIR = Path("data/processed/json")
OUTPUT_DIR = Path("data/processed/sample_by_legislature")
REPORT_DIR = Path("data/processed/reports")


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_utterance_count(data):
    return len(data.get("utterances", []))


def get_legislature(data):
    meta = data.get("debate_meta", {})
    return str(meta.get("legislature", "UNKNOWN_LEGISLATURE"))


def get_date(data):
    meta = data.get("debate_meta", {})
    return str(meta.get("date", ""))


def sanitize_legislature(leg):
    return leg if leg.isdigit() else "UNK"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    json_files = sorted(JSON_DIR.glob("*.json"))

    debates_by_legislature = defaultdict(list)
    errors = []

    for file_path in json_files:
        try:
            data = load_json(file_path)
            legislature = get_legislature(data)
            utterance_count = get_utterance_count(data)
            date = get_date(data)

            debates_by_legislature[legislature].append({
                "file": file_path,
                "file_name": file_path.name,
                "legislature": legislature,
                "date": date,
                "utterances": utterance_count
            })

        except Exception as e:
            errors.append((file_path.name, str(e)))

    selected = []

    for legislature, debates in debates_by_legislature.items():
        counts = [d["utterances"] for d in debates]
        median_value = statistics.median(counts)

        ranked_debates = sorted(
            debates,
            key=lambda d: (
                abs(d["utterances"] - median_value),
                d["date"],
                d["file_name"]
            )
        )

        chosen = ranked_debates[1] if len(ranked_debates) > 1 else ranked_debates[0]

        chosen["median_utterances"] = median_value
        chosen["distance_to_median"] = abs(chosen["utterances"] - median_value)
        chosen["selection_rank_by_median_proximity"] = (
            2 if len(ranked_debates) > 1 else 1
        )

        selected.append(chosen)

    selected = sorted(
        selected,
        key=lambda d: int(d["legislature"]) if d["legislature"].isdigit() else 9999
    )

    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")

    report_path = REPORT_DIR / f"annotation_sample_selection_second_closest_{timestamp}.txt"
    manifest_path = OUTPUT_DIR / f"annotation_sample_manifest_second_closest_{timestamp}.json"

    manifest = []
    lines = []

    lines.append("ANNOTATION SAMPLE SELECTION REPORT")
    lines.append(f"Date: {now.strftime('%Y-%m-%d')}")
    lines.append(f"Time: {now.strftime('%H:%M:%S')}")
    lines.append(f"Timestamp: {timestamp}")
    lines.append("")
    lines.append("Selection rule:")
    lines.append("One debate per legislature using the second closest debate to the median utterance count.")
    lines.append("If a legislature has only one debate, that debate is selected.")
    lines.append("")

    for item in selected:
        leg = sanitize_legislature(item["legislature"])

        new_name = f"2nd || {leg} - {item['file_name']}"
        destination = OUTPUT_DIR / new_name

        shutil.copy2(item["file"], destination)

        manifest_item = {
            "legislature": item["legislature"],
            "original_file": item["file_name"],
            "new_file": new_name,
            "date": item["date"],
            "utterances": item["utterances"],
            "median_utterances": item["median_utterances"],
            "distance_to_median": item["distance_to_median"],
            "selection_rank_by_median_proximity": item["selection_rank_by_median_proximity"]
        }

        manifest.append(manifest_item)

        lines.append(f"Legislature: {item['legislature']}")
        lines.append(f"  Original file: {item['file_name']}")
        lines.append(f"  Copied as: {new_name}")
        lines.append(f"  Utterances: {item['utterances']}")
        lines.append(f"  Median: {item['median_utterances']}")
        lines.append(f"  Distance: {item['distance_to_median']}")
        lines.append(
            f"  Selection rank by median proximity: "
            f"{item['selection_rank_by_median_proximity']}"
        )
        lines.append("")

    if errors:
        lines.append("FILES WITH ERRORS")
        for file_name, err in errors:
            lines.append(f"  - {file_name}: {err}")
        lines.append("")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"Relatório: {report_path}")
    print(f"Manifest: {manifest_path}")
    print(f"Ficheiros copiados para: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()