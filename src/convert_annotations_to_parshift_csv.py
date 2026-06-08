import csv
import json
import re
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent

WAVE_1_DIR = BASE_DIR / "data" / "processed" / "sample_by_legislature" / "Gemini"
WAVE_2_DIR = BASE_DIR / "data" / "processed" / "sample_by_legislature" / "gemini 2"

PARSHIFT_DIR = BASE_DIR / "data" / "processed" / "parshift"
CSV_WAVE_1_DIR = PARSHIFT_DIR / "input_csv" / "wave_1"
CSV_WAVE_2_DIR = PARSHIFT_DIR / "input_csv" / "wave_2"
MAPPINGS_DIR = PARSHIFT_DIR / "mappings"
REPORTS_DIR = PARSHIFT_DIR / "reports"

DELIMITER = "\t"
REQUIRED_FIELDS = ["utterance_id", "speaker", "target", "text"]
DEFAULT_TARGET = "Assembleia"


def load_json_robust(path: Path):
    text = path.read_text(encoding="utf-8").strip()

    if not text:
        return []

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    idx = 0
    chunks = []

    while idx < len(text):
        while idx < len(text) and text[idx].isspace():
            idx += 1

        if idx >= len(text):
            break

        obj, end = decoder.raw_decode(text, idx)
        chunks.append(obj)
        idx = end

    if not chunks:
        raise ValueError(f"Não foi possível interpretar o JSON: {path}")

    if len(chunks) == 1:
        return chunks[0]

    merged = []

    for chunk in chunks:
        if isinstance(chunk, list):
            merged.extend(chunk)
        elif isinstance(chunk, dict):
            merged.append(chunk)
        else:
            raise ValueError(f"Estrutura JSON não suportada em {path}: {type(chunk)}")

    return merged


def extract_legislature_number(file_name: str):
    match = re.search(r"\|\|\s*(\d+)\s*-", file_name)

    if match:
        return int(match.group(1))

    match = re.match(r"^\s*(\d+)\s*-", file_name)

    if match:
        return int(match.group(1))

    raise ValueError(f"Não foi possível extrair a legislatura do ficheiro: {file_name}")


def normalize_label(value):
    value = str(value).strip()
    return value if value else "UNKNOWN"


def normalize_target(value):
    target = normalize_label(value)

    if target in ["", "UNKNOWN", "None", "none", "NONE", "null", "NULL", "Null"]:
        return DEFAULT_TARGET

    return target


def normalize_text(value):
    value = str(value).replace("\t", " ").replace("\n", " ").replace("\r", " ")
    return " ".join(value.split())


def get_or_create_participant_id(label, label_to_id, id_to_label):
    label = normalize_label(label)

    if label in label_to_id:
        return label_to_id[label]

    new_id = f"P{len(label_to_id) + 1:05d}"

    label_to_id[label] = new_id
    id_to_label[new_id] = label

    return new_id


def validate_rows(rows):
    errors = []
    warnings = []

    if not isinstance(rows, list):
        errors.append("O ficheiro não contém um array JSON.")
        return errors, warnings

    seen_ids = set()

    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            errors.append(f"Elemento na posição {index} não é objeto JSON.")
            continue

        for field in REQUIRED_FIELDS:
            if field not in row:
                errors.append(f"Campo ausente '{field}' na posição {index}.")

        utt_id = row.get("utterance_id")

        if utt_id in seen_ids:
            errors.append(f"utterance_id duplicado: {utt_id}")

        seen_ids.add(utt_id)

        speaker = str(row.get("speaker", "")).strip()
        target = str(row.get("target", "")).strip()
        text = str(row.get("text", "")).strip()

        if not speaker:
            errors.append(f"speaker vazio em utterance_id={utt_id}")

        if not target:
            warnings.append(f"target vazio em utterance_id={utt_id}; será convertido para {DEFAULT_TARGET}")

        if not text:
            warnings.append(f"text vazio em utterance_id={utt_id}")

    ids = [
        row.get("utterance_id")
        for row in rows
        if isinstance(row, dict) and "utterance_id" in row
    ]

    if ids:
        expected = list(range(1, len(ids) + 1))
        if ids != expected:
            warnings.append("utterance_id não é sequência perfeita 1..N.")

    return errors, warnings


def convert_file(input_path, output_path, label_to_id, id_to_label):
    rows = load_json_robust(input_path)

    errors, warnings = validate_rows(rows)

    if errors:
        return {
            "file": input_path.name,
            "converted": False,
            "errors": errors,
            "warnings": warnings,
            "utterances": 0
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "utterance_id",
                "speaker_id",
                "target_id",
                "utterance_text"
            ],
            delimiter=DELIMITER
        )

        writer.writeheader()

        for row in rows:
            speaker = normalize_label(row.get("speaker", "UNKNOWN"))
            target = normalize_target(row.get("target", ""))
            text = normalize_text(row.get("text", ""))

            speaker_id = get_or_create_participant_id(speaker, label_to_id, id_to_label)
            target_id = get_or_create_participant_id(target, label_to_id, id_to_label)

            writer.writerow({
                "utterance_id": row.get("utterance_id"),
                "speaker_id": speaker_id,
                "target_id": target_id,
                "utterance_text": text
            })

    return {
        "file": input_path.name,
        "converted": True,
        "errors": [],
        "warnings": warnings,
        "utterances": len(rows)
    }


def convert_wave(wave_name, input_dir, output_dir, label_to_id, id_to_label):
    results = []

    files = sorted(
        input_dir.glob("*.json"),
        key=lambda p: extract_legislature_number(p.name)
    )

    for file_path in files:
        legislature = extract_legislature_number(file_path.name)
        output_path = output_dir / f"legislature_{legislature}.csv"

        result = convert_file(file_path, output_path, label_to_id, id_to_label)

        result["wave"] = wave_name
        result["legislature"] = legislature
        result["output_file"] = output_path.name if result["converted"] else ""

        results.append(result)

    return results


def main():
    CSV_WAVE_1_DIR.mkdir(parents=True, exist_ok=True)
    CSV_WAVE_2_DIR.mkdir(parents=True, exist_ok=True)
    MAPPINGS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    label_to_id = {}
    id_to_label = {}

    all_results = []

    all_results.extend(
        convert_wave(
            "wave_1",
            WAVE_1_DIR,
            CSV_WAVE_1_DIR,
            label_to_id,
            id_to_label
        )
    )

    all_results.extend(
        convert_wave(
            "wave_2",
            WAVE_2_DIR,
            CSV_WAVE_2_DIR,
            label_to_id,
            id_to_label
        )
    )

    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")

    mapping_output = {
        "generated_at": timestamp,
        "delimiter": "TAB",
        "default_target": DEFAULT_TARGET,
        "label_to_id": label_to_id,
        "id_to_label": id_to_label
    }

    mapping_path = MAPPINGS_DIR / f"participant_mapping_{timestamp}.json"
    latest_mapping_path = MAPPINGS_DIR / "participant_mapping_latest.json"

    mapping_path.write_text(
        json.dumps(mapping_output, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    latest_mapping_path.write_text(
        json.dumps(mapping_output, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    report_path = REPORTS_DIR / f"conversion_report_{timestamp}.txt"

    lines = []
    lines.append("PARSHIFT CONVERSION REPORT")
    lines.append(f"Timestamp: {timestamp}")
    lines.append("Delimiter: TAB")
    lines.append(f"Default target: {DEFAULT_TARGET}")
    lines.append("")
    lines.append(f"Total files processed: {len(all_results)}")
    lines.append(f"Total converted: {sum(1 for r in all_results if r['converted'])}")
    lines.append(f"Total participants mapped: {len(label_to_id)}")
    lines.append("")

    for result in all_results:
        lines.append("=" * 80)
        lines.append(f"{result['wave']} | Legislature {result['legislature']} | {result['file']}")
        lines.append(f"Converted: {result['converted']}")
        lines.append(f"Utterances: {result['utterances']}")
        lines.append(f"Output: {result['output_file']}")

        if result["errors"]:
            lines.append("Errors:")
            for e in result["errors"]:
                lines.append(f" - {e}")

        if result["warnings"]:
            lines.append("Warnings:")
            for w in result["warnings"]:
                lines.append(f" - {w}")

        lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"CSV Wave 1: {CSV_WAVE_1_DIR}")
    print(f"CSV Wave 2: {CSV_WAVE_2_DIR}")
    print(f"Mapping: {latest_mapping_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()