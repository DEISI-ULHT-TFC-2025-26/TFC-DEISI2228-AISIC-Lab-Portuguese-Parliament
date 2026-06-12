import json
from pathlib import Path
from datetime import datetime
from collections import Counter

BASE_DIR = Path(__file__).resolve().parent / "data" / "processed" / "sample_by_legislature"

ANNOTATION1_DIR = BASE_DIR / "Annotation 1"
ANNOTATION2_DIR = BASE_DIR / "Annotation 2"
RESULTS_DIR = BASE_DIR

COMPARE_ONLY_FIRST_COMMON_FILE = False


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
        raise ValueError(f"Não foi possível interpretar o ficheiro JSON: {path}")

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


def load_annotation_file(path: Path):
    data = load_json_robust(path)

    if not isinstance(data, list):
        raise ValueError(f"O ficheiro {path.name} não contém uma lista JSON de anotações.")

    items = {}

    for row in data:
        if not isinstance(row, dict):
            continue

        if "utterance_id" not in row:
            continue

        utt_id = row["utterance_id"]

        speaker = str(row.get("speaker", "UNKNOWN")).strip() or "UNKNOWN"
        target = str(row.get("target", "UNKNOWN")).strip() or "UNKNOWN"

        items[utt_id] = {
            "speaker": speaker,
            "target": target,
            "joint": f"{speaker}|||{target}"
        }

    return items


def percent_agreement(labels_a, labels_b):
    if not labels_a or len(labels_a) != len(labels_b):
        return 0.0

    same = sum(1 for a, b in zip(labels_a, labels_b) if a == b)
    return same / len(labels_a)


def cohens_kappa(labels_a, labels_b):
    if not labels_a or len(labels_a) != len(labels_b):
        return 0.0

    n = len(labels_a)
    po = percent_agreement(labels_a, labels_b)

    categories = sorted(set(labels_a) | set(labels_b))
    count_a = Counter(labels_a)
    count_b = Counter(labels_b)

    pe = sum((count_a[c] / n) * (count_b[c] / n) for c in categories)

    if pe == 1:
        return 1.0

    return (po - pe) / (1 - pe)


def krippendorff_alpha_nominal(items_labels):
    if not items_labels:
        return 0.0

    filtered_items = []

    for row in items_labels:
        values = [v for v in row if v is not None]
        if len(values) >= 2:
            filtered_items.append(values)

    if not filtered_items:
        return 0.0

    observed_num = 0.0
    observed_den = 0.0

    for row in filtered_items:
        n = len(row)
        observed_den += n * (n - 1)

        counts = Counter(row)
        same_pairs = sum(c * (c - 1) for c in counts.values())
        disagree_pairs = n * (n - 1) - same_pairs

        observed_num += disagree_pairs

    if observed_den == 0:
        return 1.0

    observed_disagreement = observed_num / observed_den

    all_labels = []
    for row in filtered_items:
        all_labels.extend(row)

    total = len(all_labels)

    if total < 2:
        return 1.0

    counts_all = Counter(all_labels)
    same_pairs_all = sum(c * (c - 1) for c in counts_all.values())
    total_pairs_all = total * (total - 1)

    expected_disagreement = (total_pairs_all - same_pairs_all) / total_pairs_all

    if expected_disagreement == 0:
        return 1.0

    return 1 - (observed_disagreement / expected_disagreement)


def process_field(file_name, field, ann1_items, ann2_items):
    ann1_ids = set(ann1_items.keys())
    ann2_ids = set(ann2_items.keys())

    common_ids = sorted(ann1_ids & ann2_ids)

    ann1_only = sorted(ann1_ids - ann2_ids)
    ann2_only = sorted(ann2_ids - ann1_ids)

    ann1_labels = [ann1_items[i][field] for i in common_ids]
    ann2_labels = [ann2_items[i][field] for i in common_ids]

    items_2 = [
        [ann1_items[i][field], ann2_items[i][field]]
        for i in common_ids
    ]

    return {
        "file": file_name,
        "field": field,
        "n_items_common": len(common_ids),
        "n_items_annotation1_only": len(ann1_only),
        "n_items_annotation2_only": len(ann2_only),
        "percent_agreement": percent_agreement(ann1_labels, ann2_labels),
        "cohen_kappa": cohens_kappa(ann1_labels, ann2_labels),
        "krippendorff_alpha": krippendorff_alpha_nominal(items_2)
    }


def average_metric(results, key):
    values = [r[key] for r in results if key in r]
    return sum(values) / len(values) if values else 0.0


def get_common_file_names():
    annotation1_names = {f.name for f in ANNOTATION1_DIR.glob("*.json")}
    annotation2_names = {f.name for f in ANNOTATION2_DIR.glob("*.json")}

    common_names = sorted(annotation1_names & annotation2_names)

    if COMPARE_ONLY_FIRST_COMMON_FILE and common_names:
        return [common_names[0]]

    return common_names


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    common_names = get_common_file_names()

    if not common_names:
        print("Não foram encontrados ficheiros comuns nas duas pastas.")
        return

    all_results = {
        "speaker": [],
        "target": [],
        "joint": []
    }

    file_errors = []

    for file_name in common_names:
        try:
            ann1_items = load_annotation_file(ANNOTATION1_DIR / file_name)
            ann2_items = load_annotation_file(ANNOTATION2_DIR / file_name)

            all_results["speaker"].append(
                process_field(file_name, "speaker", ann1_items, ann2_items)
            )

            all_results["target"].append(
                process_field(file_name, "target", ann1_items, ann2_items)
            )

            all_results["joint"].append(
                process_field(file_name, "joint", ann1_items, ann2_items)
            )

            print(f"OK: {file_name}")

        except Exception as e:
            file_errors.append((file_name, str(e)))
            print(f"ERRO em {file_name}: {e}")

    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")

    report_path = RESULTS_DIR / f"annotation_agreement_2_annotators_{timestamp}.txt"

    lines = []

    lines.append("ANNOTATION AGREEMENT REPORT")
    lines.append("Two-annotator agreement over sample by legislature")
    lines.append(f"Date: {now.strftime('%Y-%m-%d')}")
    lines.append(f"Time: {now.strftime('%H:%M:%S')}")
    lines.append(f"Timestamp: {timestamp}")
    lines.append("")
    lines.append(f"Annotation 1 folder: {ANNOTATION1_DIR}")
    lines.append(f"Annotation 2 folder: {ANNOTATION2_DIR}")
    lines.append("")
    lines.append(f"Files requested for analysis: {len(common_names)}")

    for name in common_names:
        lines.append(f" - {name}")

    lines.append("")

    if file_errors:
        lines.append("FILES WITH ERRORS")
        for file_name, error in file_errors:
            lines.append(f" - {file_name}: {error}")
        lines.append("")

    successful_files = len(all_results["speaker"])
    lines.append(f"Files successfully analyzed: {successful_files}")
    lines.append("")

    for field in ["speaker", "target", "joint"]:
        if not all_results[field]:
            continue

        lines.append("=" * 80)
        lines.append(f"FIELD: {field.upper()}")
        lines.append("=" * 80)
        lines.append("")

        for result in all_results[field]:
            lines.append(f"File: {result['file']}")
            lines.append(f"  Common items: {result['n_items_common']}")
            lines.append(f"  Items only in Annotation 1: {result['n_items_annotation1_only']}")
            lines.append(f"  Items only in Annotation 2: {result['n_items_annotation2_only']}")
            lines.append(f"  Percent agreement: {result['percent_agreement']:.4f}")
            lines.append(f"  Cohen's Kappa: {result['cohen_kappa']:.4f}")
            lines.append(f"  Krippendorff's Alpha: {result['krippendorff_alpha']:.4f}")
            lines.append("")

        lines.append("SUMMARY")
        lines.append(f"  Mean percent agreement: {average_metric(all_results[field], 'percent_agreement'):.4f}")
        lines.append(f"  Mean Cohen's Kappa: {average_metric(all_results[field], 'cohen_kappa'):.4f}")
        lines.append(f"  Mean Krippendorff's Alpha: {average_metric(all_results[field], 'krippendorff_alpha'):.4f}")
        lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Relatório criado com sucesso: {report_path}")


if __name__ == "__main__":
    main()