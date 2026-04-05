import json
from pathlib import Path
from datetime import datetime
from collections import Counter

BASE_DIR = Path(__file__).resolve().parent / "data" / "processed" / "annotation_agreement"
HUMAN1_DIR = BASE_DIR / "human1"
HUMAN2_DIR = BASE_DIR / "human2"
LLM_DIR = BASE_DIR / "LLM"
RESULTS_DIR = BASE_DIR / "results"

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

    cats = sorted(set(labels_a) | set(labels_b))
    count_a = Counter(labels_a)
    count_b = Counter(labels_b)

    pe = sum((count_a[c] / n) * (count_b[c] / n) for c in cats)

    if pe == 1:
        return 1.0

    return (po - pe) / (1 - pe)


def fleiss_kappa(items_labels):
    if not items_labels:
        return 0.0

    n = len(items_labels)
    raters = len(items_labels[0])

    if raters < 2:
        return 0.0

    categories = sorted({label for row in items_labels for label in row})
    cat_index = {c: i for i, c in enumerate(categories)}

    matrix = []
    for row in items_labels:
        counts = [0] * len(categories)
        for label in row:
            counts[cat_index[label]] += 1
        matrix.append(counts)

    p_j = []
    for j in range(len(categories)):
        total_j = sum(row[j] for row in matrix)
        p_j.append(total_j / (n * raters))

    p_i = []
    for row in matrix:
        row_sum_sq = sum(v * v for v in row)
        p = (row_sum_sq - raters) / (raters * (raters - 1))
        p_i.append(p)

    p_bar = sum(p_i) / n
    p_e = sum(p ** 2 for p in p_j)

    if p_e == 1:
        return 1.0

    return (p_bar - p_e) / (1 - p_e)


def three_way_exact_agreement(items_labels):
    if not items_labels:
        return 0.0
    same = sum(1 for row in items_labels if row[0] == row[1] == row[2])
    return same / len(items_labels)


def process_field(file_name, field, h1_items, h2_items, llm_items):
    common_ids = sorted(set(h1_items.keys()) & set(h2_items.keys()) & set(llm_items.keys()))

    h1_labels = [h1_items[i][field] for i in common_ids]
    h2_labels = [h2_items[i][field] for i in common_ids]
    llm_labels = [llm_items[i][field] for i in common_ids]

    items_3 = [[h1_items[i][field], h2_items[i][field], llm_items[i][field]] for i in common_ids]

    return {
        "file": file_name,
        "field": field,
        "n_items": len(common_ids),
        "three_way_exact_agreement": three_way_exact_agreement(items_3),
        "fleiss_kappa": fleiss_kappa(items_3),
        "pairwise": {
            "human1_vs_human2": {
                "percent_agreement": percent_agreement(h1_labels, h2_labels),
                "cohen_kappa": cohens_kappa(h1_labels, h2_labels)
            },
            "human1_vs_llm": {
                "percent_agreement": percent_agreement(h1_labels, llm_labels),
                "cohen_kappa": cohens_kappa(h1_labels, llm_labels)
            },
            "human2_vs_llm": {
                "percent_agreement": percent_agreement(h2_labels, llm_labels),
                "cohen_kappa": cohens_kappa(h2_labels, llm_labels)
            }
        }
    }


def average_metric(results, path):
    vals = []
    for r in results:
        value = r
        for p in path:
            value = value[p]
        vals.append(value)
    return sum(vals) / len(vals) if vals else 0.0


def get_common_file_names():
    human1_names = {f.name for f in HUMAN1_DIR.glob("*.json")}
    human2_names = {f.name for f in HUMAN2_DIR.glob("*.json")}
    llm_names = {f.name for f in LLM_DIR.glob("*.json")}

    common_names = sorted(human1_names & human2_names & llm_names)

    if COMPARE_ONLY_FIRST_COMMON_FILE and common_names:
        return [common_names[0]]

    return common_names


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    common_names = get_common_file_names()

    if not common_names:
        print("Não foram encontrados ficheiros comuns nas três pastas.")
        return

    all_results = {
        "speaker": [],
        "target": [],
        "joint": []
    }

    file_errors = []

    for file_name in common_names:
        try:
            h1_items = load_annotation_file(HUMAN1_DIR / file_name)
            h2_items = load_annotation_file(HUMAN2_DIR / file_name)
            llm_items = load_annotation_file(LLM_DIR / file_name)

            all_results["speaker"].append(process_field(file_name, "speaker", h1_items, h2_items, llm_items))
            all_results["target"].append(process_field(file_name, "target", h1_items, h2_items, llm_items))
            all_results["joint"].append(process_field(file_name, "joint", h1_items, h2_items, llm_items))

            print(f"OK: {file_name}")

        except Exception as e:
            file_errors.append((file_name, str(e)))
            print(f"ERRO em {file_name}: {e}")

    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    report_path = RESULTS_DIR / f"annotation_agreement_results_{timestamp}.txt"

    lines = []
    lines.append("THREE-WAY ANNOTATION AGREEMENT REPORT")
    lines.append(f"Date: {now.strftime('%Y-%m-%d')}")
    lines.append(f"Time: {now.strftime('%H:%M:%S')}")
    lines.append(f"Timestamp: {timestamp}")
    lines.append("")
    lines.append(f"Files requested for analysis: {len(common_names)}")
    for name in common_names:
        lines.append(f" - {name}")
    lines.append("")

    if file_errors:
        lines.append("FILES WITH ERRORS")
        for file_name, err in file_errors:
            lines.append(f" - {file_name}: {err}")
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
            lines.append(f"  Items: {result['n_items']}")
            lines.append(f"  3-way exact agreement: {result['three_way_exact_agreement']:.4f}")
            lines.append(f"  Fleiss' Kappa: {result['fleiss_kappa']:.4f}")
            lines.append("  Pairwise:")
            lines.append(
                f"    human1 vs human2 -> agreement={result['pairwise']['human1_vs_human2']['percent_agreement']:.4f}, "
                f"kappa={result['pairwise']['human1_vs_human2']['cohen_kappa']:.4f}"
            )
            lines.append(
                f"    human1 vs llm    -> agreement={result['pairwise']['human1_vs_llm']['percent_agreement']:.4f}, "
                f"kappa={result['pairwise']['human1_vs_llm']['cohen_kappa']:.4f}"
            )
            lines.append(
                f"    human2 vs llm    -> agreement={result['pairwise']['human2_vs_llm']['percent_agreement']:.4f}, "
                f"kappa={result['pairwise']['human2_vs_llm']['cohen_kappa']:.4f}"
            )
            lines.append("")

        lines.append("SUMMARY")
        lines.append(f"  Mean 3-way exact agreement: {average_metric(all_results[field], ['three_way_exact_agreement']):.4f}")
        lines.append(f"  Mean Fleiss' Kappa: {average_metric(all_results[field], ['fleiss_kappa']):.4f}")
        lines.append(
            f"  Mean human1 vs human2 Cohen's Kappa: "
            f"{average_metric(all_results[field], ['pairwise', 'human1_vs_human2', 'cohen_kappa']):.4f}"
        )
        lines.append(
            f"  Mean human1 vs llm Cohen's Kappa: "
            f"{average_metric(all_results[field], ['pairwise', 'human1_vs_llm', 'cohen_kappa']):.4f}"
        )
        lines.append(
            f"  Mean human2 vs llm Cohen's Kappa: "
            f"{average_metric(all_results[field], ['pairwise', 'human2_vs_llm', 'cohen_kappa']):.4f}"
        )
        lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Relatório criado com sucesso: {report_path}")


if __name__ == "__main__":
    main()