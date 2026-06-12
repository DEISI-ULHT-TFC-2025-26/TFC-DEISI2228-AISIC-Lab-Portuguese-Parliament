import csv
import json
from pathlib import Path
from datetime import datetime
from collections import Counter

BASE_DIR = Path(__file__).resolve().parent / "data" / "processed" / "sample_by_legislature"
ANNOTATION1_DIR = BASE_DIR / "Annotation 1"
ANNOTATION2_DIR = BASE_DIR / "Annotation 2"

LLM_DIRS = {
    "ChatGPT": BASE_DIR / "ChatGPT",
    "Claude": BASE_DIR / "Claude",
    "Gemini": BASE_DIR / "Gemini",
}

RESULTS_DIR = BASE_DIR / "annotation_agreement_llm_tables"

COMPARE_ONLY_FIRST_COMMON_FILE = False

FIELDS = ["speaker", "target", "joint"]


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
        raise ValueError(f"Nao foi possivel interpretar o ficheiro JSON: {path}")

    if len(chunks) == 1:
        return chunks[0]

    merged = []
    for chunk in chunks:
        if isinstance(chunk, list):
            merged.extend(chunk)
        elif isinstance(chunk, dict):
            merged.append(chunk)
        else:
            raise ValueError(f"Estrutura JSON nao suportada em {path}: {type(chunk)}")

    return merged


def load_annotation_file(path: Path):
    data = load_json_robust(path)

    if not isinstance(data, list):
        raise ValueError(f"O ficheiro {path.name} nao contem uma lista JSON de anotacoes.")

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
            "joint": f"{speaker}|||{target}",
        }

    return items


def krippendorff_alpha_nominal(items_labels):
    if not items_labels:
        return None

    filtered_items = []
    for row in items_labels:
        values = [v for v in row if v is not None]
        if len(values) >= 2:
            filtered_items.append(values)

    if not filtered_items:
        return None

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


def compute_pairwise_alpha(items_a, items_b, field):
    common_ids = sorted(set(items_a.keys()) & set(items_b.keys()))
    if not common_ids:
        return None, 0

    items_2 = [[items_a[i][field], items_b[i][field]] for i in common_ids]
    return krippendorff_alpha_nominal(items_2), len(common_ids)


def average_values(values):
    filtered = [v for v in values if v is not None]
    return sum(filtered) / len(filtered) if filtered else None


def get_common_file_names(a1_dir: Path, a2_dir: Path, llm_dir: Path):
    a1_names = {f.name for f in a1_dir.glob("*.json")}
    a2_names = {f.name for f in a2_dir.glob("*.json")}
    llm_names = {f.name for f in llm_dir.glob("*.json")}

    common_names = sorted(a1_names & a2_names & llm_names)

    if COMPARE_ONLY_FIRST_COMMON_FILE and common_names:
        return [common_names[0]]

    return common_names


def write_csv(rows, output_path: Path):
    headers = [
        "file",
        "field",
        "n_a1_a2",
        "n_a1_llm",
        "n_a2_llm",
        "alpha_a1_a2",
        "alpha_a1_llm",
        "alpha_a2_llm",
        "avg_alpha",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        for row in rows:
            writer.writerow([row.get(h, "") for h in headers])


def format_float(value):
    if value is None:
        return ""
    return f"{value:.4f}"


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")

    for llm_name, llm_dir in LLM_DIRS.items():
        if not llm_dir.exists():
            print(f"Pasta LLM nao encontrada: {llm_dir}")
            continue

        common_names = get_common_file_names(ANNOTATION1_DIR, ANNOTATION2_DIR, llm_dir)
        if not common_names:
            print(f"Nao ha ficheiros comuns para {llm_name}.")
            continue

        rows = []
        field_rows = {field: [] for field in FIELDS}

        for file_name in common_names:
            try:
                a1_items = load_annotation_file(ANNOTATION1_DIR / file_name)
                a2_items = load_annotation_file(ANNOTATION2_DIR / file_name)
                llm_items = load_annotation_file(llm_dir / file_name)

                for field in FIELDS:
                    alpha_a1_a2, n_a1_a2 = compute_pairwise_alpha(a1_items, a2_items, field)
                    alpha_a1_llm, n_a1_llm = compute_pairwise_alpha(a1_items, llm_items, field)
                    alpha_a2_llm, n_a2_llm = compute_pairwise_alpha(a2_items, llm_items, field)

                    avg_alpha = average_values([alpha_a1_a2, alpha_a1_llm, alpha_a2_llm])

                    row = {
                        "file": file_name,
                        "field": field,
                        "n_a1_a2": n_a1_a2,
                        "n_a1_llm": n_a1_llm,
                        "n_a2_llm": n_a2_llm,
                        "alpha_a1_a2": format_float(alpha_a1_a2),
                        "alpha_a1_llm": format_float(alpha_a1_llm),
                        "alpha_a2_llm": format_float(alpha_a2_llm),
                        "avg_alpha": format_float(avg_alpha),
                    }

                    rows.append(row)
                    field_rows[field].append(row)

                print(f"OK: {llm_name} -> {file_name}")

            except Exception as exc:
                print(f"ERRO em {llm_name} {file_name}: {exc}")

        for field in FIELDS:
            field_set = field_rows[field]
            if not field_set:
                continue

            alpha_a1_a2_values = [
                float(r["alpha_a1_a2"]) for r in field_set if r["alpha_a1_a2"]
            ]
            alpha_a1_llm_values = [
                float(r["alpha_a1_llm"]) for r in field_set if r["alpha_a1_llm"]
            ]
            alpha_a2_llm_values = [
                float(r["alpha_a2_llm"]) for r in field_set if r["alpha_a2_llm"]
            ]
            avg_values = [float(r["avg_alpha"]) for r in field_set if r["avg_alpha"]]

            summary = {
                "file": "MEAN",
                "field": field,
                "n_a1_a2": "",
                "n_a1_llm": "",
                "n_a2_llm": "",
                "alpha_a1_a2": format_float(average_values(alpha_a1_a2_values)),
                "alpha_a1_llm": format_float(average_values(alpha_a1_llm_values)),
                "alpha_a2_llm": format_float(average_values(alpha_a2_llm_values)),
                "avg_alpha": format_float(average_values(avg_values)),
            }

            rows.append(summary)

        output_path = RESULTS_DIR / f"krippendorff_tables_{llm_name}_{timestamp}.csv"
        write_csv(rows, output_path)
        print(f"Tabela criada: {output_path}")


if __name__ == "__main__":
    main()

