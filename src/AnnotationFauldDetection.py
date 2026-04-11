import json
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent / "data" / "processed" / "annotation_agreement"
HUMAN1_DIR = BASE_DIR / "human1"
HUMAN2_DIR = BASE_DIR / "human2"
LLM_DIR = BASE_DIR / "LLM"
RESULTS_DIR = BASE_DIR / "annotation_results"

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
            "target": target
        }

    return items


def get_common_file_names():
    human1_names = {f.name for f in HUMAN1_DIR.glob("*.json")}
    human2_names = {f.name for f in HUMAN2_DIR.glob("*.json")}
    llm_names = {f.name for f in LLM_DIR.glob("*.json")}

    common_names = sorted(human1_names & human2_names & llm_names)

    if COMPARE_ONLY_FIRST_COMMON_FILE and common_names:
        return [common_names[0]]

    return common_names


def find_discrepancies(file_name, h1_items, h2_items, llm_items):
    common_ids = sorted(set(h1_items.keys()) & set(h2_items.keys()) & set(llm_items.keys()))

    speaker_discrepancies = []
    target_discrepancies = []

    for utt_id in common_ids:
        h1_speaker = h1_items[utt_id]["speaker"]
        h2_speaker = h2_items[utt_id]["speaker"]
        llm_speaker = llm_items[utt_id]["speaker"]

        h1_target = h1_items[utt_id]["target"]
        h2_target = h2_items[utt_id]["target"]
        llm_target = llm_items[utt_id]["target"]

        if len({h1_speaker, h2_speaker, llm_speaker}) > 1:
            speaker_discrepancies.append({
                "utterance_id": utt_id,
                "human1": h1_speaker,
                "human2": h2_speaker,
                "LLM": llm_speaker
            })

        if len({h1_target, h2_target, llm_target}) > 1:
            target_discrepancies.append({
                "utterance_id": utt_id,
                "human1": h1_target,
                "human2": h2_target,
                "LLM": llm_target
            })

    return {
        "file": file_name,
        "n_common_utterances": len(common_ids),
        "speaker_discrepancies": speaker_discrepancies,
        "target_discrepancies": target_discrepancies
    }


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    common_names = get_common_file_names()

    if not common_names:
        print("Não foram encontrados ficheiros comuns nas três pastas.")
        return

    all_results = []
    file_errors = []

    for file_name in common_names:
        try:
            h1_items = load_annotation_file(HUMAN1_DIR / file_name)
            h2_items = load_annotation_file(HUMAN2_DIR / file_name)
            llm_items = load_annotation_file(LLM_DIR / file_name)

            result = find_discrepancies(file_name, h1_items, h2_items, llm_items)
            all_results.append(result)

            print(f"OK: {file_name}")

        except Exception as e:
            file_errors.append((file_name, str(e)))
            print(f"ERRO em {file_name}: {e}")

    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    report_path = RESULTS_DIR / f"annotation_discrepancies_{timestamp}.txt"

    lines = []
    lines.append("ANNOTATION DISCREPANCIES REPORT")
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

    for result in all_results:
        lines.append("=" * 100)
        lines.append(f"FILE: {result['file']}")
        lines.append("=" * 100)
        lines.append(f"Utterances comuns entre os 3 ficheiros: {result['n_common_utterances']}")
        lines.append("")

        lines.append(f"DISCREPÂNCIAS EM SPEAKER ({len(result['speaker_discrepancies'])})")
        if result["speaker_discrepancies"]:
            for row in result["speaker_discrepancies"]:
                lines.append(f"  utterance_id: {row['utterance_id']}")
                lines.append(f"    human1: {row['human1']}")
                lines.append(f"    human2: {row['human2']}")
                lines.append(f"    LLM:    {row['LLM']}")
        else:
            lines.append("  Nenhuma discrepância em speaker.")
        lines.append("")

        lines.append(f"DISCREPÂNCIAS EM TARGET ({len(result['target_discrepancies'])})")
        if result["target_discrepancies"]:
            for row in result["target_discrepancies"]:
                lines.append(f"  utterance_id: {row['utterance_id']}")
                lines.append(f"    human1: {row['human1']}")
                lines.append(f"    human2: {row['human2']}")
                lines.append(f"    LLM:    {row['LLM']}")
        else:
            lines.append("  Nenhuma discrepância em target.")
        lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Relatório criado com sucesso: {report_path}")


if __name__ == "__main__":
    main()