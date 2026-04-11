import json
from pathlib import Path
from datetime import datetime
from collections import Counter

JSON_DIR = Path("data/processed/json")
METRICS_DIR = Path("data/processed/metrics_base")

def load_json_file(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def word_count(text: str):
    if not text:
        return 0
    return len(text.split())

def safe_str(value):
    if value is None:
        return "null"
    return str(value)

def main():
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    json_files = sorted(JSON_DIR.glob("*.json"))
    if not json_files:
        print(f"Não encontrei ficheiros JSON em: {JSON_DIR.resolve()}")
        return

    total_files = 0
    total_debates = 0
    total_utterances = 0
    total_words = 0
    total_generic_speakers = 0
    files_with_summary = 0

    unique_debate_ids = set()
    role_counter = Counter()
    speaker_string_counter = Counter()
    speaker_name_counter = Counter()
    party_counter = Counter()
    utterances_per_file = {}
    words_per_file = {}
    roles_per_file = {}
    invalid_files = []

    for json_file in json_files:
        try:
            data = load_json_file(json_file)
            total_files += 1

            debate_id = data.get("debate_id")
            if debate_id:
                unique_debate_ids.add(debate_id)

            utterances = data.get("utterances", [])
            utterances_per_file[json_file.name] = len(utterances)

            file_word_total = 0
            file_role_counter = Counter()
            has_summary = False

            for utt in utterances:
                total_utterances += 1

                text = utt.get("text", "")
                wc = word_count(text)
                total_words += wc
                file_word_total += wc

                speaker = utt.get("speaker", {})
                role = safe_str(speaker.get("role"))
                speaker_string = safe_str(speaker.get("string"))
                speaker_name = safe_str(speaker.get("name"))
                party = safe_str(speaker.get("party"))

                role_counter[role] += 1
                speaker_string_counter[speaker_string] += 1
                speaker_name_counter[speaker_name] += 1
                party_counter[party] += 1
                file_role_counter[role] += 1

                if utt.get("speaker_is_generic") is True:
                    total_generic_speakers += 1

                if speaker_string == "SUMÁRIO":
                    has_summary = True

            words_per_file[json_file.name] = file_word_total
            roles_per_file[json_file.name] = dict(file_role_counter)

            if has_summary:
                files_with_summary += 1

        except Exception as e:
            invalid_files.append((json_file.name, str(e)))

    total_debates = len(unique_debate_ids)
    avg_utterances_per_file = total_utterances / total_files if total_files else 0
    avg_words_per_utterance = total_words / total_utterances if total_utterances else 0
    avg_words_per_file = total_words / total_files if total_files else 0

    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")

    metrics_data = {
        "generated_at": {
            "date": now.strftime("%Y-%m-%d"),
            "time": now.strftime("%H:%M:%S"),
            "timestamp": timestamp
        },
        "global_metrics": {
            "total_files": total_files,
            "total_unique_debates": total_debates,
            "total_utterances": total_utterances,
            "total_words": total_words,
            "avg_utterances_per_file": avg_utterances_per_file,
            "avg_words_per_utterance": avg_words_per_utterance,
            "avg_words_per_file": avg_words_per_file,
            "total_generic_speakers": total_generic_speakers,
            "files_with_summary": files_with_summary,
            "total_distinct_roles": len(role_counter)
        },
        "role_metrics": dict(role_counter),
        "speaker_string_metrics": dict(speaker_string_counter),
        "speaker_name_metrics": dict(speaker_name_counter),
        "party_metrics": dict(party_counter),
        "utterances_per_file": utterances_per_file,
        "words_per_file": words_per_file,
        "roles_per_file": roles_per_file,
        "invalid_files": [
            {"file": file_name, "error": error}
            for file_name, error in invalid_files
        ]
    }

    json_output = METRICS_DIR / f"metrics_report_{timestamp}.json"
    txt_output = METRICS_DIR / f"metrics_report_{timestamp}.txt"

    with open(json_output, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, ensure_ascii=False, indent=2)

    lines = []
    lines.append("METRICS REPORT")
    lines.append(f"Data: {now.strftime('%Y-%m-%d')}")
    lines.append(f"Hora: {now.strftime('%H:%M:%S')}")
    lines.append(f"Timestamp: {timestamp}")
    lines.append("")
    lines.append("MÉTRICAS GLOBAIS")
    lines.append(f"Total de ficheiros: {total_files}")
    lines.append(f"Total de debates únicos: {total_debates}")
    lines.append(f"Total de utterances: {total_utterances}")
    lines.append(f"Total de palavras: {total_words}")
    lines.append(f"Média de utterances por ficheiro: {avg_utterances_per_file:.2f}")
    lines.append(f"Média de palavras por utterance: {avg_words_per_utterance:.2f}")
    lines.append(f"Média de palavras por ficheiro: {avg_words_per_file:.2f}")
    lines.append(f"Total de oradores genéricos: {total_generic_speakers}")
    lines.append(f"Ficheiros com SUMÁRIO: {files_with_summary}")
    lines.append(f"Total de roles distintos: {len(role_counter)}")
    lines.append("")

    lines.append("CONTAGEM DE ROLES")
    for role, count in role_counter.most_common():
        lines.append(f"{role}: {count}")
    lines.append("")

    lines.append("TOP 20 SPEAKER.STRING")
    for speaker_string, count in speaker_string_counter.most_common(20):
        lines.append(f"{speaker_string}: {count}")
    lines.append("")

    lines.append("CONTAGEM POR PARTIDO")
    for party, count in party_counter.most_common():
        lines.append(f"{party}: {count}")
    lines.append("")

    lines.append("UTTERANCES POR FICHEIRO")
    for file_name, count in sorted(utterances_per_file.items()):
        lines.append(f"{file_name}: {count}")
    lines.append("")

    lines.append("PALAVRAS POR FICHEIRO")
    for file_name, count in sorted(words_per_file.items()):
        lines.append(f"{file_name}: {count}")
    lines.append("")

    lines.append("ROLES POR FICHEIRO")
    for file_name, roles in sorted(roles_per_file.items()):
        lines.append(f"{file_name}:")
        for role, count in sorted(roles.items()):
            lines.append(f"  {role}: {count}")
        lines.append("")

    if invalid_files:
        lines.append("FICHEIROS COM ERRO")
        for file_name, error in invalid_files:
            lines.append(f"{file_name} -> {error}")
        lines.append("")

    with open(txt_output, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Métricas JSON criadas: {json_output}")
    print(f"Métricas TXT criadas: {txt_output}")

if __name__ == "__main__":
    main()