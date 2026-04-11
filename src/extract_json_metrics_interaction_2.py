import json
from pathlib import Path
from datetime import datetime
from collections import Counter, defaultdict
import statistics


JSON_DIR = Path("data/processed/json")
METRICS_DIR = Path("data/processed/metrics_Interaction_2")

#FOR TESTING
#JSON_DIR = Path("data/Testing/Json")
#METRICS_DIR = Path("data/Testing/Results")

def load_json_file(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def word_count(text):
    return len(text.split()) if text else 0


def normalize_speaker(value):
    if value is None:
        return "UNKNOWN_SPEAKER"
    value = str(value).strip()
    return value if value else "UNKNOWN_SPEAKER"


def normalize_role(value):
    if value is None:
        return "NO_ROLE"
    value = str(value).strip()
    return value if value else "NO_ROLE"


def main():
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    json_files = sorted(JSON_DIR.glob("*.json"))

    global_stats = {
        "total_files": 0,
        "total_utterances": 0,
        "total_words": 0,
    }

    role_counter = Counter()
    speaker_counter = Counter()
    transition_counter = Counter()

    per_file_metrics = {}

    all_block_lengths = []
    alternation_rates = []
    unique_speakers_counts = []

    for file in json_files:
        data = load_json_file(file)

        utterances = data.get("utterances", [])
        words_per_utt = []
        blocks = []

        prev_speaker = None
        current_block_size = 0
        changes = 0

        file_speaker_counter = Counter()
        file_role_counter = Counter()

        for utt in utterances:
            speaker_data = utt.get("speaker", {})
            speaker = normalize_speaker(speaker_data.get("string"))
            role = normalize_role(speaker_data.get("role"))
            text = utt.get("text", "")

            file_speaker_counter[speaker] += 1
            speaker_counter[speaker] += 1

            file_role_counter[role] += 1
            role_counter[role] += 1

            wc = word_count(text)
            words_per_utt.append(wc)

            if prev_speaker is not None:
                #adiconar contador
                if speaker != prev_speaker:
                    changes += 1
                    transition_counter[(prev_speaker, speaker)] += 1

                    if current_block_size > 0:
                        blocks.append(current_block_size)

                    current_block_size = 1
                else:
                    current_block_size += 1
            else:
                current_block_size = 1

            prev_speaker = speaker

        if current_block_size > 0:
            blocks.append(current_block_size)

        total_utts = len(utterances)
        total_words = sum(words_per_utt)
        unique_speakers = len(file_speaker_counter)

        alternation_rate = changes / total_utts if total_utts else 0
        avg_block = statistics.mean(blocks) if blocks else 0

        sorted_speakers = file_speaker_counter.most_common()
        top1 = sorted_speakers[0][1] / total_utts if total_utts else 0
        top3 = sum(x[1] for x in sorted_speakers[:3]) / total_utts if total_utts else 0

        per_file_metrics[file.name] = {
            "utterances": total_utts,
            "words": total_words,
            "unique_speakers": unique_speakers,
            "avg_words_per_utterance": total_words / total_utts if total_utts else 0,
            "alternation_rate": alternation_rate,
            "avg_block_length": avg_block,
            "top1_dominance": top1,
            "top3_dominance": top3,
            "roles": dict(file_role_counter)
        }

        global_stats["total_files"] += 1
        global_stats["total_utterances"] += total_utts
        global_stats["total_words"] += total_words

        all_block_lengths.extend(blocks)
        alternation_rates.append(alternation_rate)
        unique_speakers_counts.append(unique_speakers)

    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    output = {
        "global": {
            **global_stats,
            "avg_words_per_utterance": global_stats["total_words"] / global_stats["total_utterances"] if global_stats["total_utterances"] else 0,
            "avg_alternation_rate": statistics.mean(alternation_rates) if alternation_rates else 0,
            "avg_unique_speakers": statistics.mean(unique_speakers_counts) if unique_speakers_counts else 0,
            "avg_block_length": statistics.mean(all_block_lengths) if all_block_lengths else 0
        },
        "roles": dict(role_counter),
        "top_speakers": dict(speaker_counter.most_common(20)),
        "transitions": {f"{k[0]} -> {k[1]}": v for k, v in transition_counter.items()},
        "per_file": per_file_metrics
    }

    json_path = METRICS_DIR / f"metrics_{now}.json"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)



    print(f"Metrics geradas: {json_path}")



if __name__ == "__main__":
    main()