import json
import math
import statistics
from pathlib import Path
from datetime import datetime
from collections import Counter, defaultdict

JSON_DIR = Path("data/processed/json")
METRICS_DIR = Path("data/processed/metrics 3")

# FOR TESTING
# JSON_DIR = Path("data/Testing/Json")
# METRICS_DIR = Path("data/Testing/Results")


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


def normalize_party(value):
    if value is None:
        return "NO_PARTY"
    value = str(value).strip()
    return value if value else "NO_PARTY"


def is_president_utterance(speaker_data):
    role = normalize_role(speaker_data.get("role"))
    speaker_string = normalize_speaker(speaker_data.get("string")).lower()
    return role == "president" or "presidente" in speaker_string


def is_summary_utterance(speaker_data):
    speaker_string = normalize_speaker(speaker_data.get("string"))
    return speaker_string == "SUMÁRIO"


def is_political_utterance(speaker_data):
    if is_president_utterance(speaker_data):
        return False
    if is_summary_utterance(speaker_data):
        return False
    return True


def safe_mean(values):
    return statistics.mean(values) if values else 0


def safe_std(values):
    if len(values) < 2:
        return 0
    return statistics.pstdev(values)


def percentile(sorted_values, p):
    if not sorted_values:
        return 0
    if len(sorted_values) == 1:
        return sorted_values[0]

    k = (len(sorted_values) - 1) * (p / 100)
    f = math.floor(k)
    c = math.ceil(k)

    if f == c:
        return sorted_values[int(k)]

    d0 = sorted_values[f] * (c - k)
    d1 = sorted_values[c] * (k - f)
    return d0 + d1


def build_box_stats(values):
    if not values:
        return {
            "count": 0,
            "min": 0,
            "q1": 0,
            "median": 0,
            "q3": 0,
            "p90": 0,
            "max": 0,
            "mean": 0,
            "std": 0,
            "iqr": 0,
            "lower_fence": 0,
            "upper_fence": 0,
            "whisker_low": 0,
            "whisker_high": 0
        }

    values_sorted = sorted(values)

    q1 = percentile(values_sorted, 25)
    median = percentile(values_sorted, 50)
    q3 = percentile(values_sorted, 75)
    p90 = percentile(values_sorted, 90)
    iqr = q3 - q1

    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr

    whisker_candidates_low = [v for v in values_sorted if v >= lower_fence]
    whisker_candidates_high = [v for v in values_sorted if v <= upper_fence]

    whisker_low = min(whisker_candidates_low) if whisker_candidates_low else values_sorted[0]
    whisker_high = max(whisker_candidates_high) if whisker_candidates_high else values_sorted[-1]

    return {
        "count": len(values_sorted),
        "min": values_sorted[0],
        "q1": q1,
        "median": median,
        "q3": q3,
        "p90": p90,
        "max": values_sorted[-1],
        "mean": safe_mean(values_sorted),
        "std": safe_std(values_sorted),
        "iqr": iqr,
        "lower_fence": lower_fence,
        "upper_fence": upper_fence,
        "whisker_low": whisker_low,
        "whisker_high": whisker_high
    }


def sorted_counter_dict(counter_obj):
    return dict(sorted(counter_obj.items(), key=lambda x: x[1], reverse=True))


def main():
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    json_files = sorted(JSON_DIR.glob("*.json"))

    global_stats = {
        "total_files": 0,
        "total_utterances": 0,
        "total_words": 0,
    }

    speaker_counter = Counter()
    party_counter = Counter()
    party_counter_without_president = Counter()

    transition_counter = Counter()
    transitions_by_legislature = defaultdict(Counter)

    per_file_metrics = {}

    all_block_lengths = []
    alternation_rates = []
    unique_speakers_counts = []
    same_speaker_follow_rates = []
    avg_words_per_utterance_values = []
    top1_dominance_values = []
    top3_dominance_values = []
    top1_party_dominance_values = []
    top3_party_dominance_values = []
    top1_party_dominance_without_president_values = []
    top3_party_dominance_without_president_values = []
    utterances_per_file_values = []
    words_per_file_values = []

    unique_speakers_by_leg_session = defaultdict(list)
    unique_speakers_by_debate = {}

    for file in json_files:
        data = load_json_file(file)

        debate_meta = data.get("debate_meta", {})
        legislature = str(debate_meta.get("legislature", "UNKNOWN_LEGISLATURE"))
        legislative_session = str(debate_meta.get("legislative_session", "UNKNOWN_SESSION"))
        leg_session_key = f"legislature_{legislature}_session_{legislative_session}"

        utterances = data.get("utterances", [])
        words_per_utt = []
        blocks = []

        prev_speaker = None
        current_block_size = 0
        changes = 0
        same_speaker_follow_count = 0

        file_speaker_counter = Counter()
        file_role_counter = Counter()
        file_party_counter = Counter()
        file_party_counter_without_president = Counter()

        for utt in utterances:
            speaker_data = utt.get("speaker", {})
            speaker = normalize_speaker(speaker_data.get("string"))
            role = normalize_role(speaker_data.get("role"))
            party = normalize_party(speaker_data.get("party"))
            text = utt.get("text", "")

            file_speaker_counter[speaker] += 1
            speaker_counter[speaker] += 1

            file_role_counter[role] += 1

            file_party_counter[party] += 1
            party_counter[party] += 1

            if is_political_utterance(speaker_data):
                file_party_counter_without_president[party] += 1
                party_counter_without_president[party] += 1

            wc = word_count(text)
            words_per_utt.append(wc)

            if prev_speaker is not None:
                if speaker != prev_speaker:
                    changes += 1
                    transition_counter[(prev_speaker, speaker)] += 1
                    transitions_by_legislature[legislature][(prev_speaker, speaker)] += 1

                    if current_block_size > 0:
                        blocks.append(current_block_size)

                    current_block_size = 1
                else:
                    same_speaker_follow_count += 1
                    current_block_size += 1
            else:
                current_block_size = 1

            prev_speaker = speaker

        if current_block_size > 0:
            blocks.append(current_block_size)

        total_utts = len(utterances)
        total_words = sum(words_per_utt)
        unique_speakers = len(file_speaker_counter)
        unique_parties = len(file_party_counter)

        alternation_rate = changes / total_utts if total_utts else 0
        avg_block = safe_mean(blocks)
        same_speaker_follow_rate = same_speaker_follow_count / total_utts if total_utts else 0
        avg_words_per_utt = total_words / total_utts if total_utts else 0

        sorted_speakers = file_speaker_counter.most_common()
        top1 = sorted_speakers[0][1] / total_utts if total_utts else 0
        top3 = sum(x[1] for x in sorted_speakers[:3]) / total_utts if total_utts else 0

        sorted_parties = file_party_counter.most_common()
        top1_party = sorted_parties[0][1] / total_utts if total_utts else 0
        top3_party = sum(x[1] for x in sorted_parties[:3]) / total_utts if total_utts else 0

        political_only_total = sum(file_party_counter_without_president.values())
        sorted_parties_without_president = file_party_counter_without_president.most_common()
        top1_party_without_president = (
            sorted_parties_without_president[0][1] / political_only_total
            if political_only_total and sorted_parties_without_president else 0
        )
        top3_party_without_president = (
            sum(x[1] for x in sorted_parties_without_president[:3]) / political_only_total
            if political_only_total else 0
        )

        per_file_metrics[file.name] = {
            "legislature": legislature,
            "legislative_session": legislative_session,
            "utterances": total_utts,
            "words": total_words,
            "unique_speakers": unique_speakers,
            "unique_parties": unique_parties,
            "avg_words_per_utterance": avg_words_per_utt,
            "alternation_rate": alternation_rate,
            "avg_block_length": avg_block,
            "same_speaker_follow_count": same_speaker_follow_count,
            "same_speaker_follow_rate": same_speaker_follow_rate,
            "top1_dominance": top1,
            "top3_dominance": top3,
            "top1_party_dominance": top1_party,
            "top3_party_dominance": top3_party,
            "top1_party_dominance_without_president": top1_party_without_president,
            "top3_party_dominance_without_president": top3_party_without_president,
            "parties": sorted_counter_dict(file_party_counter),
            "parties_without_president": sorted_counter_dict(file_party_counter_without_president),
        }

        global_stats["total_files"] += 1
        global_stats["total_utterances"] += total_utts
        global_stats["total_words"] += total_words

        all_block_lengths.extend(blocks)
        alternation_rates.append(alternation_rate)
        unique_speakers_counts.append(unique_speakers)
        same_speaker_follow_rates.append(same_speaker_follow_rate)
        avg_words_per_utterance_values.append(avg_words_per_utt)
        top1_dominance_values.append(top1)
        top3_dominance_values.append(top3)
        top1_party_dominance_values.append(top1_party)
        top3_party_dominance_values.append(top3_party)
        top1_party_dominance_without_president_values.append(top1_party_without_president)
        top3_party_dominance_without_president_values.append(top3_party_without_president)
        utterances_per_file_values.append(total_utts)
        words_per_file_values.append(total_words)

        unique_speakers_by_leg_session[leg_session_key].append(unique_speakers)
        unique_speakers_by_debate[file.name] = build_box_stats([unique_speakers])

    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    transitions_by_legislature_output = {}
    for leg in sorted(transitions_by_legislature.keys(), key=lambda x: int(x) if x.isdigit() else x):
        counter_obj = transitions_by_legislature[leg]
        ordered = sorted(counter_obj.items(), key=lambda x: x[1], reverse=True)
        transitions_by_legislature_output[leg] = {
            f"{k[0]} -> {k[1]}": v for k, v in ordered
        }

    output = {
        "generated_at": now,
        "global": {
            **global_stats,
            "avg_words_per_utterance": global_stats["total_words"] / global_stats["total_utterances"] if global_stats["total_utterances"] else 0,
            "avg_alternation_rate": safe_mean(alternation_rates),
            "avg_unique_speakers": safe_mean(unique_speakers_counts),
            "avg_block_length": safe_mean(all_block_lengths),
            "avg_same_speaker_follow_rate": safe_mean(same_speaker_follow_rates),
            "avg_top1_dominance": safe_mean(top1_dominance_values),
            "avg_top3_dominance": safe_mean(top3_dominance_values),
            "avg_top1_party_dominance": safe_mean(top1_party_dominance_values),
            "avg_top3_party_dominance": safe_mean(top3_party_dominance_values),
            "avg_top1_party_dominance_without_president": safe_mean(top1_party_dominance_without_president_values),
            "avg_top3_party_dominance_without_president": safe_mean(top3_party_dominance_without_president_values)
        },
        "top_speakers": dict(speaker_counter.most_common(20)),
        "all_speakers": sorted_counter_dict(speaker_counter),
        "party_metrics": {
            "full": sorted_counter_dict(party_counter),
            "without_president": sorted_counter_dict(party_counter_without_president)
        },
        "transitions": {
            "global": {
                f"{k[0]} -> {k[1]}": v
                for k, v in sorted(transition_counter.items(), key=lambda x: x[1], reverse=True)
            },
            "by_legislature": transitions_by_legislature_output
        },
        "per_file": per_file_metrics
    }

    boxplot_output = {
        "generated_at": now,
        "boxplot_metrics": {
            "unique_speakers": {
                "all_debates": build_box_stats(unique_speakers_counts),
                "by_legislature_session": {
                    key: build_box_stats(values)
                    for key, values in sorted(unique_speakers_by_leg_session.items())
                },
                "by_debate": unique_speakers_by_debate
            }
        }
    }

    json_path = METRICS_DIR / f"metrics_{now}.json"
    boxplot_path = METRICS_DIR / f"boxplot_{now}.json"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    with open(boxplot_path, "w", encoding="utf-8") as f:
        json.dump(boxplot_output, f, indent=2, ensure_ascii=False)

    print(f"Metrics geradas: {json_path}")
    print(f"Boxplot stats geradas: {boxplot_path}")


if __name__ == "__main__":
    main()