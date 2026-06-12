import json
import shutil
from pathlib import Path
from datetime import datetime

import pandas as pd
from parshift import Parshift

BASE_DIR = Path(__file__).resolve().parent

PARSHIFT_DIR = BASE_DIR / "data" / "processed" / "parshift"

INPUT_CSV_DIR = PARSHIFT_DIR / "input_csv"
WAVE_1_DIR = INPUT_CSV_DIR / "wave_1"
WAVE_2_DIR = INPUT_CSV_DIR / "wave_2"

RESULTS_DIR = PARSHIFT_DIR / "results"
FINAL_RESULTS_DIR = RESULTS_DIR / "final_dataset"
REPORTS_DIR = PARSHIFT_DIR / "reports"
TEMP_DIR = PARSHIFT_DIR / "temp_parshift_input"

N_SEGMENTS = 3
CSV_DELIMITER = "\t"


def clean_output_dir(path: Path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def extract_legislature_number(file_name: str):
    name = Path(file_name).stem
    parts = name.split("_")

    for part in parts:
        if part.isdigit():
            return int(part)

    return 9999


def dataframe_to_records(df):
    try:
        return df.to_dict(orient="records")
    except Exception:
        return []


def prepare_parshift_input_csv(original_csv_path: Path, temp_dir: Path):
    temp_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(
        original_csv_path,
        sep=CSV_DELIMITER,
        dtype=str,
        keep_default_na=False
    )

    if "utterance" not in df.columns:
        if "utterance_text" in df.columns:
            df["utterance"] = df["utterance_text"]
        else:
            raise ValueError("CSV não tem coluna 'utterance' nem 'utterance_text'.")

    last_utterance_by_speaker = {}
    reply_to_ids = []

    for _, row in df.iterrows():
        speaker_id = row["speaker_id"]
        target_id = row["target_id"]

        reply_to_id = ""

        if target_id and target_id in last_utterance_by_speaker:
            reply_to_id = last_utterance_by_speaker[target_id]

        reply_to_ids.append(reply_to_id)
        last_utterance_by_speaker[speaker_id] = row["utterance_id"]

    df["reply_to_id"] = reply_to_ids

    required_columns = [
        "utterance_id",
        "speaker_id",
        "target_id",
        "reply_to_id",
        "utterance"
    ]

    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"CSV preparado continua sem a coluna obrigatória: {col}")

    df = df[required_columns]

    temp_csv_path = temp_dir / original_csv_path.name

    df.to_csv(
        temp_csv_path,
        sep=CSV_DELIMITER,
        index=False,
        encoding="utf-8"
    )

    return temp_csv_path


def collect_final_input_files():
    files = []

    for source_phase, input_dir in [
        ("annotation_phase_1", WAVE_1_DIR),
        ("annotation_phase_2", WAVE_2_DIR)
    ]:
        if not input_dir.exists():
            continue

        for csv_path in sorted(input_dir.glob("*.csv"), key=lambda p: extract_legislature_number(p.name)):
            files.append({
                "source_phase": source_phase,
                "path": csv_path,
                "legislature": extract_legislature_number(csv_path.name)
            })

    files = sorted(files, key=lambda item: (item["legislature"], item["source_phase"], item["path"].name))

    sample_counter_by_legislature = {}

    for item in files:
        legislature = item["legislature"]
        sample_counter_by_legislature[legislature] = sample_counter_by_legislature.get(legislature, 0) + 1
        item["sample"] = sample_counter_by_legislature[legislature]

    return files


def run_parshift_for_file(file_info, output_dir: Path):
    csv_path = file_info["path"]
    legislature = file_info["legislature"]
    sample = file_info["sample"]
    source_phase = file_info["source_phase"]

    leg_output_dir = output_dir / f"legislature_{legislature}"
    leg_output_dir.mkdir(parents=True, exist_ok=True)

    temp_sample_dir = TEMP_DIR / f"legislature_{legislature}" / f"sample_{sample}"

    model = Parshift()

    result = {
        "dataset": "final_dataset",
        "source_phase": source_phase,
        "legislature": legislature,
        "sample": sample,
        "input_file": csv_path.name,
        "success": False,
        "error": None,
        "prepared_input": None,
        "stats_files": [],
        "propensity_files": [],
        "stats_summary": []
    }

    try:
        prepared_csv_path = prepare_parshift_input_csv(csv_path, temp_sample_dir)
        result["prepared_input"] = str(prepared_csv_path)

        model.process(
            str(prepared_csv_path),
            N=N_SEGMENTS,
            delimiter=CSV_DELIMITER
        )

        stats_prefix = leg_output_dir / f"legislature_{legislature}_sample_{sample}_stats"
        prop_prefix = leg_output_dir / f"legislature_{legislature}_sample_{sample}_propensities"

        try:
            model.show_stats(filename=str(stats_prefix))
        except Exception as e:
            result["stats_export_warning"] = str(e)

        try:
            model.get_propensities(filename=str(prop_prefix))
        except Exception as e:
            result["propensities_export_warning"] = str(e)

        for file in sorted(leg_output_dir.glob("*")):
            if f"legislature_{legislature}_sample_{sample}" in file.name:
                if "stats" in file.name:
                    result["stats_files"].append(file.name)

                if "propensities" in file.name or "propensity" in file.name:
                    result["propensity_files"].append(file.name)

        if hasattr(model, "stats"):
            for idx, df in enumerate(model.stats):
                segment_csv = leg_output_dir / f"legislature_{legislature}_sample_{sample}_segment_{idx + 1}_stats.csv"
                segment_json = leg_output_dir / f"legislature_{legislature}_sample_{sample}_segment_{idx + 1}_stats.json"

                try:
                    df.to_csv(
                        segment_csv,
                        index=False,
                        encoding="utf-8"
                    )

                    segment_json.write_text(
                        json.dumps(
                            dataframe_to_records(df),
                            indent=2,
                            ensure_ascii=False
                        ),
                        encoding="utf-8"
                    )

                    result["stats_summary"].append({
                        "segment": idx + 1,
                        "rows": len(df),
                        "csv": segment_csv.name,
                        "json": segment_json.name
                    })

                except Exception as e:
                    result["stats_summary"].append({
                        "segment": idx + 1,
                        "error": str(e)
                    })

        result["success"] = True

    except Exception as e:
        result["success"] = False
        result["error"] = str(e)

    return result


def run_final_dataset(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    files = collect_final_input_files()
    results = []

    for file_info in files:
        print(
            f"A correr ParShift: conjunto final | "
            f"Legislatura {file_info['legislature']} | "
            f"Amostra {file_info['sample']} | "
            f"{file_info['path'].name}"
        )

        result = run_parshift_for_file(file_info, output_dir)
        results.append(result)

        if result["success"]:
            print(f"  OK: Legislatura {result['legislature']} | Amostra {result['sample']}")
        else:
            print(f"  ERRO: Legislatura {result['legislature']} | Amostra {result['sample']} -> {result['error']}")

    return results


def write_report(report_path: Path, output):
    results = output["results"]

    lines = []

    lines.append("PARSHIFT EXECUTION REPORT")
    lines.append("Dataset: final_dataset")
    lines.append(f"Generated at: {output['generated_at']}")
    lines.append("")
    lines.append(f"N segments: {N_SEGMENTS}")
    lines.append(f"CSV delimiter: {repr(CSV_DELIMITER)}")
    lines.append("")
    lines.append("=" * 80)
    lines.append("FINAL DATASET")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"Files processed: {len(results)}")
    lines.append(f"Successful: {sum(1 for r in results if r['success'])}")
    lines.append(f"Failed: {sum(1 for r in results if not r['success'])}")
    lines.append("")

    legislatures = sorted(set(r["legislature"] for r in results))

    for legislature in legislatures:
        leg_results = [r for r in results if r["legislature"] == legislature]

        lines.append("-" * 80)
        lines.append(f"Legislature {legislature}")
        lines.append("-" * 80)

        for result in sorted(leg_results, key=lambda r: r["sample"]):
            lines.append(f"Sample {result['sample']} | {result['input_file']}")
            lines.append(f"  Source phase: {result['source_phase']}")
            lines.append(f"  Success: {result['success']}")

            if result.get("prepared_input"):
                lines.append(f"  Prepared input: {result['prepared_input']}")

            if result["error"]:
                lines.append(f"  Error: {result['error']}")

            if "stats_export_warning" in result:
                lines.append(f"  Stats export warning: {result['stats_export_warning']}")

            if "propensities_export_warning" in result:
                lines.append(f"  Propensities export warning: {result['propensities_export_warning']}")

            if result["stats_files"]:
                lines.append("  Stats files:")
                for f in result["stats_files"]:
                    lines.append(f"    - {f}")

            if result["propensity_files"]:
                lines.append("  Propensity files:")
                for f in result["propensity_files"]:
                    lines.append(f"    - {f}")

            if result["stats_summary"]:
                lines.append("  Stats summary:")
                for item in result["stats_summary"]:
                    if "error" in item:
                        lines.append(f"    - Segment {item['segment']}: ERROR {item['error']}")
                    else:
                        lines.append(
                            f"    - Segment {item['segment']}: "
                            f"{item['rows']} rows | {item['csv']}"
                        )

            lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR)

    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    clean_output_dir(FINAL_RESULTS_DIR)

    final_results = run_final_dataset(FINAL_RESULTS_DIR)

    output = {
        "generated_at": timestamp,
        "dataset": "final_dataset",
        "n_segments": N_SEGMENTS,
        "delimiter": CSV_DELIMITER,
        "results": final_results
    }

    json_path = REPORTS_DIR / f"parshift_execution_final_dataset_{timestamp}.json"
    txt_path = REPORTS_DIR / f"parshift_execution_final_dataset_{timestamp}.txt"

    json_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    write_report(txt_path, output)

    print("")
    print(f"JSON criado: {json_path}")
    print(f"Relatório criado: {txt_path}")
    print(f"Resultados em: {FINAL_RESULTS_DIR}")
    print(f"CSV temporários em: {TEMP_DIR}")


if __name__ == "__main__":
    main()