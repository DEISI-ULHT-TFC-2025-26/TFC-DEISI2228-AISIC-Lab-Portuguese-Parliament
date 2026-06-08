import json
import re
import shutil
from pathlib import Path
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import squarify

from parshift import Parshift

BASE_DIR = Path(__file__).resolve().parent

PARSHIFT_DIR = BASE_DIR / "data" / "processed" / "parshift"

INPUT_CSV_DIR = PARSHIFT_DIR / "input_csv"
WAVE_1_DIR = INPUT_CSV_DIR / "wave_1"
WAVE_2_DIR = INPUT_CSV_DIR / "wave_2"

AGGREGATED_DIR = PARSHIFT_DIR / "aggregated"
REPORTS_DIR = PARSHIFT_DIR / "reports"

PLOTS_DIR = PARSHIFT_DIR / "plots"
NATIVE_DIR = PLOTS_DIR / "native"
NATIVE_WAVE_1_DIR = NATIVE_DIR / "wave_1"
NATIVE_WAVE_2_DIR = NATIVE_DIR / "wave_2"

FINAL_BY_LEGISLATURE_DIR = PLOTS_DIR / "final_by_legislature"
FINAL_GLOBAL_DIR = PLOTS_DIR / "final_global"

TEMP_INPUT_DIR = PARSHIFT_DIR / "temp_plots_input"

N_SEGMENTS = 3
CSV_DELIMITER = "\t"
DPI = 300


def extract_legislature_number(file_name):
    match = re.search(r"legislature_(\d+)", file_name)
    return int(match.group(1)) if match else 9999


def latest_file(folder: Path, pattern: str):
    files = sorted(folder.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


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

    df = df[
        [
            "utterance_id",
            "speaker_id",
            "target_id",
            "reply_to_id",
            "utterance"
        ]
    ]

    temp_csv_path = temp_dir / original_csv_path.name

    df.to_csv(
        temp_csv_path,
        sep=CSV_DELIMITER,
        index=False,
        encoding="utf-8"
    )

    return temp_csv_path


def get_input_csv_files(input_dir: Path):
    return sorted(
        [
            p for p in input_dir.glob("legislature_*.csv")
            if "_with_reply" not in p.name
        ],
        key=lambda p: extract_legislature_number(p.name)
    )


def generate_native_plots_for_wave(wave_name, input_dir, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []

    csv_files = get_input_csv_files(input_dir)

    for csv_path in csv_files:
        legislature = extract_legislature_number(csv_path.name)

        try:
            temp_dir = TEMP_INPUT_DIR / wave_name / f"legislature_{legislature}"
            prepared_csv_path = prepare_parshift_input_csv(csv_path, temp_dir)

            model = Parshift()
            model.process(
                str(prepared_csv_path),
                N=N_SEGMENTS,
                delimiter=CSV_DELIMITER
            )

            pshift_path = output_dir / f"{wave_name}_legislature_{legislature:02d}_participation_shifts_treemap.png"
            pclass_path = output_dir / f"{wave_name}_legislature_{legislature:02d}_shift_classes_treemap.png"

            model.show_plot(type="Pshift", filename=str(pshift_path))
            plt.close("all")

            model.show_plot(type="Pshift_class", filename=str(pclass_path))
            plt.close("all")

            results.append({
                "wave": wave_name,
                "legislature": legislature,
                "success": True,
                "prepared_csv": str(prepared_csv_path),
                "participation_shifts_plot": str(pshift_path),
                "shift_classes_plot": str(pclass_path),
                "error": None
            })

            print(f"OK native: {wave_name} legislature {legislature}")

        except Exception as e:
            results.append({
                "wave": wave_name,
                "legislature": legislature,
                "success": False,
                "prepared_csv": "",
                "participation_shifts_plot": "",
                "shift_classes_plot": "",
                "error": str(e)
            })

            print(f"ERRO native: {wave_name} legislature {legislature} -> {e}")

    return results


def combine_two_images(left_image, right_image, output_path, title, left_title, right_title):
    left = mpimg.imread(left_image)
    right = mpimg.imread(right_image)

    fig, axes = plt.subplots(1, 2, figsize=(18, 9))

    axes[0].imshow(left)
    axes[0].axis("off")
    axes[0].set_title(left_title)

    axes[1].imshow(right)
    axes[1].axis("off")
    axes[1].set_title(right_title)

    fig.suptitle(title, fontsize=16)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close()


def generate_final_legislature_montages():
    FINAL_BY_LEGISLATURE_DIR.mkdir(parents=True, exist_ok=True)

    results = []

    for legislature in range(1, 15):
        wave_1_plot = NATIVE_WAVE_1_DIR / f"wave_1_legislature_{legislature:02d}_participation_shifts_treemap.png"
        wave_2_plot = NATIVE_WAVE_2_DIR / f"wave_2_legislature_{legislature:02d}_participation_shifts_treemap.png"

        output_path = FINAL_BY_LEGISLATURE_DIR / f"legislature_{legislature:02d}_wave_1_vs_wave_2_participation_shifts.png"

        if wave_1_plot.exists() and wave_2_plot.exists():
            combine_two_images(
                wave_1_plot,
                wave_2_plot,
                output_path,
                f"Legislature {legislature} - Wave 1 vs Wave 2",
                "Wave 1",
                "Wave 2"
            )

            results.append(str(output_path))
            print(f"OK montagem legislatura {legislature}")

        else:
            print(f"ERRO montagem legislatura {legislature}: falta wave_1 ou wave_2")

    return results


def load_aggregated_results():
    path = latest_file(AGGREGATED_DIR, "all_parshift_results_*.csv")

    if path is None:
        path = latest_file(AGGREGATED_DIR, "parshift_by_wave_*.csv")

    if path is None:
        path = latest_file(AGGREGATED_DIR, "parshift_by_wave_legislature_*.csv")

    if path is None:
        raise FileNotFoundError("Não encontrei CSV agregado em parshift/aggregated")

    df = pd.read_csv(path)

    rename_map = {
        "Pshift": "pshift",
        "Frequency": "frequency",
        "Probability": "probability",
        "Wave": "wave",
        "Legislature": "legislature",
        "Segment": "segment"
    }

    df = df.rename(columns=rename_map)

    required_columns = ["wave", "pshift", "frequency"]

    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        raise ValueError(f"Faltam colunas no agregado: {missing}")

    df["frequency"] = pd.to_numeric(df["frequency"], errors="coerce").fillna(0)

    return df, path


def save_global_treemap(ax, data, title):
    data = data[data["frequency"] > 0].copy()
    data = data.sort_values("frequency", ascending=False)

    labels = [
        f"{row.pshift}\n{int(row.frequency)}"
        for row in data.itertuples()
    ]

    sizes = data["frequency"].tolist()

    ax.set_title(title)

    if not sizes:
        ax.text(0.5, 0.5, "Sem dados", ha="center", va="center")
        ax.axis("off")
        return

    squarify.plot(
        sizes=sizes,
        label=labels,
        ax=ax,
        alpha=0.8,
        text_kwargs={"fontsize": 9}
    )

    ax.axis("off")


def generate_global_final_plot():
    FINAL_GLOBAL_DIR.mkdir(parents=True, exist_ok=True)

    df, source_path = load_aggregated_results()

    by_wave = (
        df.groupby(["wave", "pshift"], as_index=False)
        .agg(frequency=("frequency", "sum"))
    )

    output_path = FINAL_GLOBAL_DIR / "all_legislatures_wave_1_vs_wave_2_participation_shifts_overview.png"

    fig, axes = plt.subplots(1, 2, figsize=(18, 9))

    wave_1_df = by_wave[by_wave["wave"] == "wave_1"]
    wave_2_df = by_wave[by_wave["wave"] == "wave_2"]

    save_global_treemap(
        axes[0],
        wave_1_df,
        "Wave 1 - All Legislatures"
    )

    save_global_treemap(
        axes[1],
        wave_2_df,
        "Wave 2 - All Legislatures"
    )

    fig.suptitle("Global Participation Shifts Overview - Wave 1 vs Wave 2", fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close()

    print(f"OK gráfico global: {output_path}")

    return str(output_path), str(source_path)


def write_report(native_results, montage_paths, global_plot, aggregated_source):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    report_path = REPORTS_DIR / f"parshift_final_plots_report_{timestamp}.txt"

    lines = []

    lines.append("PARSHIFT FINAL PLOTS REPORT")
    lines.append(f"Generated at: {timestamp}")
    lines.append("")
    lines.append(f"Aggregated source: {aggregated_source}")
    lines.append("")
    lines.append("=" * 80)
    lines.append("NATIVE PLOTS")
    lines.append("=" * 80)

    for result in native_results:
        lines.append(
            f"{result['wave']} | Legislature {result['legislature']} | Success: {result['success']}"
        )

        if result["error"]:
            lines.append(f"  Error: {result['error']}")
        else:
            lines.append(f"  Participation shifts: {result['participation_shifts_plot']}")
            lines.append(f"  Shift classes: {result['shift_classes_plot']}")

    lines.append("")
    lines.append("=" * 80)
    lines.append("FINAL LEGISLATURE MONTAGES")
    lines.append("=" * 80)

    for path in montage_paths:
        lines.append(f" - {path}")

    lines.append("")
    lines.append("=" * 80)
    lines.append("GLOBAL FINAL PLOT")
    lines.append("=" * 80)
    lines.append(f" - {global_plot}")

    report_path.write_text("\n".join(lines), encoding="utf-8")

    return report_path


def main():
    NATIVE_WAVE_1_DIR.mkdir(parents=True, exist_ok=True)
    NATIVE_WAVE_2_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_BY_LEGISLATURE_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_GLOBAL_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    if TEMP_INPUT_DIR.exists():
        shutil.rmtree(TEMP_INPUT_DIR)

    TEMP_INPUT_DIR.mkdir(parents=True, exist_ok=True)

    native_results = []

    native_results.extend(
        generate_native_plots_for_wave(
            "wave_1",
            WAVE_1_DIR,
            NATIVE_WAVE_1_DIR
        )
    )

    native_results.extend(
        generate_native_plots_for_wave(
            "wave_2",
            WAVE_2_DIR,
            NATIVE_WAVE_2_DIR
        )
    )

    montage_paths = generate_final_legislature_montages()

    global_plot, aggregated_source = generate_global_final_plot()

    report_path = write_report(
        native_results,
        montage_paths,
        global_plot,
        aggregated_source
    )

    print("")
    print(f"Native plots em: {NATIVE_DIR}")
    print(f"Montagens finais por legislatura em: {FINAL_BY_LEGISLATURE_DIR}")
    print(f"Gráfico global final em: {FINAL_GLOBAL_DIR}")
    print(f"Relatório criado: {report_path}")


if __name__ == "__main__":
    main()