from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent

PARSHIFT_DIR = BASE_DIR / "data" / "processed" / "parshift"
AGGREGATED_DIR = PARSHIFT_DIR / "aggregated"
AGGREGATED_PLOTS_DIR = PARSHIFT_DIR / "plots" / "aggregated"

DPI = 300

AGGREGATED_PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def latest_file(folder: Path, pattern: str):
    files = sorted(folder.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def load_aggregated_results():
    path = latest_file(AGGREGATED_DIR, "all_parshift_results_*.csv")

    if path is None:
        raise FileNotFoundError(
            "Não encontrei all_parshift_results_*.csv em src/data/processed/parshift/aggregated"
        )

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

    required_columns = [
        "wave",
        "legislature",
        "segment",
        "pshift",
        "frequency",
        "probability"
    ]

    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        raise ValueError(f"Faltam colunas no ficheiro agregado: {missing}")

    df["frequency"] = pd.to_numeric(df["frequency"], errors="coerce").fillna(0)
    df["probability"] = pd.to_numeric(df["probability"], errors="coerce").fillna(0)
    df["legislature"] = pd.to_numeric(df["legislature"], errors="coerce").fillna(0).astype(int)
    df["segment"] = pd.to_numeric(df["segment"], errors="coerce").fillna(0).astype(int)

    return df, path


def save_bar_plot(df, x_col, y_col, title, xlabel, ylabel, output_path):
    plt.figure(figsize=(13, 7))
    plt.bar(df[x_col].astype(str), df[y_col])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close()


def save_line_plot(df, x_col, y_col, group_col, title, xlabel, ylabel, output_path):
    plt.figure(figsize=(14, 8))

    for group, group_df in df.groupby(group_col):
        group_df = group_df.sort_values(x_col)
        plt.plot(group_df[x_col], group_df[y_col], marker="o", label=str(group))

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close()


def save_grouped_bar_plot(df, category_col, group_col, value_col, title, xlabel, ylabel, output_path):
    pivot = df.pivot(index=category_col, columns=group_col, values=value_col).fillna(0)

    ax = pivot.plot(kind="bar", figsize=(14, 8))

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close()


def generate_aggregated_plots():
    df, source_path = load_aggregated_results()

    plot_paths = []

    by_wave_shift = (
        df.groupby(["wave", "pshift"], as_index=False)
        .agg(
            total_frequency=("frequency", "sum"),
            avg_probability=("probability", "mean")
        )
    )

    for wave in sorted(by_wave_shift["wave"].unique()):
        wave_df = by_wave_shift[by_wave_shift["wave"] == wave]

        wave_freq = wave_df.sort_values("total_frequency", ascending=False)
        output_path = AGGREGATED_PLOTS_DIR / f"{wave}_01_total_frequency_by_participation_shift.png"

        save_bar_plot(
            wave_freq,
            "pshift",
            "total_frequency",
            f"{wave} - Total Frequency by Participation Shift",
            "Participation Shift",
            "Total Frequency",
            output_path
        )

        plot_paths.append(output_path)

        wave_prob = wave_df.sort_values("avg_probability", ascending=False)
        output_path = AGGREGATED_PLOTS_DIR / f"{wave}_02_average_probability_by_participation_shift.png"

        save_bar_plot(
            wave_prob,
            "pshift",
            "avg_probability",
            f"{wave} - Average Probability by Participation Shift",
            "Participation Shift",
            "Average Probability",
            output_path
        )

        plot_paths.append(output_path)

    output_path = AGGREGATED_PLOTS_DIR / "both_waves_03_total_frequency_by_participation_shift.png"

    save_grouped_bar_plot(
        by_wave_shift,
        "pshift",
        "wave",
        "total_frequency",
        "Wave 1 vs Wave 2 - Total Frequency by Participation Shift",
        "Participation Shift",
        "Total Frequency",
        output_path
    )

    plot_paths.append(output_path)

    output_path = AGGREGATED_PLOTS_DIR / "both_waves_04_average_probability_by_participation_shift.png"

    save_grouped_bar_plot(
        by_wave_shift,
        "pshift",
        "wave",
        "avg_probability",
        "Wave 1 vs Wave 2 - Average Probability by Participation Shift",
        "Participation Shift",
        "Average Probability",
        output_path
    )

    plot_paths.append(output_path)

    comparison_df = by_wave_shift.pivot(
        index="pshift",
        columns="wave",
        values="avg_probability"
    ).reset_index()

    if "wave_1" in comparison_df.columns and "wave_2" in comparison_df.columns:
        comparison_df["wave_2_minus_wave_1"] = comparison_df["wave_2"] - comparison_df["wave_1"]
        comparison_df = comparison_df.sort_values("wave_2_minus_wave_1", ascending=False)

        output_path = AGGREGATED_PLOTS_DIR / "both_waves_05_probability_difference_wave_2_minus_wave_1.png"

        save_bar_plot(
            comparison_df,
            "pshift",
            "wave_2_minus_wave_1",
            "Probability Difference by Participation Shift: Wave 2 - Wave 1",
            "Participation Shift",
            "Probability Difference",
            output_path
        )

        plot_paths.append(output_path)

    selected_shifts = [
        "A0-X0",
        "A0-XA",
        "AB-BA",
        "AB-XA",
        "AB-B0",
        "AB-X0"
    ]

    evolution_df = (
        df[df["pshift"].isin(selected_shifts)]
        .groupby(["wave", "legislature", "pshift"], as_index=False)
        .agg(avg_probability=("probability", "mean"))
    )

    for wave in sorted(evolution_df["wave"].unique()):
        wave_df = evolution_df[evolution_df["wave"] == wave]

        output_path = AGGREGATED_PLOTS_DIR / f"{wave}_06_evolution_main_shifts_by_legislature.png"

        save_line_plot(
            wave_df,
            "legislature",
            "avg_probability",
            "pshift",
            f"{wave} - Evolution of Main Participation Shifts by Legislature",
            "Legislature",
            "Average Probability",
            output_path
        )

        plot_paths.append(output_path)

    segment_df = (
        df[df["pshift"].isin(selected_shifts)]
        .groupby(["wave", "segment", "pshift"], as_index=False)
        .agg(avg_probability=("probability", "mean"))
    )

    for wave in sorted(segment_df["wave"].unique()):
        wave_df = segment_df[segment_df["wave"] == wave]

        output_path = AGGREGATED_PLOTS_DIR / f"{wave}_07_segment_profile_main_shifts.png"

        save_line_plot(
            wave_df,
            "segment",
            "avg_probability",
            "pshift",
            f"{wave} - Segment Profile of Main Participation Shifts",
            "Segment",
            "Average Probability",
            output_path
        )

        plot_paths.append(output_path)

    class_map = {
        "AB-BA": "Turn Receiving",
        "AB-B0": "Turn Receiving",
        "AB-BY": "Turn Receiving",
        "A0-X0": "Turn Claiming",
        "A0-XA": "Turn Claiming",
        "A0-XY": "Turn Claiming",
        "AB-X0": "Turn Usurping",
        "AB-XA": "Turn Usurping",
        "AB-XB": "Turn Usurping",
        "AB-XY": "Turn Usurping",
        "A0-AY": "Turn Continuing",
        "AB-A0": "Turn Continuing",
        "AB-AY": "Turn Continuing"
    }

    df_class = df.copy()
    df_class["pshift_class"] = df_class["pshift"].map(class_map).fillna("Unknown")

    class_df = (
        df_class.groupby(["wave", "pshift_class"], as_index=False)
        .agg(total_frequency=("frequency", "sum"))
    )

    output_path = AGGREGATED_PLOTS_DIR / "both_waves_08_total_frequency_by_shift_class.png"

    save_grouped_bar_plot(
        class_df,
        "pshift_class",
        "wave",
        "total_frequency",
        "Wave 1 vs Wave 2 - Total Frequency by Participation Shift Class",
        "Participation Shift Class",
        "Total Frequency",
        output_path
    )

    plot_paths.append(output_path)

    report_path = AGGREGATED_PLOTS_DIR / "aggregated_plots_generated_files.txt"

    lines = []
    lines.append("AGGREGATED PARSHIFT PLOTS")
    lines.append(f"Source file: {source_path}")
    lines.append("")

    for path in plot_paths:
        lines.append(str(path))

    report_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Fonte agregada: {source_path}")
    print(f"Gráficos agregados criados em: {AGGREGATED_PLOTS_DIR}")

    for path in plot_paths:
        print(f" - {path}")


if __name__ == "__main__":
    generate_aggregated_plots()