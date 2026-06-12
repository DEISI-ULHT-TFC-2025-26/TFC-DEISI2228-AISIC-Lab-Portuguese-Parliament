from pathlib import Path
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt
import squarify

BASE_DIR = Path(__file__).resolve().parent

PARSHIFT_DIR = BASE_DIR / "data" / "processed" / "parshift"
AGGREGATED_DIR = PARSHIFT_DIR / "aggregated"
PLOTS_DIR = PARSHIFT_DIR / "plots"

FINAL_PLOTS_DIR = PLOTS_DIR / "final_dataset"
GLOBAL_DIR = FINAL_PLOTS_DIR / "global"
LEGISLATURE_DIR = FINAL_PLOTS_DIR / "by_legislature"
SEGMENT_DIR = FINAL_PLOTS_DIR / "by_segment"
CLASS_DIR = FINAL_PLOTS_DIR / "by_class"
TREEMAP_DIR = FINAL_PLOTS_DIR / "treemaps"
REPORTS_DIR = PARSHIFT_DIR / "reports"

DPI = 300

MAIN_SHIFTS = [
    "A0-X0",
    "A0-XA",
    "AB-B0",
    "AB-BA",
    "AB-XA",
    "AB-X0",
    "AB-A0"
]

CLASS_ORDER = [
    "Turn Claiming",
    "Turn Receiving",
    "Turn Usurping",
    "Turn Continuing"
]


def latest_file(folder: Path, pattern: str):
    files = sorted(folder.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def ensure_dirs():
    for folder in [
        FINAL_PLOTS_DIR,
        GLOBAL_DIR,
        LEGISLATURE_DIR,
        SEGMENT_DIR,
        CLASS_DIR,
        TREEMAP_DIR,
        REPORTS_DIR
    ]:
        folder.mkdir(parents=True, exist_ok=True)


def load_csv(pattern: str, required_columns):
    path = latest_file(AGGREGATED_DIR, pattern)

    if path is None:
        raise FileNotFoundError(f"Não encontrei ficheiro com o padrão: {pattern}")

    df = pd.read_csv(path)

    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        raise ValueError(f"O ficheiro {path.name} não contém as colunas obrigatórias: {missing}")

    return df, path


def normalize_numeric(df, columns):
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    return df


def save_bar_plot(df, x_col, y_col, title, xlabel, ylabel, output_path, rotation=45):
    plt.figure(figsize=(13, 7))
    plt.bar(df[x_col].astype(str), df[y_col])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=rotation, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close()

    return output_path


def save_line_plot(df, x_col, y_col, group_col, title, xlabel, ylabel, output_path):
    plt.figure(figsize=(14, 8))

    for group, group_df in df.groupby(group_col):
        group_df = group_df.sort_values(x_col)
        plt.plot(
            group_df[x_col],
            group_df[y_col],
            marker="o",
            label=str(group)
        )

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close()

    return output_path


def save_grouped_bar_plot(df, category_col, group_col, value_col, title, xlabel, ylabel, output_path):
    pivot = df.pivot_table(
        index=category_col,
        columns=group_col,
        values=value_col,
        aggfunc="sum",
        fill_value=0
    )

    ax = pivot.plot(kind="bar", figsize=(16, 8))

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    plt.xticks(rotation=0)
    plt.legend(title=group_col)
    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close()

    return output_path


def save_treemap(df, label_col, value_col, title, output_path):
    data = df[df[value_col] > 0].copy()
    data = data.sort_values(value_col, ascending=False)

    labels = [
        f"{row[label_col]}\n{int(row[value_col])}"
        for _, row in data.iterrows()
    ]

    sizes = data[value_col].tolist()

    plt.figure(figsize=(13, 8))

    if sizes:
        squarify.plot(
            sizes=sizes,
            label=labels,
            alpha=0.85,
            text_kwargs={"fontsize": 10}
        )
        plt.axis("off")
    else:
        plt.text(0.5, 0.5, "Sem dados", ha="center", va="center")
        plt.axis("off")

    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close()

    return output_path


def generate_global_plots():
    df, source_path = load_csv(
        "parshift_global_*.csv",
        ["pshift", "total_frequency", "global_probability"]
    )

    df = normalize_numeric(df, ["total_frequency", "global_probability"])
    df = df[df["pshift"].isin(MAIN_SHIFTS)].copy()

    plot_paths = []

    frequency_df = df.sort_values("total_frequency", ascending=False)

    plot_paths.append(
        save_bar_plot(
            frequency_df,
            "pshift",
            "total_frequency",
            "Frequência global dos principais Participation Shifts",
            "Participation Shift",
            "Frequência total",
            GLOBAL_DIR / "01_global_frequency_by_participation_shift.png"
        )
    )

    probability_df = df.sort_values("global_probability", ascending=False)

    plot_paths.append(
        save_bar_plot(
            probability_df,
            "pshift",
            "global_probability",
            "Probabilidade global dos principais Participation Shifts",
            "Participation Shift",
            "Probabilidade global",
            GLOBAL_DIR / "02_global_probability_by_participation_shift.png"
        )
    )

    plot_paths.append(
        save_treemap(
            frequency_df,
            "pshift",
            "total_frequency",
            "Distribuição global dos Participation Shifts",
            TREEMAP_DIR / "01_global_participation_shifts_treemap.png"
        )
    )

    return plot_paths, source_path


def generate_legislature_plots():
    df, source_path = load_csv(
        "parshift_by_legislature_*.csv",
        ["legislature", "pshift", "total_frequency", "global_probability"]
    )

    df = normalize_numeric(df, ["legislature", "total_frequency", "global_probability"])
    df["legislature"] = df["legislature"].astype(int)

    df = df[df["pshift"].isin(MAIN_SHIFTS)].copy()

    plot_paths = []

    plot_paths.append(
        save_line_plot(
            df,
            "legislature",
            "global_probability",
            "pshift",
            "Evolução dos principais Participation Shifts por legislatura",
            "Legislatura",
            "Probabilidade global",
            LEGISLATURE_DIR / "01_main_participation_shifts_by_legislature_line.png"
        )
    )

    plot_paths.append(
        save_grouped_bar_plot(
            df,
            "legislature",
            "pshift",
            "global_probability",
            "Distribuição dos principais Participation Shifts por legislatura",
            "Legislatura",
            "Probabilidade global",
            LEGISLATURE_DIR / "02_main_participation_shifts_by_legislature_bars.png"
        )
    )

    top_by_legislature = (
        df.sort_values(["legislature", "total_frequency"], ascending=[True, False])
        .groupby("legislature")
        .head(1)
        .copy()
    )

    top_by_legislature["label"] = top_by_legislature["legislature"].apply(lambda x: f"L{x}")

    plot_paths.append(
        save_bar_plot(
            top_by_legislature,
            "label",
            "global_probability",
            "Padrão dominante por legislatura",
            "Legislatura",
            "Probabilidade do padrão dominante",
            LEGISLATURE_DIR / "03_dominant_participation_shift_by_legislature.png",
            rotation=0
        )
    )

    for legislature in sorted(df["legislature"].unique()):
        leg_df = df[df["legislature"] == legislature].sort_values("total_frequency", ascending=False)

        plot_paths.append(
            save_treemap(
                leg_df,
                "pshift",
                "total_frequency",
                f"Participation Shifts na Legislatura {legislature}",
                TREEMAP_DIR / f"legislature_{legislature:02d}_participation_shifts_treemap.png"
            )
        )

    return plot_paths, source_path


def generate_segment_plots():
    df, source_path = load_csv(
        "parshift_by_segment_*.csv",
        ["segment", "pshift", "total_frequency", "global_probability"]
    )

    df = normalize_numeric(df, ["segment", "total_frequency", "global_probability"])
    df["segment"] = df["segment"].astype(int)

    df = df[df["pshift"].isin(MAIN_SHIFTS)].copy()

    plot_paths = []

    plot_paths.append(
        save_line_plot(
            df,
            "segment",
            "global_probability",
            "pshift",
            "Perfil dos principais Participation Shifts por segmento",
            "Segmento",
            "Probabilidade global",
            SEGMENT_DIR / "01_main_participation_shifts_by_segment_line.png"
        )
    )

    plot_paths.append(
        save_grouped_bar_plot(
            df,
            "segment",
            "pshift",
            "global_probability",
            "Distribuição dos principais Participation Shifts por segmento",
            "Segmento",
            "Probabilidade global",
            SEGMENT_DIR / "02_main_participation_shifts_by_segment_bars.png"
        )
    )

    return plot_paths, source_path


def generate_class_plots():
    df, source_path = load_csv(
        "parshift_by_class_*.csv",
        ["shift_class", "total_frequency", "global_probability"]
    )

    df = normalize_numeric(df, ["total_frequency", "global_probability"])

    df["shift_class"] = pd.Categorical(
        df["shift_class"],
        categories=CLASS_ORDER,
        ordered=True
    )

    df = df.sort_values("shift_class")

    plot_paths = []

    plot_paths.append(
        save_bar_plot(
            df,
            "shift_class",
            "total_frequency",
            "Frequência global por classe de Participation Shift",
            "Classe de Participation Shift",
            "Frequência total",
            CLASS_DIR / "01_global_frequency_by_shift_class.png"
        )
    )

    plot_paths.append(
        save_bar_plot(
            df,
            "shift_class",
            "global_probability",
            "Probabilidade global por classe de Participation Shift",
            "Classe de Participation Shift",
            "Probabilidade global",
            CLASS_DIR / "02_global_probability_by_shift_class.png"
        )
    )

    plot_paths.append(
        save_treemap(
            df.rename(columns={"shift_class": "class_label"}),
            "class_label",
            "total_frequency",
            "Distribuição global por classe de Participation Shift",
            TREEMAP_DIR / "02_global_shift_classes_treemap.png"
        )
    )

    return plot_paths, source_path


def generate_class_by_legislature_plots():
    df, source_path = load_csv(
        "parshift_class_by_legislature_*.csv",
        ["legislature", "shift_class", "total_frequency", "global_probability"]
    )

    df = normalize_numeric(df, ["legislature", "total_frequency", "global_probability"])
    df["legislature"] = df["legislature"].astype(int)

    df["shift_class"] = pd.Categorical(
        df["shift_class"],
        categories=CLASS_ORDER,
        ordered=True
    )

    plot_paths = []

    plot_paths.append(
        save_grouped_bar_plot(
            df,
            "legislature",
            "shift_class",
            "global_probability",
            "Classes de Participation Shift por legislatura",
            "Legislatura",
            "Probabilidade global",
            CLASS_DIR / "03_shift_classes_by_legislature.png"
        )
    )

    return plot_paths, source_path


def write_report(plot_paths, source_paths):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    report_path = REPORTS_DIR / f"parshift_final_plots_report_{timestamp}.txt"

    lines = []
    lines.append("PARSHIFT FINAL PLOTS REPORT")
    lines.append("Dataset: final_dataset")
    lines.append(f"Generated at: {timestamp}")
    lines.append("")
    lines.append("Sources:")

    for source in sorted(set(str(path) for path in source_paths)):
        lines.append(f" - {source}")

    lines.append("")
    lines.append("Generated plots:")

    for path in plot_paths:
        lines.append(f" - {path}")

    report_path.write_text("\n".join(lines), encoding="utf-8")

    return report_path


def main():
    ensure_dirs()

    all_plot_paths = []
    source_paths = []

    generators = [
        generate_global_plots,
        generate_legislature_plots,
        generate_segment_plots,
        generate_class_plots,
        generate_class_by_legislature_plots
    ]

    for generator in generators:
        try:
            plot_paths, source_path = generator()
            all_plot_paths.extend(plot_paths)
            source_paths.append(source_path)
        except Exception as e:
            print(f"ERRO em {generator.__name__}: {e}")

    report_path = write_report(all_plot_paths, source_paths)

    print("")
    print(f"Gráficos finais criados em: {FINAL_PLOTS_DIR}")
    print(f"Relatório criado: {report_path}")
    print("")

    for path in all_plot_paths:
        print(f" - {path}")


if __name__ == "__main__":
    main()