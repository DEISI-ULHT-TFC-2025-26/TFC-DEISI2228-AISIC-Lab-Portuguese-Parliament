import json
from pathlib import Path
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
METRICS_DIR = BASE_DIR / "data/processed/metrics_evaluation_3"
INPUT_FILE = METRICS_DIR / "metrics_2026-04-01_14-46-11.json"
OUTPUT_FILE = METRICS_DIR / "global_bar_charts.png"


def load_json_file(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def format_number(value):
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return f"{value:.2f}"
    return str(value)


def truncate_label(text, max_len=35):
    text = str(text)
    if len(text) <= max_len:
        return text
    return text[:max_len - 3] + "..."


def plot_horizontal_bar_chart(ax, title, data_dict, max_items=None):
    if not data_dict:
        ax.text(0.5, 0.5, f"{title}\n(no data)", ha='center', va='center')
        ax.axis('off')
        return

    items = list(data_dict.items())
    # Ordenar do maior para o menor
    items.sort(key=lambda x: x[1], reverse=True)

    if max_items is not None:
        items = items[:max_items]

    total = sum(v for _, v in items)

    # Inverter a ordem para o matplotlib desenhar o maior no topo
    items.reverse()

    labels = [truncate_label(k) for k, _ in items]
    values = [v for _, v in items]

    bars = ax.barh(labels, values, color='steelblue', edgecolor='black', alpha=0.8)

    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Count', fontsize=11)

    # Adicionar os valores e percentagens à frente de cada barra
    for bar, val in zip(bars, values):
        pct = (val / total * 100) if total else 0.0
        text = f" {format_number(val)} ({pct:.2f}%)"
        ax.text(val, bar.get_y() + bar.get_height()/2, text,
                va='center', ha='left', fontsize=10, color='darkblue')

    # Adicionar uma margem extra no eixo X para o texto não ser cortado
    max_val = max(values) if values else 0
    ax.set_xlim(0, max_val * 1.25)


def save_bar_charts(data):
    top_speakers = data.get("top_speakers", {})
    party_metrics = data.get("party_metrics", {})
    party_full = party_metrics.get("full", {})

    # Criar uma figura com 2 subgráficos (1 linha, 2 colunas se quisermos lado a lado, ou 2 linhas 1 coluna)
    # Como são gráficos de barras horizontais com muitos labels, 2 linhas fica melhor.
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 16))

    if top_speakers:
        plot_horizontal_bar_chart(ax1, "TOP SPEAKERS (Top 20)", top_speakers, max_items=20)
    else:
        ax1.axis('off')

    if party_full:
        plot_horizontal_bar_chart(ax2, "PARTY DISTRIBUTION", party_full, max_items=None)
    else:
        ax2.axis('off')

    plt.tight_layout(pad=4.0)
    plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches='tight')
    print(f"Grafico de barras guardado em: {OUTPUT_FILE}")

    try:
        plt.show()
    except Exception:
        pass


def main():
    try:
        data = load_json_file(INPUT_FILE)
        save_bar_charts(data)
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()

