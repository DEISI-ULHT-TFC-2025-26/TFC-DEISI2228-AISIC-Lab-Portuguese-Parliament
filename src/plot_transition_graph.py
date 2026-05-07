import json
from pathlib import Path
import matplotlib.pyplot as plt
import networkx as nx

BASE_DIR = Path(__file__).resolve().parent
METRICS_DIR = BASE_DIR / "data/processed/metrics_evaluation_3"
INPUT_FILE = METRICS_DIR / "metrics_2026-04-01_14-46-11.json"
OUTPUT_FILE = METRICS_DIR / "transition_graph.png"


def load_json_file(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_transition_key(key: str):
    if " -> " not in key:
        return key, "UNKNOWN"
    src, tgt = key.split(" -> ", 1)
    return src.strip(), tgt.strip()


def build_transition_graph_visual(transitions_dict, top_n=120):
    items = sorted(transitions_dict.items(), key=lambda x: x[1], reverse=True)

    if top_n is not None:
        items = items[:top_n]

    if not items:
        print("No transition data to plot.")
        return

    # Calculate max and total AFTER slicing, matching the original script's logic
    max_count = max(count for _, count in items)
    total = sum(count for _, count in items)
    
    # Filter to only show records with more than 1% participation
    items = [(key, count) for key, count in items if (count / total * 100) > 1.0]

    if not items:
        print("Nenhuma transicao com mais de 1% de participacao para mostrar.")
        return

    # Calculate figure height based on number of items (approx 0.4 inches per row)
    fig_height = max(5.0, len(items) * 0.4 + 2)
    fig, ax = plt.subplots(figsize=(14.0, fig_height))
    ax.axis('off')
    
    y_start = len(items)
    
    # Headers
    ax.text(0.00, y_start + 1, 'FROM', fontweight='bold', ha='left', fontsize=11, color='darkblue')
    ax.text(0.35, y_start + 1, 'FLOW', fontweight='bold', ha='center', fontsize=11, color='darkblue')
    ax.text(0.50, y_start + 1, 'TO', fontweight='bold', ha='left', fontsize=11, color='darkblue')
    ax.text(0.85, y_start + 1, 'COUNT', fontweight='bold', ha='right', fontsize=11, color='darkblue')
    ax.text(0.95, y_start + 1, '%', fontweight='bold', ha='right', fontsize=11, color='darkblue')
    
    # Header line
    ax.plot([0, 1], [y_start + 0.5, y_start + 0.5], color='black', lw=2)
    
    for i, (key, count) in enumerate(items):
        y = y_start - i
        src, tgt = parse_transition_key(key)
        
        # Crop text to avoid overlap
        src_label = src[:36] + '...' if len(src) > 36 else src
        tgt_label = tgt[:36] + '...' if len(tgt) > 36 else tgt
        
        ax.text(0.00, y, src_label, va='center', fontsize=10)
        
        # Flow (Bar + Arrow pointing right)
        # Scaled to a max length of 0.20 in axes coordinates
        min_len = 0.01 
        bar_len = max(min_len, (count / max_count) * 0.20) if max_count else min_len
        start_x = 0.25
        end_x = start_x + bar_len
        
        ax.annotate('', xy=(end_x, y), xytext=(start_x, y),
                    arrowprops=dict(arrowstyle="->", color="steelblue", lw=2, mutation_scale=15))
        
        ax.text(0.50, y, tgt_label, va='center', fontsize=10)
        ax.text(0.85, y, str(count), va='center', ha='right', fontsize=10)
        
        pct = (count / total * 100) if total else 0
        ax.text(0.95, y, f"{pct:.2f}%", va='center', ha='right', fontsize=10)
        
        # Thin divider between rows
        ax.plot([0, 1], [y - 0.5, y - 0.5], color='gray', lw=0.5, alpha=0.3)

    plt.title(f"Transition Graph (Most Frequent Transitions)", fontsize=16, fontweight='bold', pad=20)
    
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.5, y_start + 2)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches='tight')
    print(f"Chart saved successfully to: {OUTPUT_FILE}")
    
    try:
        plt.show()
    except Exception:
        pass


def main():
    try:
        data = load_json_file(INPUT_FILE)
        transitions = data.get("transitions", {}).get("global", {})

        if not transitions:
            raise ValueError("No transitions.global found in input JSON")

        build_transition_graph_visual(transitions_dict=transitions, top_n=120)
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()

