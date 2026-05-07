import json
import matplotlib.pyplot as plt
from pathlib import Path

def plot_boxplots(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    unique_speakers = data.get('boxplot_metrics', {}).get('unique_speakers', {})

    # Extract the pre-calculated metrics for all debates
    all_debates = unique_speakers.get('all_debates', {})

    if not all_debates:
        print("No overall metrics found for plotting.")
        return

    # Prepare data for plotting just the overall stats
    stats = [{
        'label': 'All Debates',
        'mean': all_debates.get('mean'),
        'med': all_debates.get('median'),
        'q1': all_debates.get('q1'),
        'q3': all_debates.get('q3'),
        'whislo': all_debates.get('whisker_low'),
        'whishi': all_debates.get('whisker_high'),
        'fliers': [all_debates.get('min'), all_debates.get('max')] # Represent min/max as outliers if they are outside whiskers
    }]

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 4))

    # vert=False makes the boxplot horizontal
    ax.bxp(stats, showmeans=True, vert=False)

    ax.set_title('Overall Boxplot for Unique Speakers', fontsize=15)
    ax.set_xlabel('Number of Unique Speakers', fontsize=12)
    
    # Hide the y-axis ticks since there's only one box
    ax.set_yticks([])

    # Adjust the y-axis limits appropriately for a single boxplot
    ax.set_ylim(0.5, 1.5)

    plt.tight_layout()

    # Save the plot
    output_path = Path(json_path).parent / 'overall_boxplot.png'
    plt.savefig(output_path, dpi=300)
    print(f"Chart saved successfully to: {output_path}")

    # Show the plot
    plt.show()

if __name__ == '__main__':
    json_file = 'src/data/processed/metrics_evaluation_3/boxplot_2026-04-01_14-46-11.json'
    plot_boxplots(json_file)
