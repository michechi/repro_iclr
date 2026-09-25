import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

df_ICL_512_3 = pd.read_csv()

# Sample data (replace with your actual data)
prompts = ["No", "Naive", "full-no-time (rnd)", "compact-no-time (rnd)", "compact", "compact-narrative", "full"]
temporal_order = [1, 2, 3, 4, 5, 6, 7]

# Truncation percentages (0-1)
complexity = {
    "ClinicalBERT (512)": [0.2092, 0.0, 0.3495, 0.0511, 0.0602, 0.0695, 0.3727],
    "MedBERT (512)":      [0.2432, 0.0, 0.3895, 0.0774, 0.0894, 0.0869, 0.4114],
    "ModernBERT (512)":  [0.1577, 0.0, 0.3045, 0.0392, 0.0465, 0.0495, 0.3286],
    "ModernBERT (1024)":  [0.0128, 0.0, 0.052, 0.0014, 0.0017, 0.0018, 0.0581],
    "ModernBERT (2048)":  [0.0001, 0.0, 0.0010, 0.0, 0.0, 0.0, 0.0011],
    "LLaMA-7B (512)":    [0.1653, 0.0, 0.3019, 0.0384, 0.0451, 0.0483, 0.3254],
    "LLaMA-7B (1024)":    [0.0142, 0.0, 0.0535, 0.0014, 0.0016, 0.0018, 0.0597],
    "LLaMA-7B (2048)":    [0.0001, 0.0, 0.0010, 0.0, 0.0, 0.00001, 0.0012]
}

# Create DataFrame
df = pd.DataFrame({
    "Prompt": prompts * 8,
    "TemporalOrder": temporal_order * 8,
    "Complexity": sum(complexity.values(), []),
    "LLM": [llm for llm in complexity.keys() for _ in range(len(prompts))]
})

# Set up ggplot-like bw style
plt.style.use('ggplot')  # Base ggplot style
plt.rcParams.update({
    'axes.facecolor': 'white',
    'grid.color': '0.85',  # Lighter grid lines
    'grid.linestyle': '-',
    'axes.edgecolor': '0.3',
    'axes.linewidth': 0.8,
})

# Create figure
fig, ax = plt.subplots(figsize=(12, 7))

# Pastel colors (manually defined to match ggplot's bw theme)
colors = {
    "MedBERT (512)":      '#FCAC9D',  # Light red
    "ClinicalBERT (512)": '#B29DFC',  # Light blue
    "ModernBERT (512)":   '#9DFCD7',
    "ModernBERT (1024)":  '#A7A58E',
    "ModernBERT (2048)":  '#7D5C55',
    "LLaMA-7B (512)":     '#5E557D',  # Light green
    "LLaMA-7B (1024)":    '#557D6D',
    "LLaMA-7B (2048)":    '#FCF39D'   # Light purple
}

# Plot each LLM
for llm, group in df.groupby("LLM"):
    # Line connecting the points
    ax.plot(
        group["TemporalOrder"],
        group["Complexity"],
        color=colors[llm],
        linewidth=0.7,
        alpha=0.7
    )
    ax.scatter(
        x=group["TemporalOrder"],
        y=group["Complexity"],
        color=colors[llm],
        s=40,
        edgecolor='0.0',  # Darker border
        linewidth=0.0,
        alpha=0.6,
        label=llm
    )

# Add critical threshold line
ax.axhline(y=0.5, color='#FF5D5C', linestyle='--', alpha=0.5, linewidth=1)
ax.text(x=max(temporal_order)+0.1, y=0.5, s=" Critical (50%)", va='center', color='#FF5D5C')

# legend inside the plot
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))  # avoid duplicates
ax.legend(by_label.values(), by_label.keys(),
          title="LLM (Context Window)",
          frameon=True, framealpha=0.8, edgecolor='0.8',
          loc='upper left')  # inside, top right

# # Title and legend
ax.set_title("Prompt Analysis: Temporal Order vs. Truncation by LLM", pad=20)


# Labels and title
ax.set_xlabel("Temporal Order of Prompts", fontsize=12)
ax.set_ylabel("Context-Loss-Index (CLI)", fontsize=12)
ax.set_xticks(temporal_order)
ax.set_xticklabels(prompts)
ax.set_yticks(np.arange(0, 0.51, 0.1))
ax.set_yticklabels([f"{int(x*100)}%" for x in np.arange(0, 0.051, 0.1)])
ax.set_ylim(0, 0.50)


# ax.legend(title="LLM (Context Window)", frameon=True, framealpha=1, edgecolor='0.8', loc='center left', bbox_to_anchor=(1, 0.5))

# # Add grid
# ax.grid(True, linestyle='-', alpha=0.3)

# # Tight layout
plt.tight_layout(rect=[0, 0, 0.85, 1])
# plt.show()