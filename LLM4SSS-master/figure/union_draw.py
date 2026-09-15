import os
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

selected_models = ["AutoTimes", "GPT4SSS", "S2IPLLM", "UniTime", "TimeMixer"]
selected_dataset = "sald"
max_points = 100

plt.figure(figsize=(10, 7))

colors = sns.color_palette("tab10", n_colors=len(selected_models))

for idx, model in enumerate(selected_models):
    file_path = f"figure/error/{model}_{selected_dataset}.txt"
    data = []
    try:
        with open(file_path, "r") as file:
            for line_num, line in enumerate(file, start=1):
                try:
                    value = float(line.strip())
                    data.append(value)
                except ValueError:
                    print(f"[{model}] Line {line_num}: '{line.strip()}' cannot be converted to float.")
    except FileNotFoundError:
        print(f"[{model}] Error: File '{file_path}' not found.")
        continue

    if len(data) > max_points:
        print(f"[{model}] Warning: More than {max_points} points, truncating.")
        data = data[:max_points]
    elif len(data) < max_points:
        print(f"[{model}] Warning: Only {len(data)} data points found.")

    plt.plot(
        range(1, len(data) + 1),
        data,
        label=model,
        color=colors[idx],
        linewidth=2,
        marker='o',
        markersize=3
    )

plt.title(f"Training Loss Comparison on {selected_dataset}", fontsize=16, fontweight='bold')
plt.xlabel("Epoch", fontsize=14)
plt.ylabel("Loss", fontsize=14)
plt.xticks(range(0, 101, 5))
plt.xlim(1, max_points)
plt.grid(True, linestyle='--', linewidth=0.6, alpha=0.7)
plt.legend(title="Model", fontsize=12)
plt.tight_layout()

save_path = f"figure/figure/comparison_{selected_dataset}.png"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
plt.savefig(save_path, dpi=300)
plt.close()
print(f"Comparison figure saved to {save_path}")
