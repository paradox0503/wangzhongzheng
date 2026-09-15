import os
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

selected_model = "TimeMixer"
selected_dataset = "astro"

file_path = f"figure/error/{selected_model}_{selected_dataset}.txt"
save_path = f"figure/figure/{selected_model}_{selected_dataset}.png"

data = []
try:
    with open(file_path, "r") as file:
        for idx, line in enumerate(file, start=1):
            try:
                value = float(line.strip())
                data.append(value)
            except ValueError:
                print(f"Line {idx}: '{line.strip()}' cannot be converted to float.")
except FileNotFoundError:
    print(f"Error: File '{file_path}' not found.")

if len(data) > 100:
    print(f"Warning: Data has more than 100 points ({len(data)}), truncating to 100.")
    data = data[:100]
elif len(data) < 100:
    print(f"Warning: Only {len(data)} data points found, x-axis will still show 1–100.")

os.makedirs(os.path.dirname(save_path), exist_ok=True)

plt.figure(figsize=(8, 6))
plt.plot(
    range(1, len(data) + 1),
    data,
    label="Loss",
    color="#1f77b4",
    linewidth=2.5,
    marker='o',
    markersize=4
)

plt.title(f"Training Loss: {selected_model} on {selected_dataset}", fontsize=16, fontweight='bold')
plt.xlabel("Epoch", fontsize=14)
plt.ylabel("Loss", fontsize=14)
plt.xticks(range(0, 101, 5), fontsize=10)
plt.yticks(fontsize=10)

plt.xlim(1, 100)
plt.grid(True, linestyle='--', linewidth=0.6, alpha=0.7)
plt.legend(fontsize=12, loc='best')
plt.tight_layout()
plt.savefig(save_path, dpi=300)
plt.close()
print(f"Figure saved to {save_path}")