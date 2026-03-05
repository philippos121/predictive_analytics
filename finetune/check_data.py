"""Quick sanity check of the classification dataset."""
from pathlib import Path
from collections import Counter
from datasets import load_from_disk

ds_path = Path(__file__).resolve().parent.parent / "data" / "prepared_dataset"
ds = load_from_disk(str(ds_path))
print(ds)
print("---")

for split in ["train", "validation"]:
    labels = ds[split]["label"]
    counts = Counter(labels)
    print(f"{split}: {len(labels)} samples | label distribution: {dict(counts)}")

print("---")
print("First 3 training examples:\n")
for i in range(min(3, len(ds["train"]))):
    row = ds["train"][i]
    print(f"[{i}] Label: {row['label']} | Text: {row['text'][:300]}...")
    print()
