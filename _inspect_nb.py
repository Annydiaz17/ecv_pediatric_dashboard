"""Inspeccionar el notebook SEMMA para entender el pipeline."""
import json, sys
sys.stdout.reconfigure(encoding='utf-8')

with open("SEMMA_dataset_timbiqui_2024_estandarizado.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

cells = nb["cells"]
print(f"Total cells: {len(cells)}")
print("=" * 80)

for i, c in enumerate(cells):
    ctype = c["cell_type"]
    src = "".join(c["source"])
    preview = src[:400].replace("\n", " | ")
    print(f"\n--- Cell {i} ({ctype}) ---")
    print(preview)
