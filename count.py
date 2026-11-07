import os
import re
import numpy as np
from collections import defaultdict, OrderedDict

# Map OM directories to names (added OM12)
om_names = {
    "OM1":  "Circular Walking",
    "OM2":  "Straight Walking",
    "OM3":  "Lateral Step",
    "OM4":  "Single Leg Stand",
    "OM5":  "Calf Raise",
    "OM6":  "Squat Rep",
    "OM7":  "Forward Lunge",
    "OM8":  "Leg Kick",
    "OM9":  "Push \\& Pull",
    "OM10": "Throwing Ball",
    "OM11": "Full-Body Stretches",
}

def extract_om_index(om_name):
    """Return integer index from strings like 'OM12'."""
    m = re.search(r'(\d+)', om_name)
    return int(m.group(1)) if m else float('inf')

def count_om_frames(root_dir):
    """
    Walk subject folders and count frames in Pressure.npy inside each OM dir.
    Returns a dict mapping OM name -> total frames (int).
    """
    counts = defaultdict(int)

    for subject in sorted(os.listdir(root_dir)):
        subject_path = os.path.join(root_dir, subject)
        if not os.path.isdir(subject_path):
            continue

        if 'Subject1' in subject_path:
            continue
        for item in sorted(os.listdir(subject_path)):
            if not item.startswith("OM"):
                continue
            if item.startswith('OM12'):
                continue
            om_dir = os.path.join(subject_path, item)
            if not os.path.isdir(om_dir):
                continue

            npy_path = os.path.join(om_dir, "Pressure.npy")
            if os.path.isfile(npy_path):
                try:
                    arr = np.load(npy_path)
                    counts[item] += len(arr)
                except Exception as e:
                    print(f"Warning: failed loading {npy_path}: {e}")

    return dict(counts)

def generate_latex_table(om_counts, om_names, include_missing=True, outfile=None):
    """
    om_counts: dict like {'OM1': 4698, 'OM2': 2359, ...}
    om_names: dict mapping OM1 -> Activity name
    include_missing: if True, include OM keys from om_names even if count==0
    outfile: if given, save LaTeX to this path
    """
    # Build the full set of OM keys to display
    keys = set(om_counts.keys())
    if include_missing:
        keys |= set(om_names.keys())

    # Sort keys numerically by their OM index
    sorted_keys = sorted(keys, key=extract_om_index)

    total = sum(om_counts.get(k, 0) for k in sorted_keys)

    lines = []
    lines.append("\\section{Ordinary Movements Data}")
    lines.append("\\begin{table}[h!]")
    lines.append("\\small")
    lines.append("\\centering")
    lines.append("\\begin{tabular}{r|l|r}")
    lines.append("\\hline")
    lines.append("\\textbf{\\#} & \\textbf{Activity} & \\textbf{\\# of Frames} \\\\")
    lines.append("\\hline")

    for idx, om in enumerate(sorted_keys, start=1):
        name = om_names.get(om, om)  # fallback to OM name if missing mapping
        count = om_counts.get(om, 0)
        lines.append(f"{idx} & {name} & {count} \\\\")

    lines.append("\\hline")
    lines.append(f"\\multicolumn{{2}}{{c|}}{{\\textbf{{Total}}}} & \\textbf{{{total}}} \\\\")
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\vspace{0.3cm}")
    lines.append("\\caption{Summary of the collected ordinary movement motions with frame counts for each action set.}")
    lines.append("\\label{{tab:om_summary}}")
    lines.append("\\end{table}")

    latex = "\n".join(lines)

    if outfile:
        with open(outfile, "w", encoding="utf-8") as f:
            f.write(latex)
        print(f"Wrote LaTeX table to: {outfile}")

    return latex

# --------------------
# Example usage:
# --------------------
if __name__ == "__main__":
    from data_paths import COMPLETE
    root = str(COMPLETE)   # <--- set this to your top-level directory
    om_counts = count_om_frames(root)
    latex_table = generate_latex_table(om_counts, om_names, include_missing=True, outfile="om_table.tex")
    print(latex_table)

