"""
Build an invariance summary table from the log files in this directory.

Usage:
    python make_table.py [--dir <path>] [--out <file.tex>]

Each file is named  drunet_{model}_{distortion}_{level}.txt
Models     : nat, vl
Distortions: blur, contrast, hue, rot, scale

Layout (transposed):
  One sub-table per distortion.
  Columns = non-zero distortion levels.
  Rows    = Δ(nat) | Δ(vl) | Δ(vl)−Δ(nat)

Δ is always relative to the undistorted baseline (rot_0).
Δ(vl)−Δ(nat) > 0 means VL degrades less (more robust) than Nat.
Level 0 is excluded from all tables.
"""

import re
import argparse
from pathlib import Path

# ── parsing ──────────────────────────────────────────────────────────────────

FILENAME_RE = re.compile(
    r"drunet_(?P<model>\w+)_(?P<distortion>\w+)_(?P<level>[\d.]+)\.txt"
)

PSNR_RE = re.compile(r"PSNR\s+:\s+([\d.inf]+)\s+dB")


def parse_file(path: Path) -> float:
    m = PSNR_RE.search(path.read_text())
    if not m:
        return float("nan")
    try:
        return float(m.group(1))
    except ValueError:
        return float("nan")


def load_all(directory: Path) -> dict:
    """Return data[model][distortion][level] = psnr (float)."""
    data: dict = {}
    for path in sorted(directory.glob("*.txt")):
        m = FILENAME_RE.match(path.name)
        if not m:
            continue
        model      = m.group("model")
        distortion = m.group("distortion")
        level      = float(m.group("level"))
        data.setdefault(model, {}).setdefault(distortion, {})[level] = parse_file(path)
    return data


def apply_fixes(data: dict) -> None:
    """Replace invalid contrast_0 with the undistorted baseline (rot_0)."""
    for model, distortions in data.items():
        baseline = distortions.get("rot", {}).get(0.0)
        if baseline is None:
            continue
        if "contrast" in distortions and 0.0 in distortions["contrast"]:
            distortions["contrast"][0.0] = baseline


# ── display helpers ───────────────────────────────────────────────────────────

MODELS = ["nat", "vl"]

DISTORTION_META = {
    "blur":     ("Blur",     "Gaussian $\\sigma$",   ""),
    "contrast": ("Contrast", "S-curve $k$",          ""),
    "hue":      ("Hue",      "shift",                "°"),
    "rot":      ("Rotation", "",                     "°"),
    "scale":    ("Scale",    "downsample",           "×"),
}

ROW_LABELS = [
    r"$\Delta$(Nat)",
    r"$\Delta$(VL)",
    r"$\Delta$(VL)$-$$\Delta$(Nat)",
]

ROW_LABELS_TEX = [
    r"$\Delta(\mathrm{Nat})$",
    r"$\Delta(\mathrm{VL})$",
    r"$\Delta(\mathrm{VL}) - \Delta(\mathrm{Nat})$",
]


def fmt_delta(val: float, sign: bool = True) -> str:
    if val != val:
        return "—"
    if sign:
        return f"{val:+.2f}"
    return f"{val:.2f}"


def build_distortion_data(distortion: str, data: dict) -> tuple[list, list[list]]:
    """
    Returns (levels, rows) where:
      levels   : sorted non-zero level values for this distortion
      rows     : 3 lists of floats — [Δ(nat)], [Δ(vl)], [Δ(vl)−Δ(nat)]
    """
    baselines = {
        model: data.get(model, {}).get("rot", {}).get(0.0, float("nan"))
        for model in MODELS
    }

    all_levels = sorted(
        {lvl for model in MODELS for lvl in data.get(model, {}).get(distortion, {})}
    )
    levels = [lvl for lvl in all_levels if lvl != 0.0]   # drop level 0

    delta_nat, delta_vl, diff = [], [], []
    for lvl in levels:
        v_nat = data.get("nat", {}).get(distortion, {}).get(lvl, float("nan"))
        v_vl  = data.get("vl",  {}).get(distortion, {}).get(lvl, float("nan"))
        dn = v_nat - baselines["nat"]
        dv = v_vl  - baselines["vl"]
        delta_nat.append(dn)
        delta_vl.append(dv)
        diff.append(dv - dn)

    return levels, [delta_nat, delta_vl, diff]


# ── terminal print ────────────────────────────────────────────────────────────

def print_tables(data: dict) -> None:
    for distortion in ["blur", "contrast", "hue", "rot", "scale"]:
        if not any(distortion in data.get(m, {}) for m in MODELS):
            continue

        short, param, unit = DISTORTION_META[distortion]
        title = f"{short}" + (f" ({param})" if param else "")
        levels, rows = build_distortion_data(distortion, data)

        # header
        col_w = 18
        lv_w  = 9
        header = f"{'':>{col_w}}" + "".join(
            f"  {lvl:g}{unit}".rjust(lv_w) for lvl in levels
        )
        print(f"\n── {title} {'─' * max(0, 70 - len(title))}")
        print(header)
        print("  " + "─" * (col_w + lv_w * len(levels) + 2 * len(levels) - 2))

        for label, row in zip(ROW_LABELS, rows):
            cells = "".join(fmt_delta(v).rjust(lv_w) for v in row)
            print(f"  {label:<{col_w}}{cells}")


# ── LaTeX export ──────────────────────────────────────────────────────────────

TEX_HEADER = r"""\documentclass{article}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{amsmath}
\usepackage{siunitx}
\usepackage{caption}
\begin{document}
"""

TEX_FOOTER = r"\end{document}"

TEX_CAP = {
    "blur":     r"Blur (Gaussian $\sigma$). Levels are $\sigma$ values.",
    "contrast": r"Contrast (S-curve $k$). Levels are $k$ values.",
    "hue":      r"Hue shift. Levels are rotation angles in degrees.",
    "rot":      r"Rotation. Levels are rotation angles in degrees.",
    "scale":    r"Scale (downsample factor).",
}


def make_tex_table(distortion: str, levels: list, rows: list[list], unit: str) -> str:
    n = len(levels)
    col_spec = "l" + " r" * n

    def escape(v: float) -> str:
        if v != v:
            return r"\text{---}"
        return f"{v:+.2f}"

    # Level header row
    level_cells = " & ".join(
        f"${lvl:g}\\,\\mathrm{{{unit}}}$" if unit else f"${lvl:g}$"
        for lvl in levels
    )

    tex_rows = []
    for label, row in zip(ROW_LABELS_TEX, rows):
        cells = " & ".join(f"${escape(v)}$" for v in row)
        tex_rows.append(f"    {label} & {cells} \\\\")

    lines = [
        r"\begin{table}[h]",
        r"  \centering",
        r"  \small",
        f"  \\begin{{tabular}}{{{col_spec}}}",
        r"    \toprule",
        f"    & {level_cells} \\\\",
        r"    \midrule",
    ]
    lines += tex_rows
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        f"  \\caption{{{TEX_CAP.get(distortion, distortion.capitalize())}}}",
        f"  \\label{{tab:inv_{distortion}}}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def export_tex(data: dict, out_path: Path) -> None:
    blocks = [TEX_HEADER]
    for distortion in ["blur", "contrast", "hue", "rot", "scale"]:
        if not any(distortion in data.get(m, {}) for m in MODELS):
            continue
        _, _, unit = DISTORTION_META[distortion]
        levels, rows = build_distortion_data(distortion, data)
        blocks.append(make_tex_table(distortion, levels, rows, unit))
        blocks.append("")
    blocks.append(TEX_FOOTER)
    out_path.write_text("\n".join(blocks))
    print(f"\n✓ LaTeX written to {out_path}")


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default=".", help="directory containing the .txt log files")
    parser.add_argument("--out", default="invariance_table.tex",
                        help="output .tex file (default: invariance_table.tex)")
    args = parser.parse_args()

    directory = Path(args.dir)
    data = load_all(directory)
    apply_fixes(data)

    print_tables(data)
    export_tex(data, directory / args.out)


if __name__ == "__main__":
    main()
