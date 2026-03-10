import json
from pathlib import Path


NOTEBOOKS = {
    "notebooks/dcgan/pt/notebook-dc.ipynb": "train_py/train_dcgan_from_nb.py",
    "notebooks/progan/notebook-pro.ipynb": "train_py/train_progan_from_nb.py",
    "notebooks/stylegan/v2/notebook-style.ipynb": "train_py/train_stylegan2_from_nb.py",
    "notebooks/wgan-gp/notebook-w.ipynb": "train_py/train_wgan_gp_from_nb.py",
    "notebooks/vit/notebook-vit.ipynb": "train_py/train_vit_from_nb.py",
}


def keep_line(line: str) -> bool:
    s = line.strip()
    if not s:
        return True
    if s.startswith("!"):
        return False
    if s.startswith("%"):
        return False
    if "google.colab" in s:
        return False
    if "drive.mount(" in s:
        return False
    return True


def convert_notebook(nb_path: Path, out_path: Path) -> None:
    data = json.loads(nb_path.read_text(encoding="utf-8"))
    cells = data.get("cells", [])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="\n") as f:
        f.write("# Auto-generated from notebook.\n")
        f.write("# Source: " + str(nb_path.as_posix()) + "\n")
        f.write("# Note: Review dataset/save paths before running.\n\n")
        for i, cell in enumerate(cells):
            if cell.get("cell_type") != "code":
                continue
            src = cell.get("source", [])
            kept = [ln for ln in src if keep_line(ln)]
            if not kept:
                continue
            f.write(f"# ===== Cell {i} =====\n")
            for ln in kept:
                f.write(ln)
            if kept and not kept[-1].endswith("\n"):
                f.write("\n")
            f.write("\n")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    for src, dst in NOTEBOOKS.items():
        convert_notebook(root / src, root / dst)
        print(f"Converted: {src} -> {dst}")


if __name__ == "__main__":
    main()

