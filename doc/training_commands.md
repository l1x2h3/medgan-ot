# MedGAN Training Code, Dataset Paths, and Commands

## 1) Where training code is

This repo currently has no standalone `train_*.py` scripts.
Training entry points are notebooks under `notebooks/`:

- `notebooks/dcgan/pt/notebook-dc.ipynb`
- `notebooks/progan/notebook-pro.ipynb`
- `notebooks/stylegan/v2/notebook-style.ipynb` (StyleGAN2)
- `notebooks/wgan-gp/notebook-w.ipynb`
- `notebooks/vit/notebook-vit.ipynb`

`medgan/*.py` files are mainly model definitions and inference helpers.

## 2) Do dataset paths need changes?

Yes.

Most notebooks are written for Colab (`/content/...`) or a local author path (`C:\Users\...`).
Your current dataset appears to be:

- `data/archive/Training`
- `data/archive/Testing`

Recommended:

- GAN notebooks (DCGAN / ProGAN / StyleGAN2 / WGAN): use `data/archive/Training`
- ViT notebook: use both `Training` and `Testing`

## 3) Environment setup and Jupyter launch

From project root:

```powershell
pip install -r requirements.txt
pip install jupyter
jupyter lab
```

## 4) Per-model commands and variables to edit

Note:
- Preferred workflow: open notebook and run cells interactively.
- Optional: run full notebook with `nbconvert` after path edits.

### 4.1 DCGAN

Notebook:

- `notebooks/dcgan/pt/notebook-dc.ipynb`

Edit:

- Replace or skip unzip cell: `!unzip /content/archive_5.zip -d /content/images`
- Change:
  - `dataset = datasets.ImageFolder('/content/images', transform=transform)`
  - to:
  - `dataset = datasets.ImageFolder('data/archive/Training', transform=transform)`

Run:

```powershell
jupyter lab notebooks/dcgan/pt/notebook-dc.ipynb
```

Optional batch execute:

```powershell
jupyter nbconvert --to notebook --execute --inplace notebooks/dcgan/pt/notebook-dc.ipynb
```

### 4.2 ProGAN

Notebook:

- `notebooks/progan/notebook-pro.ipynb`

Edit:

- Change:
  - `DATASET = r"C:\Users\mzlwm\OneDrive\Desktop\MEDGAN\dataset"`
  - to:
  - `DATASET = r"data/archive/Training"`

Run:

```powershell
jupyter lab notebooks/progan/notebook-pro.ipynb
```

Optional batch execute:

```powershell
jupyter nbconvert --to notebook --execute --inplace notebooks/progan/notebook-pro.ipynb
```

### 4.3 StyleGAN2

Notebook:

- `notebooks/stylegan/v2/notebook-style.ipynb`

Edit:

- Comment out or remove Colab cells:
  - `drive.mount('/content/drive')`
  - zip extraction from `/content/drive/...`
- Change:
  - `DATASET = r"dataset"`
  - to:
  - `DATASET = r"data/archive/Training"`
- Replace all save paths under `/content/drive/...` with local paths, for example:
  - `save_dir="models/stylegan2-local"`
  - `base_dir="outputs/stylegan2/saved_examples"`

Run:

```powershell
jupyter lab notebooks/stylegan/v2/notebook-style.ipynb
```

Optional batch execute:

```powershell
jupyter nbconvert --to notebook --execute --inplace notebooks/stylegan/v2/notebook-style.ipynb
```

### 4.4 WGAN-GP

Notebook:

- `notebooks/wgan-gp/notebook-w.ipynb`

Edit:

- Comment out Colab cells (`drive.mount`, `!unzip ...`).
- Change:
  - `dataroot = "/content/extracted_files"`
  - to:
  - `dataroot = "data/archive/Training"`
- Replace all `/content/drive/MyDrive/WGAN New Dataset/...` save/load paths with local output paths, for example:
  - `outputs/wgan/checkpoint_epoch_X.pt`
  - `outputs/wgan/final_checkpoint.pt`
  - `outputs/wgan/mu_real.npy`
  - `outputs/wgan/sigma_real.npy`

Run:

```powershell
jupyter lab notebooks/wgan-gp/notebook-w.ipynb
```

Optional batch execute:

```powershell
jupyter nbconvert --to notebook --execute --inplace notebooks/wgan-gp/notebook-w.ipynb
```

### 4.5 ViT classifier

Notebook:

- `notebooks/vit/notebook-vit.ipynb`

Edit:

- Comment out Colab mount/unzip cells.
- Change train/test dirs to local:
  - `train_dir = "data/archive/Training"`
  - `test_dir = "data/archive/Testing"`
- `save_model(... target_dir="models", ...)` can stay as-is (saves under repo `models/`).

Run:

```powershell
jupyter lab notebooks/vit/notebook-vit.ipynb
```

Optional batch execute:

```powershell
jupyter nbconvert --to notebook --execute --inplace notebooks/vit/notebook-vit.ipynb
```

## 5) Open all training notebooks at once (optional)

```powershell
jupyter lab `
  notebooks/dcgan/pt/notebook-dc.ipynb `
  notebooks/progan/notebook-pro.ipynb `
  notebooks/stylegan/v2/notebook-style.ipynb `
  notebooks/wgan-gp/notebook-w.ipynb `
  notebooks/vit/notebook-vit.ipynb
```

## 6) Important note for class-specific GAN training

`ImageFolder` expects class subfolders under a root directory.

If you want one GAN per tumor class, prepare separate roots like:

- `data/glioma_train/glioma/*.jpg`
- `data/meningioma_train/meningioma/*.jpg`
- `data/pituitary_train/pituitary/*.jpg`

Then point each notebook root (`DATASET` / `dataroot`) to the corresponding parent folder.

