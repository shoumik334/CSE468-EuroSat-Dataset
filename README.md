# Land-Use & Land-Cover (LULC) Classification on EuroSAT and UC Merced

Two complementary Jupyter notebooks for **Land Use / Land Cover image classification** on
the **EuroSAT** (Sentinel-2 RGB, 10 classes) and **UC Merced Land Use** (21 classes) datasets.

| # | Notebook | What it does |
|---|----------|--------------|
| 1 | [`custom model on eurosat.ipynb`](./custom%20model%20on%20eurosat.ipynb) | Trains a single **custom Residual + Squeeze-and-Excitation CNN** (`CustomEuroSATCNN`) on EuroSAT end-to-end, with a **locked test phase** and full **XAI** (Grad-CAM + LIME). Kaggle / dual-T4 ready. |
| 2 | [`ViT_LULC_Classification 2 today.ipynb`](./ViT_LULC_Classification%202%20today.ipynb) | Benchmarks **6 architectures** (ViT-Base, ResNet-18, EfficientNet-B0, DenseNet-121, Simple CNN, Swin-Tiny) on **EuroSAT** and **UC Merced** and compares results in a single report. CPU-friendly (Mac M1 optimized). |

---

## Table of Contents

- [Repository layout](#repository-layout)
- [Datasets](#datasets)
- [Environment & dependencies](#environment--dependencies)
- [Notebook 1 — Custom CNN on EuroSAT](#notebook-1--custom-cnn-on-eurosat)
- [Notebook 2 — Multi-model LULC Benchmark (EuroSAT + UC Merced)](#notebook-2--multi-model-lulc-benchmark-eurosat--uc-merced)
- [Results](#results)
- [Outputs](#outputs)
- [Reproducibility](#reproducibility)
- [License](#license)

---

## Repository layout

```text
.
├── custom model on eurosat.ipynb      # Notebook 1 — Custom CNN + XAI
├── ViT_LULC_Classification 2 today.ipynb  # Notebook 2 — multi-model benchmark
├── outputs/                           # Auto-generated artefacts (Notebook 1)
│   ├── checkpoints/                   # best_<key>.pth, last_<key>.pth, *_weights.pth
│   ├── predictions/                   # per-image top-1/top-2 predictions
│   ├── test_metrics/                  # classification report + confusion matrix
│   ├── xai/
│   │   ├── gradcam/                   # Grad-CAM overlays per class
│   │   ├── lime/                      # LIME segmentation explanations
│   │   └── bundles/                   # self-contained .pth XAI bundle
│   ├── final_test/                    # final integration-test images
│   ├── samples_per_class/             # one sample grid per class
│   └── *.csv / *.json                 # split, label map, metrics
└── serve/                             # (optional) lightweight inference server
```

---

## Datasets

### EuroSAT (RGB, 10 classes)
- 27,000 Sentinel-2 patches (64×64 native, resized for training).
- Classes: `AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial,
  Pasture, PermanentCrop, Residential, River, SeaLake`.
- Both notebooks expect the standard **`train.csv` / `test.csv`** split files alongside
  the image folder. On Kaggle, attach the EuroSAT dataset and the notebook auto-detects it
  under `/kaggle/input/...`.

### UC Merced Land Use (21 classes, Notebook 2 only)
- 2,100 aerial images (256×256), 100 per class.
- 21 classes including `agricultural, airplane, baseballdiamond, beach, buildings,
  chaparral, denseresidential, forest, freeway, golfcourse, harbor, intersection,
  mediumresidential, mobilehomepark, overpass, parkinglot, river, runway,
  sparseresidential, storagetanks, tenniscourt`.
- Notebook 2 splits 70 / 15 / 15 stratified (train / val / test).

---

## Environment & dependencies

Both notebooks install everything they need in their first cell. To run locally:

```bash
# Core
pip install torch torchvision torchaudio
pip install timm pandas numpy pillow scikit-learn matplotlib seaborn tqdm

# Notebook 1 (XAI + multispectral I/O)
pip install grad-cam==1.5.4 lime==0.2.0.1 opencv-python-headless rasterio

# Notebook 2 (attribution methods)
pip install captum
```

| | Notebook 1 | Notebook 2 |
|---|---|---|
| Target hardware | CUDA GPU (Kaggle dual-T4 supported, single-GPU by default) | CPU (Mac M1/Intel) — also works on GPU |
| Image size | 224 × 224 | 128 × 128 |
| Mixed precision | ✅ `torch.amp` | ❌ |
| Reproducibility seed | 42 | 42 |

---

## Notebook 1 — Custom CNN on EuroSAT

[`custom model on eurosat.ipynb`](./custom%20model%20on%20eurosat.ipynb)

End-to-end pipeline for a **single custom architecture**, designed for Kaggle but portable.

### Pipeline

1. **Auto-detect dataset.** Robust scan of `/kaggle/input/*` for the EuroSAT RGB folder
   plus `train.csv` / `test.csv`. Falls back to CWD when run locally. Multispectral
   `EuroSATallBands` is explicitly skipped in favour of the JPG split.
2. **Stratified train / val split** from `train.csv` (deterministic seed = 42).
   The **test set is reserved**: it is never seen during training, validation, or
   model selection — eliminating the test-leakage bug present in the previous version.
3. **Custom architecture — `CustomEuroSATCNN`:**
   - Stem: `Conv3×3 → BN → SiLU` (64 ch)
   - 4 stages of **Residual blocks** with **Squeeze-and-Excitation** attention
     (`64 → 128 → 256 → 512`)
   - `AdaptiveAvgPool` + `Dropout(0.3)` + `Linear` head
   - ~11 M parameters
4. **Training:**
   - AdamW (`lr=3e-4`, `wd=1e-4`), label smoothing 0.05, gradient clipping 1.0.
   - `ReduceLROnPlateau` on val accuracy, **early stopping on val accuracy** only.
   - **AMP** (`torch.amp`) on a single primary GPU (`cuda:0`) — `nn.DataParallel` was
     deliberately removed because the combination of *DataParallel + AMP + custom
     SE/SiLU blocks* is a documented cause of `CUDA error: an illegal memory access`
     on dual-T4 Kaggle runners.
   - Per epoch we save `best_<key>.pth`, `last_<key>.pth`, and `<key>_weights.pth`.
     Each checkpoint contains: `epoch, model_state_dict, optimizer_state_dict,
     scheduler_state_dict, val_accuracy, model_name, class_to_idx, cfg`.
5. **Locked final test.** Loads `best_custom.pth` and evaluates `test.csv`
   **exactly once**, producing:
   - classification report
   - confusion matrix (PNG + CSV)
   - per-image predictions CSV (top-1, top-2, confidences)
6. **Explainability (XAI):**
   - **Grad-CAM** overlays for one image per class.
   - **LIME** segmentation explanations.
   - A self-contained **XAI bundle** (`*.pth`) with weights + class map +
     reshape kind, ready to re-explain new images without the notebook.
7. **Integration test.** Reloads the checkpoint from disk and renders Grad-CAM + LIME
   for one image per class to verify everything round-trips.

### Default config

```python
@dataclass
class CFG:
    image_size: int = 224
    batch_size: int = 128       # 96 on 1 GPU, 32 on CPU
    epochs:     int = 12
    lr:         float = 3e-4
    weight_decay: float = 1e-4
    label_smoothing: float = 0.05
    grad_clip:  float = 1.0
    early_stop_patience: int = 3
    plateau_patience:    int = 2
    plateau_factor:      float = 0.5
    amp:        bool = True
```

### Run it

```bash
jupyter lab "custom model on eurosat.ipynb"
# or, on Kaggle: attach the EuroSAT dataset, "Run All".
```

---

## Notebook 2 — Multi-model LULC Benchmark (EuroSAT + UC Merced)

[`ViT_LULC_Classification 2 today.ipynb`](./ViT_LULC_Classification%202%20today.ipynb)

A **single notebook** that trains, evaluates, and compares 6 architectures on EuroSAT,
then independently repeats the experiment on UC Merced.

### Models trained on EuroSAT

| # | Model | Source | Pretrained | ~Params |
|---|-------|--------|------------|---------|
| 1 | **ViT-Base/16** (`vit_base_patch16_224`) | `timm` | ✅ (ImageNet-21k → 1k) | ~86 M |
| 2 | **ResNet-18** | `torchvision` | ✅ (ImageNet-1k) | ~11 M |
| 3 | **EfficientNet-B0** | `timm` | ✅ | ~5 M |
| 4 | **DenseNet-121** | `torchvision` | ✅ | ~8 M |
| 5 | **Simple CNN** | from scratch | ❌ | ~1 M |
| 6 | **Swin-Tiny** (`swin_tiny_patch4_window7_224`) | `timm` | ✅ | ~28 M |

### Models trained on UC Merced

The same 5 transferable backbones (DenseNet-121, ViT, Swin-Tiny, ResNet-18, Simple CNN),
each trained independently on UC Merced with a 70 / 15 / 15 stratified split.

### Shared training utilities

- **Preprocessing:** `Resize(128×128)` + ImageNet normalisation;
  augmentations (`RandomHorizontalFlip`, `RandomRotation(10°)`, `ColorJitter`) on train only.
- **Optimiser:** AdamW, default `lr=1e-4` for fine-tuning pretrained models.
- **Loss:** Cross-Entropy.
- **Early stopping:** patience 5, monitor val loss.
- **Metrics:** accuracy, balanced accuracy, macro F1, weighted F1, confusion matrix.
- **Attribution (Captum):** `IntegratedGradients`, `GradientShap`, `Saliency`,
  `Occlusion` are imported and ready to use on any of the trained models.

### Common settings

```python
device       = torch.device('cpu')   # Mac M1 optimised; switch to 'cuda' if available
image_size   = 128
batch_size   = 32
num_epochs   = 10                    # ViT/ResNet/DenseNet/Swin/SimpleCNN
learning_rate = 1e-4
early_stopping_patience = 5
```

### Run it

```bash
jupyter lab "ViT_LULC_Classification 2 today.ipynb"
```

The notebook is structured so each model is a self-contained section — you can
re-run only one model without re-running the others. A final section assembles
`all_model_results` into a comparison table and bar charts, then repeats the
exercise on UC Merced and produces a cross-dataset comparison.

---

## Results

### Notebook 1 — `CustomEuroSATCNN` (locked test set)

From `outputs/test_metrics.csv`:

| Metric | Value |
|---|---|
| Test accuracy           | **96.59 %** |
| Test top-2 accuracy     | **99.48 %** |
| Macro F1                | 0.9639 |
| Weighted F1             | 0.9659 |
| Balanced accuracy       | 0.9640 |
| Parameters (total / trainable) | 11,263,042 / 11,263,042 |
| Best checkpoint         | `outputs/checkpoints/best_custom.pth` |

> The notebook also writes `outputs/final_test/` with the per-image predictions,
> classification report, and confusion-matrix PNG.

### Notebook 2 — multi-model comparison

Each model is evaluated on a held-out test split and compared against the original
EuroSAT paper baselines. The notebook produces:

- A **comparison table** with `Test Accuracy / Macro F1 / Balanced Acc / Total Params`
  for all 6 models on EuroSAT.
- A second comparison for the 5 UC Merced models.
- A side-by-side table that places the EuroSAT and UC Merced numbers next to each other.

(Numbers will vary depending on epochs and hardware — run the notebook end-to-end to
populate them.)

---

## Outputs

Notebook 1 writes everything under `outputs/`:

```text
outputs/
├── checkpoints/
│   ├── best_custom.pth
│   ├── last_custom.pth
│   └── custom_weights.pth
├── checkpoints.json              # registry of saved checkpoints
├── label_map.json                # class_to_idx
├── split_train.csv               # deterministic train split
├── split_val.csv                 # deterministic val split
├── samples_per_class/            # quick visual sanity check
├── class_overview.png
├── predictions/                  # per-image top-1/top-2 + confidence
├── test_metrics/
│   ├── classification_report.txt
│   └── confusion_matrix.png
├── test_metrics.csv              # one-row summary (see above)
├── val_metrics.csv               # epoch-level val metrics
├── history_custom.csv            # full training history
├── xai/
│   ├── gradcam/                  # one Grad-CAM PNG per (model, class)
│   ├── lime/                     # one LIME PNG per (model, class)
│   └── bundles/
│       └── xai_bundle_custom.pth # weights + class map + reshape kind
├── xai_report.csv
├── xai_sample_set.csv
└── final_gradcam_grid.png        # one-image-per-class summary grid
```

---

## Reproducibility

- Global seed = `42` is set for `random`, `numpy`, and `torch` (CPU + CUDA).
- `torch.backends.cudnn.deterministic = False` and `benchmark = True` for throughput
  (this is documented in Notebook 1; flip them if you need bit-exact reruns).
- DataLoader workers are seeded via `seed_worker`.
- The train/val split is deterministic and saved to `outputs/split_train.csv` /
  `outputs/split_val.csv` so reruns reuse the exact same partition.

---

## License

Code in this repository is released for research / coursework purposes.
EuroSAT and UC Merced retain their original licenses — please cite the original
papers when using them:

- Helber et al., *EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land
  Use and Land Cover Classification*, IEEE JSTARS 2019.
- Yang & Newsam, *Bag-of-Visual-Words and Spatial Extensions for Land-Use
  Classification*, ACM SIGSPATIAL 2010 (UC Merced).

