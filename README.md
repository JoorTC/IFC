[README.md](https://github.com/user-attachments/files/29479492/README.md)
# IFC — Automated Detection of Intravenous Fluid Contamination

Code repository for the paper:

> **Automated Detection of Intravenous Fluid Contamination Using Longitudinal Routine Laboratory Data: Development and External Validation of a Multitask Learning Framework**  
> Tortosa-Carreres J, Martí-Montoro N, Acevedo-Galvis JA, Pascual-Escrivà C, Alonso-Díaz R, Fuster-Lluch Ó, Sahuquillo-Frías L, Laiz-Marro B.  
> *Citation will be updated upon publication.*

---

## Overview

A PyTorch multitask neural network (MTNN) that simultaneously performs three tasks from Normalized Rate-of-Change (NRC) features derived from paired routine laboratory results:

| Head | Task | Loss |
|---|---|---|
| Binary | Contamination yes/no | BCEWithLogitsLoss |
| Multiclass | Contaminant type — NS (normal saline) or gC (glucose-containing) | CrossEntropyLoss |
| Regression | Contamination severity grade (%) | HuberLoss |

The model achieved ROC-AUC 0.94–0.96 across internal (IRP, n=30 splits) and external validation cohorts, with sensitivity and specificity ≥85%. A structured risk stratification framework derived from model outputs achieved 99.1% sensitivity in the external cohort while reducing alert-triggering samples by ~17%.

---

## Repository structure

```
IFC/
├── train.py          # full training pipeline (single script)
├── README.md
├── requirements.txt
├── LICENSE
└── outputs/          # created on first run
    ├── model.pt            # trained weights + metadata
    ├── preprocessor.pkl    # fitted imputer + scaler
    ├── metrics.csv         # summary with 95% CI
    └── metrics.xlsx        # per-split + summary sheets
```

---

## Dataset format (`train_set`)

The script reads directly from a pandas DataFrame named `train_set` already loaded in memory. Required columns:

| Column | Type | Description |
|---|---|---|
| `NRC_*` | str / float | Pre-computed NRC features (e.g. `NRC_Glu`, `NRC_Na`, `NRC_PLT`, …) |
| `Outcome` | int (0/1) | 0 = not contaminated, 1 = contaminated |
| `Contaminante` | str | `"Salino"`, `"Glucosado"`, `"Glucosalino"`, `"No"` |
| `Tipo` | str / float | Contamination severity grade (0–5 scale) |

NRC variables are calculated as:

```
NRC = (x_current - x_previous) / ((x_current + x_previous) / 2)
```

Glucose-containing fluids (`"Glucosado"` and `"Glucosalino"`) are merged into a single class (`"gC"`) during training, consistent with their similar analytical signature (multiclass macro-F1 = 0.72 when treated separately; 95% CI 0.69–0.78).

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Usage

Open `train.py` in your IDE with `train_set` already loaded in the namespace and run it. No command-line arguments needed.

Change `OUT_DIR` at the top of the script to set the output folder.

---

## Outputs

### `model.pt`

```python
{
  "model_state_dict": ...,    # network weights
  "input_dim":        int,    # number of NRC features
  "n_classes":        int,    # number of contaminant classes
  "le_classes":       list,   # ["GC", "No", "NS"]
  "feature_cols":     list,   # ordered NRC column names
  "seed":             int,
}
```

### `metrics.csv` / `metrics.xlsx`
- **per_split**: AUC, sensitivity, specificity, PPV, NPV, F1, accuracy (binary + multiclass), MAE, RMSE, R², Spearman ρ — one row per split.
- **summary_IC95**: mean ± SD and 95% percentile CI across all 30 splits.

---

## Inference

```python
import pickle
import numpy as np
import torch
from train import MultiTaskNet

ckpt = torch.load("outputs/model.pt", map_location="cpu")
model = MultiTaskNet(ckpt["input_dim"], ckpt["n_classes"])
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

with open("outputs/preprocessor.pkl", "rb") as f:
    pipe = pickle.load(f)

# X_new: DataFrame with the same NRC_* columns as train_set
X_proc = torch.tensor(pipe.transform(X_new).astype("float32"))

with torch.no_grad():
    logit_bin, logit_cont, log_reg = model(X_proc)

prob_contaminated  = torch.sigmoid(logit_bin).numpy()          # P(contaminated)
contaminant_class  = torch.argmax(logit_cont, dim=1).numpy()   # class index
severity_grade     = np.expm1(log_reg.numpy()).clip(0)          # grade (%)
contaminant_label  = np.array(ckpt["le_classes"])[contaminant_class]  # "GC", "NS", or "No"
```

---

## Architecture

```
Input (NRC features)
    │
    ▼
Linear → BatchNorm → ReLU → Dropout(0.30)   [64 units]
    │
    ▼
Linear → BatchNorm → ReLU → Dropout(0.20)   [32 units]
    │
    ▼
Linear → ReLU                               [16 units]
    │
    ├──► head_bin  : Linear(16→1)  → sigmoid  →  P(contaminated)
    ├──► head_cont : Linear(16→n)  → softmax  →  contaminant type
    └──► head_reg  : Linear(16→1)  → expm1    →  severity grade (%)
```

---

## Validation strategy

Repeated hold-out (IRP, 30 splits, 80/20 stratified by contaminant type × severity grade). Early stopping on total validation loss (patience = 10). A representative split (IRP-ref) was selected as the partition whose metrics were closest to the IRP mean (minimum sum of standardised distances for AUC, accuracy, F1, and RMSE). External validation (EVC) used a model trained on the full internal cohort.

---

## Requirements

```
torch>=2.0
numpy>=1.24
pandas>=2.0
scikit-learn>=1.3
scipy>=1.11
openpyxl>=3.1
```

---

## License

MIT — see `LICENSE`.

---

## Citation

If you use this code, please cite:

```
Citation will be added upon publication.
```
