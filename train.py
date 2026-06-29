"""
Multi-Task Neural Network — IV fluid contamination detection

Assumes train_set is already loaded in memory as a pandas DataFrame.
Run this script from your IDE (e.g. Spyder) with train_set in the namespace.

Outputs (saved to OUT_DIR):
  model.pt          – trained weights + metadata (input_dim, le_classes, feature_cols)
  preprocessor.pkl  – fitted imputer + scaler pipeline
  metrics.csv/.xlsx – per-split metrics and summary with 95% CI

Inference example:
  import pickle, numpy as np, torch
  from train import MultiTaskNet
  ckpt  = torch.load("model.pt", map_location="cpu")
  model = MultiTaskNet(ckpt["input_dim"], ckpt["n_classes"])
  model.load_state_dict(ckpt["model_state_dict"])
  model.eval()
  with open("preprocessor.pkl", "rb") as f:
      pipe = pickle.load(f)
  X_proc = torch.tensor(pipe.transform(X_new).astype("float32"))
  with torch.no_grad():
      logit_bin, logit_cont, log_reg = model(X_proc)
  prob  = torch.sigmoid(logit_bin).numpy()           # P(contaminated)
  cls   = torch.argmax(logit_cont, dim=1).numpy()    # contaminant class index
  grade = np.expm1(log_reg.numpy()).clip(0)           # severity grade (%)
  label = np.array(ckpt["le_classes"])[cls]          # contaminant label
"""

import os
import copy
import random
import pickle
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score,
    mean_absolute_error, mean_squared_error,
    precision_score, recall_score, r2_score, roc_auc_score,
)
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# SETTINGS
# ──────────────────────────────────────────────────────────────────────────────

SEED       = 42
EPOCHS     = 100
PATIENCE   = 10
BATCH_SIZE = 32
LR         = 1e-3
N_REPEATS  = 30
TEST_SIZE  = 0.20
OUT_DIR    = "mtnn_outputs"

# ──────────────────────────────────────────────────────────────────────────────
# REPRODUCIBILITY AND DEVICE
# ──────────────────────────────────────────────────────────────────────────────

def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

seed_everything(SEED)

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Device : {device}")
os.makedirs(OUT_DIR, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# DATA
# ──────────────────────────────────────────────────────────────────────────────

def to_numeric(s):
    return pd.to_numeric(
        s.astype(str).str.strip().str.replace(",", ".", regex=False)
         .replace(["nan", "None", "", "NaN", "NAN"], np.nan),
        errors="coerce",
    )

data = train_set.copy()

feature_cols = [c for c in data.columns if c.startswith("NRC_")]

for c in feature_cols:
    data[c] = to_numeric(data[c])

data["Tipo"]     = to_numeric(data["Tipo"])
data["Outcome"]  = data["Outcome"].astype(int)
data["Contaminante"] = data["Contaminante"].astype(str).str.strip()

# known encoding fix
data.loc[(data["Contaminante"] == "Salino") & (data["Tipo"] == 6), "Tipo"] = 5

print(f"Dataset: {data.shape[0]} rows × {len(feature_cols)} features")

# ── targets ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
y_bin = data["Outcome"].values.astype(int)

data["Contaminante_model"] = data["Contaminante"].replace(
    {"Glucosado": "GC", "Glucosalino": "GC", "No": "No", "Salino": "NS"}
)
le = LabelEncoder()
y_cont = le.fit_transform(data["Contaminante_model"])
print(f"Contaminant classes: {list(le.classes_)}")
print(data["Contaminante_model"].value_counts().to_string())

y_reg_log = np.log1p(data["Tipo"].fillna(0).values.astype(float))

# ── stratification ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────
tipo_str  = data["Tipo"].astype(str).where(data["Tipo"].notna(), "unknown")
strata    = data["Contaminante_model"].astype(str) + "__" + tipo_str
freq      = strata.value_counts()
rare      = freq[freq < 2].index
data["strata_final"] = strata.where(
    ~strata.isin(rare),
    data["Contaminante_model"].astype(str) + "__RARE"
)

X_full = data[feature_cols].copy()

# ──────────────────────────────────────────────────────────────────────────────
# MODEL
# ──────────────────────────────────────────────────────────────────────────────

class MultiTaskNet(nn.Module):
    def __init__(self, input_dim, n_classes):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.30),
            nn.Linear(64, 32),        nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(0.20),
            nn.Linear(32, 16),        nn.ReLU(),
        )
        self.head_bin  = nn.Linear(16, 1)
        self.head_cont = nn.Linear(16, n_classes)
        self.head_reg  = nn.Linear(16, 1)

    def forward(self, x):
        h = self.shared(x)
        return self.head_bin(h), self.head_cont(h), self.head_reg(h)

# ──────────────────────────────────────────────────────────────────────────────
# ENTRENAR UN SPLIT
# ──────────────────────────────────────────────────────────────────────────────

loss_b = nn.BCEWithLogitsLoss()
loss_c = nn.CrossEntropyLoss()
loss_r = nn.HuberLoss()

def train_one_split(split_seed):
    (X_tr, X_va,
     yb_tr, yb_va,
     yc_tr, yc_va,
     yr_tr, yr_va,
     df_tr, df_va) = train_test_split(
        X_full, y_bin, y_cont, y_reg_log, data,
        test_size=TEST_SIZE,
        random_state=split_seed,
        stratify=data["strata_final"],
    )
    for df_ in [X_tr, X_va, df_tr, df_va]:
        df_.reset_index(drop=True, inplace=True)

    pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("scl", StandardScaler())])
    Xtr = pipe.fit_transform(X_tr).astype(np.float32)
    Xva = pipe.transform(X_va).astype(np.float32)

    def tt(a, dtype): return torch.tensor(a, dtype=dtype)

    loader = DataLoader(
        TensorDataset(tt(Xtr, torch.float32),
                      tt(yb_tr.reshape(-1,1), torch.float32),
                      tt(yc_tr, torch.long),
                      tt(yr_tr.reshape(-1,1), torch.float32)),
        batch_size=BATCH_SIZE, shuffle=True
    )

    model = MultiTaskNet(Xtr.shape[1], len(le.classes_)).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=5, min_lr=1e-5)

    Xva_t  = tt(Xva, torch.float32).to(device)
    yb_va_t = tt(yb_va.reshape(-1,1), torch.float32).to(device)
    yc_va_t = tt(yc_va, torch.long).to(device)
    yr_va_t = tt(yr_va.reshape(-1,1), torch.float32).to(device)

    best_val, no_imp, best_state = np.inf, 0, None

    for epoch in range(EPOCHS):
        model.train()
        for xb, yb_, yc_, yr_ in loader:
            xb, yb_, yc_, yr_ = xb.to(device), yb_.to(device), yc_.to(device), yr_.to(device)
            pb, pc, pr = model(xb)
            loss = loss_b(pb, yb_) + loss_c(pc, yc_) + loss_r(pr, yr_)
            opt.zero_grad(); loss.backward(); opt.step()

        model.eval()
        with torch.no_grad():
            pb, pc, pr = model(Xva_t)
            val_loss = (loss_b(pb, yb_va_t) + loss_c(pc, yc_va_t) + loss_r(pr, yr_va_t)).item()

        sched.step(val_loss)
        if val_loss < best_val:
            best_val, no_imp = val_loss, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            no_imp += 1
        if no_imp >= PATIENCE:
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pb, pc, pr = model(Xva_t)

    prob_bin  = torch.sigmoid(pb).cpu().numpy().ravel()
    pred_bin  = (prob_bin >= 0.5).astype(int)
    pred_cont = torch.argmax(pc, dim=1).cpu().numpy()
    pred_reg  = np.expm1(pr.cpu().numpy().ravel()).clip(0)
    y_reg_real = np.expm1(yr_va)

    tn, fp, fn, tp = confusion_matrix(yb_va, pred_bin, labels=[0,1]).ravel()
    rho, _ = spearmanr(y_reg_real, pred_reg)

    return {
        "AUC":          roc_auc_score(yb_va, prob_bin) if len(np.unique(yb_va)) > 1 else np.nan,
        "sensitivity":  tp/(tp+fn) if (tp+fn) else np.nan,
        "specificity":  tn/(tn+fp) if (tn+fp) else np.nan,
        "PPV":          tp/(tp+fp) if (tp+fp) else np.nan,
        "NPV":          tn/(tn+fn) if (tn+fn) else np.nan,
        "accuracy_bin": accuracy_score(yb_va, pred_bin),
        "F1_bin":       f1_score(yb_va, pred_bin, zero_division=0),
        "accuracy_multi":  accuracy_score(yc_va, pred_cont),
        "macro_f1":        f1_score(yc_va, pred_cont, average="macro",    zero_division=0),
        "weighted_f1":     f1_score(yc_va, pred_cont, average="weighted", zero_division=0),
        "MAE":          mean_absolute_error(y_reg_real, pred_reg),
        "RMSE":         float(np.sqrt(mean_squared_error(y_reg_real, pred_reg))),
        "R2":           r2_score(y_reg_real, pred_reg),
        "Spearman_rho": rho,
    }

# ──────────────────────────────────────────────────────────────────────────────
# REPEATED VALIDATION
# ──────────────────────────────────────────────────────────────────────────────

print(f"\n── Repeated validation ({N_REPEATS} splits) ──")
rows = []
for rep in range(N_REPEATS):
    split_seed = SEED + rep
    seed_everything(split_seed)
    print(f"  Split {rep+1:02d}/{N_REPEATS} | seed={split_seed}", end=" ")
    m = train_one_split(split_seed)
    m["split"] = rep + 1
    rows.append(m)
    print(f"| AUC={m['AUC']:.3f}  F1={m['F1_bin']:.3f}  MAE={m['MAE']:.3f}")

df_splits = pd.DataFrame(rows)

# ──────────────────────────────────────────────────────────────────────────────
# SUMMARY WITH 95% CI
# ──────────────────────────────────────────────────────────────────────────────

metric_cols = [c for c in df_splits.columns if c != "split"]
summary_rows = []
for col in metric_cols:
    v = df_splits[col].dropna().values
    summary_rows.append({
        "metric":     col,
        "mean":       np.mean(v),
        "sd":         np.std(v, ddof=1) if len(v) > 1 else np.nan,
        "IC95_low":   np.percentile(v, 2.5),
        "IC95_high":  np.percentile(v, 97.5),
        "n_splits":   len(v),
    })
df_summary = pd.DataFrame(summary_rows)

print("\n── Final summary (mean ± SD, 95% CI) ──")
print(df_summary.to_string(index=False))

# ──────────────────────────────────────────────────────────────────────────────
# SAVE METRICS
# ──────────────────────────────────────────────────────────────────────────────

metrics_path = os.path.join(OUT_DIR, "metrics.xlsx")
with pd.ExcelWriter(metrics_path) as w:
    df_splits.to_excel(w, sheet_name="per_split", index=False)
    df_summary.to_excel(w, sheet_name="summary_IC95", index=False)
df_summary.to_csv(os.path.join(OUT_DIR, "metrics.csv"), index=False)
print(f"\nMetrics → {metrics_path}")

# ──────────────────────────────────────────────────────────────────────────────
# FINAL MODEL (full dataset)
# ──────────────────────────────────────────────────────────────────────────────

print("\n── Training final model on full dataset ──")
seed_everything(SEED)

pipe_final = Pipeline([("imp", SimpleImputer(strategy="median")),
                       ("scl", StandardScaler())])
Xall = pipe_final.fit_transform(X_full).astype(np.float32)

def tt(a, dtype): return torch.tensor(a, dtype=dtype)

loader_final = DataLoader(
    TensorDataset(tt(Xall, torch.float32),
                  tt(y_bin.reshape(-1,1), torch.float32),
                  tt(y_cont, torch.long),
                  tt(y_reg_log.reshape(-1,1), torch.float32)),
    batch_size=BATCH_SIZE, shuffle=True
)

model_final = MultiTaskNet(Xall.shape[1], len(le.classes_)).to(device)
opt_final   = torch.optim.Adam(model_final.parameters(), lr=LR)

for epoch in range(EPOCHS):
    model_final.train()
    for xb, yb_, yc_, yr_ in loader_final:
        xb, yb_, yc_, yr_ = xb.to(device), yb_.to(device), yc_.to(device), yr_.to(device)
        pb, pc, pr = model_final(xb)
        loss = loss_b(pb, yb_) + loss_c(pc, yc_) + loss_r(pr, yr_)
        opt_final.zero_grad(); loss.backward(); opt_final.step()
    if (epoch + 1) % 20 == 0:
        print(f"  Epoch {epoch+1}/{EPOCHS}")

# ──────────────────────────────────────────────────────────────────────────────
# SAVE MODEL AND PREPROCESSOR
# ──────────────────────────────────────────────────────────────────────────────

model_path = os.path.join(OUT_DIR, "model.pt")
torch.save({
    "model_state_dict": model_final.state_dict(),
    "input_dim":        Xall.shape[1],
    "n_classes":        len(le.classes_),
    "le_classes":       list(le.classes_),
    "feature_cols":     feature_cols,
    "seed":             SEED,
}, model_path)

prep_path = os.path.join(OUT_DIR, "preprocessor.pkl")
with open(prep_path, "wb") as f:
    pickle.dump(pipe_final, f)

print(f"Model        → {model_path}")
print(f"Preprocessor → {prep_path}")
print("\nDone ✓")
