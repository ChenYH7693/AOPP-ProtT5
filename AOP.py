#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
train_aopp_prott5_cv.py

Antioxidant peptide binary classification with ProtT5 + 5-fold CV

Input CSV columns:
- SEQUENCE
- label
(also supports accidental column name: labe)

Example:
python train_aopp_prott5_cv.py \
    --csv_path data.csv \
    --output_dir outputs \
    --epochs 10 \
    --batch_size 4 \
    --max_len 256 \
    --lr_head 2e-4 \
    --lr_encoder 1e-5 \
    --unfreeze_last_n_layers 1
"""

import os
import re
import gc
import json
import math
import time
import copy
import random
import argparse
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    matthews_corrcoef,
    accuracy_score,
    precision_score,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)

from transformers import (
    T5EncoderModel,
    T5Tokenizer,
    get_cosine_schedule_with_warmup,
)

warnings.filterwarnings("ignore")


# =========================
# Utility
# =========================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def current_time():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def clean_sequence(seq: str) -> str:
    seq = str(seq).strip().upper().replace(" ", "")
    seq = re.sub(r"[^A-Z]", "", seq)
    seq = re.sub(r"[UZOB]", "X", seq)
    return seq


def format_sequence_for_prott5(seq: str) -> str:
    return " ".join(list(seq))


def safe_div(a: float, b: float) -> float:
    return a / b if b != 0 else 0.0


def save_json(obj, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# =========================
# Metrics
# =========================
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    sensitivity = safe_div(tp, tp + fn)
    specificity = safe_div(tn, tn + fp)

    return {
        "MCC": float(mcc),
        "Accuracy": float(acc),
        "Precision": float(prec),
        "Sensitivity": float(sensitivity),
        "Specificity": float(specificity),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


def find_best_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: str = "Accuracy"
):
    thresholds = np.arange(0.05, 0.951, 0.01)
    best_thr = 0.5
    best_metrics = None
    best_score = -1e9

    for thr in thresholds:
        y_pred = (y_prob >= thr).astype(int)
        metrics = compute_metrics(y_true, y_pred)
        score = metrics[metric]

        if score > best_score:
            best_score = score
            best_thr = float(thr)
            best_metrics = metrics

    return best_thr, best_metrics


# =========================
# Dataset
# =========================
class PeptideDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return {
            "sequence": self.df.loc[idx, "SEQUENCE"],
            "label": int(self.df.loc[idx, "label"]),
        }


class Collator:
    def __init__(self, tokenizer, max_len: int = 256):
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __call__(self, batch):
        seqs = [format_sequence_for_prott5(x["sequence"]) for x in batch]
        labels = torch.tensor([x["label"] for x in batch], dtype=torch.float32)

        enc = self.tokenizer(
            seqs,
            padding=True,
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )

        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": labels,
        }


# =========================
# Loss
# =========================
class FocalBCELoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, pos_weight=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none", pos_weight=self.pos_weight
        )
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_weight = self.alpha * (1 - pt).pow(self.gamma)
        loss = focal_weight * bce
        return loss.mean()


# =========================
# EMA
# =========================
class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                new_average = self.decay * self.shadow[name] + (1.0 - self.decay) * param.data
                self.shadow[name] = new_average.clone()

    def apply_shadow(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


# =========================
# Model
# =========================
class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.2):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        score = self.attn(x).squeeze(-1)
        score = score.masked_fill(mask == 0, -1e4)
        weight = torch.softmax(score, dim=-1)
        pooled = torch.sum(x * weight.unsqueeze(-1), dim=1)
        return pooled


class GeMPooling(nn.Module):
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x, mask):
        mask = mask.unsqueeze(-1).float()
        x = x.clamp(min=self.eps) * mask
        denom = mask.sum(dim=1).clamp(min=self.eps)
        x = (x.pow(self.p).sum(dim=1) / denom).pow(1.0 / self.p)
        return x


class ResidualMLP(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.3):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * 2)
        self.fc2 = nn.Linear(dim * 2, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x + residual


class MultiSampleDropoutHead(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim=1, dropout=0.4, samples=5):
        super().__init__()
        self.samples = samples
        self.dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in range(samples)])
        self.proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        logits = [self.proj(dp(x)) for dp in self.dropouts]
        logits = torch.stack(logits, dim=0).mean(dim=0)
        return logits.squeeze(-1)


class AdvancedProtT5Classifier(nn.Module):
    def __init__(
        self,
        model_name: str = "Rostlab/prot_t5_xl_half_uniref50-enc",
        dropout: float = 0.3,
        unfreeze_last_n_layers: int = 2,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.encoder = T5EncoderModel.from_pretrained(model_name)

        if gradient_checkpointing and hasattr(self.encoder, "gradient_checkpointing_enable"):
            self.encoder.gradient_checkpointing_enable()

        hidden_dim = self.encoder.config.d_model

        for p in self.encoder.parameters():
            p.requires_grad = False

        if unfreeze_last_n_layers > 0 and hasattr(self.encoder.encoder, "block"):
            blocks = self.encoder.encoder.block
            for block in blocks[-unfreeze_last_n_layers:]:
                for p in block.parameters():
                    p.requires_grad = True
            for p in self.encoder.encoder.final_layer_norm.parameters():
                p.requires_grad = True

        self.attn_pool = AttentionPooling(hidden_dim, dropout=dropout)
        self.gem_pool = GeMPooling()

        fusion_dim = hidden_dim * 4
        self.fusion = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.res1 = ResidualMLP(hidden_dim, dropout=dropout)
        self.res2 = ResidualMLP(hidden_dim, dropout=dropout)

        self.cls = MultiSampleDropoutHead(
            in_dim=hidden_dim,
            hidden_dim=hidden_dim // 2,
            out_dim=1,
            dropout=dropout,
            samples=5,
        )

    @staticmethod
    def masked_mean_pool(x: torch.Tensor, mask: torch.Tensor):
        mask = mask.unsqueeze(-1).float()
        x = x * mask
        summed = x.sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1e-6)
        return summed / denom

    @staticmethod
    def masked_max_pool(x: torch.Tensor, mask: torch.Tensor):
        mask = mask.unsqueeze(-1).bool()
        x = x.masked_fill(~mask, -1e4)
        return x.max(dim=1).values

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        x = out.last_hidden_state

        mean_pool = self.masked_mean_pool(x, attention_mask)
        max_pool = self.masked_max_pool(x, attention_mask)
        attn_pool = self.attn_pool(x, attention_mask)
        gem_pool = self.gem_pool(x, attention_mask)

        feat = torch.cat([mean_pool, max_pool, attn_pool, gem_pool], dim=-1)
        feat = self.fusion(feat)
        feat = self.res1(feat)
        feat = self.res2(feat)

        logits = self.cls(feat)
        return logits


# =========================
# Config
# =========================
@dataclass
class Config:
    csv_path: str
    output_dir: str
    model_name: str
    n_splits: int
    epochs: int
    batch_size: int
    num_workers: int
    max_len: int
    lr_head: float
    lr_encoder: float
    weight_decay: float
    warmup_ratio: float
    early_stopping_patience: int
    seed: int
    device: str
    dropout: float
    unfreeze_last_n_layers: int
    grad_accum_steps: int
    max_grad_norm: float
    use_amp: bool
    gradient_checkpointing: bool
    ema_decay: float
    use_focal_loss: bool
    focal_alpha: float
    focal_gamma: float
    select_metric: str


# =========================
# Data loading
# =========================
def load_dataframe(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    col_map = {c.lower(): c for c in df.columns}

    if "sequence" not in col_map:
        raise ValueError("CSV must contain column: SEQUENCE")

    if "label" in col_map:
        label_col = col_map["label"]
    elif "labe" in col_map:
        label_col = col_map["labe"]
    else:
        raise ValueError("CSV must contain column: label or labe")

    seq_col = col_map["sequence"]

    df = df[[seq_col, label_col]].copy()
    df.columns = ["SEQUENCE", "label"]

    df["SEQUENCE"] = df["SEQUENCE"].astype(str).apply(clean_sequence)
    df["label"] = df["label"].astype(int)
    df = df[df["SEQUENCE"].str.len() > 0].reset_index(drop=True)

    unique_labels = sorted(df["label"].unique().tolist())
    if set(unique_labels) - {0, 1}:
        raise ValueError(f"Labels must be binary 0/1, got {unique_labels}")

    return df


# =========================
# Optimization
# =========================
def build_optimizer(model: nn.Module, cfg: Config):
    encoder_params = []
    head_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("encoder."):
            encoder_params.append(param)
        else:
            head_params.append(param)

    param_groups = []
    if len(head_params) > 0:
        param_groups.append({
            "params": head_params,
            "lr": cfg.lr_head,
            "weight_decay": cfg.weight_decay,
        })
    if len(encoder_params) > 0:
        param_groups.append({
            "params": encoder_params,
            "lr": cfg.lr_encoder,
            "weight_decay": cfg.weight_decay,
        })

    return torch.optim.AdamW(param_groups)


def get_current_lrs(optimizer):
    lrs = [group["lr"] for group in optimizer.param_groups]
    if len(lrs) == 1:
        return lrs[0], lrs[0]
    return lrs[0], lrs[1]


# =========================
# Plotting
# =========================
def plot_fold_curves(history_df: pd.DataFrame, fold_dir: str, fold: int):
    plt.figure(figsize=(8, 6))
    plt.plot(history_df["epoch"], history_df["train_loss"], marker="o", label="Train Loss")
    plt.plot(history_df["epoch"], history_df["val_loss"], marker="s", label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Fold {fold} Loss Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fold_dir, f"fold_{fold}_loss_curve.png"), dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.plot(history_df["epoch"], history_df["Accuracy"], marker="o", label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"Fold {fold} Accuracy Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fold_dir, f"fold_{fold}_accuracy_curve.png"), dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.plot(history_df["epoch"], history_df["MCC"], marker="o", label="Val MCC")
    plt.xlabel("Epoch")
    plt.ylabel("MCC")
    plt.title(f"Fold {fold} MCC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fold_dir, f"fold_{fold}_mcc_curve.png"), dpi=300, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix_figure(cm, save_path, title="Confusion Matrix"):
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["Negative", "Positive"])
    plt.yticks(tick_marks, ["Negative", "Positive"])

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, format(cm[i, j], "d"),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black"
            )

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_roc_curve_figure(y_true, y_prob, save_path, title="ROC Curve"):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_pr_curve_figure(y_true, y_prob, save_path, title="PR Curve"):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)

    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label=f"AP = {ap:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_cv_metric_bars(results_df: pd.DataFrame, output_dir: str):
    metrics = ["MCC", "Accuracy", "Precision", "Sensitivity", "Specificity"]

    plt.figure(figsize=(12, 6))
    x = np.arange(len(results_df))
    width = 0.15

    for i, metric in enumerate(metrics):
        plt.bar(x + i * width, results_df[metric], width=width, label=metric)

    plt.xticks(x + width * 2, [f"Fold {i}" for i in results_df["fold"]])
    plt.ylabel("Score")
    plt.title("5-Fold Cross Validation Metrics")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cv_fold_metrics.png"), dpi=300, bbox_inches="tight")
    plt.close()

    means = [results_df[m].mean() for m in metrics]
    stds = [results_df[m].std(ddof=1) if len(results_df) > 1 else 0.0 for m in metrics]

    plt.figure(figsize=(8, 6))
    plt.bar(metrics, means, yerr=stds, capsize=5)
    plt.ylabel("Score")
    plt.title("5-Fold Mean Metrics (Mean ± Std)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cv_mean_metrics.png"), dpi=300, bbox_inches="tight")
    plt.close()


# =========================
# Train / Eval
# =========================
def train_one_epoch(model, loader, optimizer, scheduler, scaler, criterion, ema, cfg: Config, fold: int, epoch: int):
    model.train()
    total_loss = 0.0
    total_steps = 0

    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(loader, desc=f"Fold {fold} Epoch {epoch} [Train]", leave=True, ncols=140)

    for step, batch in enumerate(pbar):
        input_ids = batch["input_ids"].to(cfg.device)
        attention_mask = batch["attention_mask"].to(cfg.device)
        labels = batch["labels"].to(cfg.device)

        with torch.cuda.amp.autocast(enabled=cfg.use_amp):
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(logits, labels)
            loss = loss / cfg.grad_accum_steps

        if cfg.use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step + 1) % cfg.grad_accum_steps == 0:
            if cfg.use_amp:
                scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)

            if cfg.use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)

            if scheduler is not None:
                scheduler.step()

            if ema is not None:
                ema.update(model)

        total_loss += loss.item() * cfg.grad_accum_steps
        total_steps += 1

        lr_head, lr_encoder = get_current_lrs(optimizer)
        pbar.set_postfix({
            "loss": f"{total_loss / max(total_steps, 1):.4f}",
            "lr_head": f"{lr_head:.2e}",
            "lr_enc": f"{lr_encoder:.2e}",
        })

    return total_loss / max(total_steps, 1)


@torch.no_grad()
def predict_probs(model, loader, cfg: Config, fold: int, epoch: int):
    model.eval()
    all_probs = []
    all_labels = []
    losses = []

    val_criterion = nn.BCEWithLogitsLoss()

    pbar = tqdm(loader, desc=f"Fold {fold} Epoch {epoch} [Eval ]", leave=True, ncols=140)

    for batch in pbar:
        input_ids = batch["input_ids"].to(cfg.device)
        attention_mask = batch["attention_mask"].to(cfg.device)
        labels = batch["labels"].to(cfg.device)

        with torch.cuda.amp.autocast(enabled=cfg.use_amp):
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = val_criterion(logits, labels)

        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels.detach().cpu().numpy())
        losses.append(loss.item())

        pbar.set_postfix({"val_loss": f"{np.mean(losses):.4f}"})

    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    return all_labels, all_probs, float(np.mean(losses))


def train_one_fold(fold, train_df, val_df, tokenizer, cfg: Config):
    fold_dir = os.path.join(cfg.output_dir, f"fold_{fold}")
    ensure_dir(fold_dir)

    print("\n" + "=" * 100)
    print(f"[{current_time()}] Fold {fold}")
    print(f"Train size: {len(train_df)} | Val size: {len(val_df)}")
    print(f"Train pos: {int(train_df['label'].sum())} | neg: {int((train_df['label'] == 0).sum())}")
    print(f"Val   pos: {int(val_df['label'].sum())} | neg: {int((val_df['label'] == 0).sum())}")
    print("=" * 100)

    train_ds = PeptideDataset(train_df)
    val_ds = PeptideDataset(val_df)

    collator = Collator(tokenizer, max_len=cfg.max_len)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=collator,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=collator,
    )

    model = AdvancedProtT5Classifier(
        model_name=cfg.model_name,
        dropout=cfg.dropout,
        unfreeze_last_n_layers=cfg.unfreeze_last_n_layers,
        gradient_checkpointing=cfg.gradient_checkpointing,
    ).to(cfg.device)

    pos_count = max(int(train_df["label"].sum()), 1)
    neg_count = max(int((train_df["label"] == 0).sum()), 1)
    pos_weight = torch.tensor([neg_count / pos_count], dtype=torch.float32, device=cfg.device)

    if cfg.use_focal_loss:
        criterion = FocalBCELoss(
            alpha=cfg.focal_alpha,
            gamma=cfg.focal_gamma,
            pos_weight=pos_weight,
        )
    else:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = build_optimizer(model, cfg)

    updates_per_epoch = math.ceil(len(train_loader) / cfg.grad_accum_steps)
    total_steps = updates_per_epoch * cfg.epochs
    warmup_steps = int(total_steps * cfg.warmup_ratio)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=cfg.use_amp)
    ema = EMA(model, decay=cfg.ema_decay)

    best_state = None
    best_val_score = -1e9
    best_epoch = -1
    best_threshold = 0.5
    patience_counter = 0
    history = []

    for epoch in range(1, cfg.epochs + 1):
        print(f"\n[{current_time()}] Fold {fold} | Epoch {epoch}/{cfg.epochs}")

        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            criterion=criterion,
            ema=ema,
            cfg=cfg,
            fold=fold,
            epoch=epoch,
        )

        ema.apply_shadow(model)
        y_val, val_probs, val_loss = predict_probs(model, val_loader, cfg, fold, epoch)
        ema.restore(model)

        best_thr_epoch, val_metrics = find_best_threshold(
            y_true=y_val,
            y_prob=val_probs,
            metric=cfg.select_metric
        )

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "best_threshold": best_thr_epoch,
            **val_metrics,
        }
        history.append(row)
        history_df = pd.DataFrame(history)
        history_df.to_csv(os.path.join(fold_dir, "train_history.csv"), index=False)
        plot_fold_curves(history_df, fold_dir, fold)

        print(
            f"[Fold {fold} Epoch {epoch}] "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"Accuracy={val_metrics['Accuracy']:.4f} | "
            f"MCC={val_metrics['MCC']:.4f} | "
            f"Precision={val_metrics['Precision']:.4f} | "
            f"Sensitivity={val_metrics['Sensitivity']:.4f} | "
            f"Specificity={val_metrics['Specificity']:.4f} | "
            f"Threshold={best_thr_epoch:.2f}"
        )

        current_score = val_metrics[cfg.select_metric]

        if current_score > best_val_score:
            best_val_score = current_score
            best_epoch = epoch
            best_threshold = best_thr_epoch

            ema.apply_shadow(model)
            best_state = copy.deepcopy(model.state_dict())
            ema.restore(model)

            patience_counter = 0

            torch.save(best_state, os.path.join(fold_dir, "best_model.pt"))
            save_json(
                {
                    "fold": fold,
                    "best_epoch": best_epoch,
                    "best_threshold": best_threshold,
                    "select_metric": cfg.select_metric,
                    "best_score": best_val_score,
                    "best_val_metrics": val_metrics,
                },
                os.path.join(fold_dir, "best_info.json"),
            )
            print(f"[Fold {fold}] New best {cfg.select_metric}: {best_val_score:.4f} at epoch {best_epoch}")
        else:
            patience_counter += 1
            print(f"[Fold {fold}] No improvement. Early-stop patience: {patience_counter}/{cfg.early_stopping_patience}")

        if patience_counter >= cfg.early_stopping_patience:
            print(f"[{current_time()}] Early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_state)

    y_val, val_probs, val_loss = predict_probs(model, val_loader, cfg, fold, best_epoch)
    val_pred = (val_probs >= best_threshold).astype(int)
    final_metrics = compute_metrics(y_val, val_pred)

    pred_df = val_df.copy().reset_index(drop=True)
    pred_df["prob"] = val_probs
    pred_df["pred"] = val_pred
    pred_df.to_csv(os.path.join(fold_dir, "val_predictions.csv"), index=False)

    cm = confusion_matrix(y_val, val_pred, labels=[0, 1])
    plot_confusion_matrix_figure(
        cm,
        os.path.join(fold_dir, f"fold_{fold}_confusion_matrix.png"),
        title=f"Fold {fold} Confusion Matrix"
    )
    plot_roc_curve_figure(
        y_val,
        val_probs,
        os.path.join(fold_dir, f"fold_{fold}_roc_curve.png"),
        title=f"Fold {fold} ROC Curve"
    )
    plot_pr_curve_figure(
        y_val,
        val_probs,
        os.path.join(fold_dir, f"fold_{fold}_pr_curve.png"),
        title=f"Fold {fold} PR Curve"
    )

    fold_result = {
        "fold": fold,
        "best_epoch": best_epoch,
        "best_threshold": best_threshold,
        "val_loss": val_loss,
        **final_metrics,
    }

    save_json(fold_result, os.path.join(fold_dir, "fold_result.json"))

    print("\n" + "-" * 100)
    print(f"Fold {fold} final result")
    print(json.dumps(fold_result, ensure_ascii=False, indent=2))
    print("-" * 100)

    del model, optimizer, scheduler, scaler, train_loader, val_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return fold_result


def summarize_results(results: List[Dict[str, float]]):
    df = pd.DataFrame(results)
    metrics = ["MCC", "Accuracy", "Precision", "Sensitivity", "Specificity", "val_loss"]
    summary = {}

    for m in metrics:
        summary[m] = {
            "mean": float(df[m].mean()),
            "std": float(df[m].std(ddof=1)) if len(df) > 1 else 0.0,
        }
    return summary


# =========================
# Main
# =========================
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--csv_path", type=str, default="AOP_data/AnOxPP/AnOxPP.csv")
    parser.add_argument("--output_dir", type=str, default="AOP-outputs-accuracy-boost-AnOxPP")

    parser.add_argument("--model_name", type=str, default="Rostlab/prot_t5_xl_uniref50")
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--max_len", type=int, default=256)

    parser.add_argument("--lr_head", type=float, default=2e-4)
    parser.add_argument("--lr_encoder", type=float, default=8e-6)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)

    parser.add_argument("--early_stopping_patience", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--dropout", type=float, default=0.35)
    parser.add_argument("--unfreeze_last_n_layers", type=int, default=2)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    parser.add_argument("--ema_decay", type=float, default=0.999)

    parser.add_argument("--use_focal_loss", action="store_true")
    parser.add_argument("--focal_alpha", type=float, default=0.25)
    parser.add_argument("--focal_gamma", type=float, default=2.0)

    parser.add_argument(
        "--select_metric",
        type=str,
        default="Accuracy",
        choices=["Accuracy", "MCC", "Precision", "Sensitivity", "Specificity"]
    )

    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--no_amp", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()
    ensure_dir(args.output_dir)
    set_seed(args.seed)

    cfg = Config(
        csv_path=args.csv_path,
        output_dir=args.output_dir,
        model_name=args.model_name,
        n_splits=args.n_splits,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_len=args.max_len,
        lr_head=args.lr_head,
        lr_encoder=args.lr_encoder,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        early_stopping_patience=args.early_stopping_patience,
        seed=args.seed,
        device=args.device,
        dropout=args.dropout,
        unfreeze_last_n_layers=args.unfreeze_last_n_layers,
        grad_accum_steps=args.grad_accum_steps,
        max_grad_norm=args.max_grad_norm,
        use_amp=not args.no_amp,
        gradient_checkpointing=args.gradient_checkpointing,
        ema_decay=args.ema_decay,
        use_focal_loss=args.use_focal_loss,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
        select_metric=args.select_metric,
    )

    print(f"[{current_time()}] Loading data: {cfg.csv_path}")
    df = load_dataframe(cfg.csv_path)

    print(f"Dataset size: {len(df)}")
    print(f"Positive: {int(df['label'].sum())} | Negative: {int((df['label'] == 0).sum())}")
    print("Sequence length statistics:")
    print(df["SEQUENCE"].str.len().describe())

    df.to_csv(os.path.join(cfg.output_dir, "cleaned_dataset.csv"), index=False)
    save_json(vars(args), os.path.join(cfg.output_dir, "config.json"))

    print(f"[{current_time()}] Loading tokenizer: {cfg.model_name}")
    tokenizer = T5Tokenizer.from_pretrained(cfg.model_name, do_lower_case=False)

    skf = StratifiedKFold(n_splits=cfg.n_splits, shuffle=True, random_state=cfg.seed)

    results = []
    outer_pbar = tqdm(
        list(skf.split(df["SEQUENCE"], df["label"])),
        desc=f"{cfg.n_splits}-Fold CV",
        leave=True,
        ncols=140
    )

    for fold, (train_idx, val_idx) in enumerate(outer_pbar, start=1):
        outer_pbar.set_postfix({"current_fold": fold})

        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)

        fold_result = train_one_fold(
            fold=fold,
            train_df=train_df,
            val_df=val_df,
            tokenizer=tokenizer,
            cfg=cfg,
        )
        results.append(fold_result)

        tmp_df = pd.DataFrame(results)
        outer_pbar.set_postfix({
            "current_fold": fold,
            "mean_ACC": f"{tmp_df['Accuracy'].mean():.4f}",
            "mean_MCC": f"{tmp_df['MCC'].mean():.4f}",
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(cfg.output_dir, "cv_results.csv"), index=False)

    summary = summarize_results(results)
    save_json(summary, os.path.join(cfg.output_dir, "cv_summary.json"))

    plot_cv_metric_bars(results_df, cfg.output_dir)

    print("\n" + "=" * 100)
    print("5-Fold Cross Validation Results")
    print(results_df)
    print("\nMean ± Std")
    for k, v in summary.items():
        print(f"{k}: {v['mean']:.4f} ± {v['std']:.4f}")
    print("=" * 100)


if __name__ == "__main__":
    main()