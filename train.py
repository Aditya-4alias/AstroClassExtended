"""
Train several galaxy classifiers on the multi-wavelength composites and combine them into an ensemble.

Multi-modal set-up (each galaxy = 3 modalities: IR, VIS, UV stored as the R, G, B channels):
  resnet18         early fusion - the 3 bands as one 3-channel image
  multiband        late fusion  - one ResNet-18 stream per band, features concatenated
  efficientnet_b0  early fusion - a different architecture family
The softmax outputs of the models are averaged (soft voting) into an ensemble.

    python train.py                                  # all three models + ensemble
    python train.py --archs resnet18 multiband       # just two
    python train.py --benchmark                      # times one batch per model, no training

Outputs (in --out_dir):
    <arch>/best_model.pth, last_checkpoint.pth, reports, confusion matrices, ROC / PR curves,
    <arch>/scatter_train_val.png        train-vs-val accuracy / loss per epoch
    <arch>/scatter_embedding_<split>.png  PCA + t-SNE of the learned features (misclassified circled)
    <arch>/scatter_probs_<split>.png      class-probability triangle (3 classes)
    ensemble/...                        same reports for the averaged ensemble
    ensemble.json, model_comparison.png, summary.json, best_galaxy_model.pth (best single model)
"""

import os
import json
import time
import shutil
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.preprocessing import label_binarize

from dataset import get_loaders
from model import get_model, extract_features, ARCHS, DEFAULT_ARCHS, DESCRIPTIONS

PALETTE = {"spiral": "#2b5c8f", "elliptical": "#d97724", "anomaly": "#bd3333"}


def class_color(name, i):
    return PALETTE.get(name, plt.cm.tab10(i % 10))


# --------------------------------------------------------------------------------------
# Training / evaluation primitives
# --------------------------------------------------------------------------------------
def get_class_weights(dataset, device):
    targets = [s[1] for s in dataset.samples]
    classes, counts = np.unique(targets, return_counts=True)
    weights = len(targets) / (len(classes) * counts.astype(np.float32))
    return torch.tensor(weights, dtype=torch.float, device=device)


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    loss_sum, correct, total = 0.0, 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        loss_sum += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)

    return loss_sum / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            probs = F.softmax(out, dim=1)

            loss_sum += loss.item() * x.size(0)
            preds = probs.argmax(1)
            correct += (preds == y).sum().item()
            total += y.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    probs_arr = np.concatenate(all_probs, axis=0) if all_probs else np.zeros((0, 0))
    return loss_sum / total, correct / total, np.array(all_preds), np.array(all_labels), probs_arr


def extract_embeddings(model, loader, device):
    model.eval()
    feats = []
    with torch.no_grad():
        for x, _ in loader:
            feats.append(extract_features(model, x.to(device)).cpu().numpy())
    return np.concatenate(feats, axis=0)


# --------------------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------------------
def plot_and_save_roc(y_true, y_probs, classes, out_path, title_prefix=""):
    n_classes = len(classes)
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))
    if y_bin.shape[1] == 1 and n_classes == 2:
        y_bin = np.hstack([1 - y_bin, y_bin])

    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    auc_scores = {}

    for i, cname in enumerate(classes):
        if np.sum(y_bin[:, i]) == 0 or np.sum(y_bin[:, i]) == len(y_true):
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_probs[:, i])
        roc_auc = auc(fpr, tpr)
        auc_scores[cname] = float(roc_auc)
        ax.plot(fpr, tpr, lw=2, label=f"{cname.capitalize()} (AUC = {roc_auc:.3f})")

    try:
        fpr_micro, tpr_micro, _ = roc_curve(y_bin.ravel(), y_probs.ravel())
        micro_auc = auc(fpr_micro, tpr_micro)
        auc_scores["micro"] = float(micro_auc)
        ax.plot(fpr_micro, tpr_micro, linestyle="--", lw=2, color="black",
                label=f"Micro-avg (AUC = {micro_auc:.3f})")
    except Exception:
        pass

    valid_aucs = [v for k, v in auc_scores.items() if k != "micro"]
    if valid_aucs:
        auc_scores["macro"] = float(np.mean(valid_aucs))

    ax.plot([0, 1], [0, 1], "k:", lw=1, alpha=0.6)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve {title_prefix}".strip())
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return auc_scores


def plot_and_save_pr(y_true, y_probs, classes, out_path, title_prefix=""):
    n_classes = len(classes)
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))
    if y_bin.shape[1] == 1 and n_classes == 2:
        y_bin = np.hstack([1 - y_bin, y_bin])

    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    ap_scores = {}

    for i, cname in enumerate(classes):
        if np.sum(y_bin[:, i]) == 0:
            continue
        prec, rec, _ = precision_recall_curve(y_bin[:, i], y_probs[:, i])
        ap = average_precision_score(y_bin[:, i], y_probs[:, i])
        ap_scores[cname] = float(ap)
        ax.plot(rec, prec, lw=2, label=f"{cname.capitalize()} (AP = {ap:.3f})")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curve {title_prefix}".strip())
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return ap_scores


def plot_embedding_scatter(feats, labels, preds, classes, out_path, title=""):
    """2-D scatter of the learned features (PCA, and t-SNE when there are enough samples).
    Points are coloured by TRUE class; misclassified galaxies get a black ring."""
    from sklearn.decomposition import PCA
    n = len(labels)
    if n < 4 or feats.shape[1] < 2:
        return False
    panels = [("PCA", PCA(n_components=2, random_state=0).fit_transform(feats))]
    if n >= 15:
        try:
            from sklearn.manifold import TSNE
            perp = float(max(2, min(30, (n - 1) // 3)))
            panels.append(("t-SNE", TSNE(n_components=2, perplexity=perp, init="pca",
                                         random_state=0).fit_transform(feats)))
        except Exception as e:
            print(f"  [WARN] t-SNE skipped: {e}")

    fig, axes = plt.subplots(1, len(panels), figsize=(6.2 * len(panels), 5.2), squeeze=False)
    wrong = labels != preds
    for ax, (pname, xy) in zip(axes[0], panels):
        for i, c in enumerate(classes):
            m = labels == i
            if m.any():
                ax.scatter(xy[m, 0], xy[m, 1], s=40, alpha=0.85, color=class_color(c, i),
                           edgecolor="white", linewidth=0.5, label=f"{c.capitalize()} (n={int(m.sum())})")
        if wrong.any():
            ax.scatter(xy[wrong, 0], xy[wrong, 1], s=150, facecolors="none", edgecolors="black",
                       linewidths=1.5, label=f"Misclassified ({int(wrong.sum())})")
        ax.set_title(f"{pname} of learned features")
        ax.set_xlabel(f"{pname} 1")
        ax.set_ylabel(f"{pname} 2")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(fontsize=8, loc="best")
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return True


def plot_probability_scatter(probs, labels, preds, classes, out_path, title=""):
    """3-class probability simplex: every galaxy is one point; corners = confident for that class,
    the middle = undecided. Coloured by true class, misclassified circled."""
    if len(classes) != 3 or len(labels) == 0:
        return False
    verts = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3) / 2]])
    xy = probs @ verts
    rng = np.random.RandomState(0)
    xy = xy + rng.normal(scale=0.004, size=xy.shape)          # tiny jitter so stacked points stay visible

    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    tri = np.vstack([verts, verts[:1]])
    ax.plot(tri[:, 0], tri[:, 1], color="black", lw=1.2)
    for v, c in zip(verts, classes):
        dx = -0.04 if v[0] < 0.5 else (0.04 if v[0] > 0.5 else 0)
        dy = -0.05 if v[1] == 0 else 0.04
        ax.text(v[0] + dx, v[1] + dy, c.capitalize(), ha="center", va="center", fontweight="bold")
    for i, c in enumerate(classes):
        m = labels == i
        if m.any():
            ax.scatter(xy[m, 0], xy[m, 1], s=42, alpha=0.8, color=class_color(c, i), edgecolor="white",
                       linewidth=0.5, label=f"True {c.capitalize()} (n={int(m.sum())})")
    wrong = labels != preds
    if wrong.any():
        ax.scatter(xy[wrong, 0], xy[wrong, 1], s=150, facecolors="none", edgecolors="black",
                   linewidths=1.5, label=f"Misclassified ({int(wrong.sum())})")
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title or "Class probabilities", fontweight="bold", pad=22)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return True


def plot_train_val_scatter(history, out_path, arch=""):
    """Each point = one epoch (colour = epoch). Points below the diagonal in the accuracy panel
    mean val < train, i.e. overfitting; far above it in the loss panel means the same."""
    n = len(history["train_loss"])
    if n < 2:
        return False
    ep = np.arange(1, n + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))
    for ax, (title, a, b, scale) in zip(axes, [("Accuracy (%)", "train_acc", "val_acc", 100.0),
                                              ("Loss", "train_loss", "val_loss", 1.0)]):
        x = np.array(history[a]) * scale
        y = np.array(history[b]) * scale
        sc = ax.scatter(x, y, c=ep, cmap="viridis", s=55, edgecolor="black", linewidth=0.5)
        lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
        pad = 0.05 * (hi - lo + 1e-9)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=1, alpha=0.6, label="train = val")
        ax.set_xlabel(f"Train {title}")
        ax.set_ylabel(f"Validation {title}")
        ax.set_title(f"{title}: train vs validation")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(loc="best", fontsize=8)
        fig.colorbar(sc, ax=ax, label="Epoch")
    fig.suptitle(f"Per-epoch scatter {('- ' + arch) if arch else ''}".strip(), fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return True


def plot_model_comparison(table, out_path):
    """table: {name: {"val": acc, "test": acc or None}}"""
    names = list(table)
    has_test = any(table[n].get("test") is not None for n in names)
    x = np.arange(len(names))
    w = 0.38 if has_test else 0.6
    fig, ax = plt.subplots(figsize=(max(6, 1.7 * len(names)), 4.5))
    bars = [ax.bar(x - (w / 2 if has_test else 0), [100 * table[n]["val"] for n in names], w,
                   label="Validation", color="#2b5c8f", edgecolor="black")]
    if has_test:
        bars.append(ax.bar(x + w / 2, [100 * (table[n].get("test") or 0) for n in names], w,
                           label="Test", color="#d97724", edgecolor="black"))
    for group in bars:
        for b in group:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.5, f"{b.get_height():.1f}",
                    ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0, 108)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Model comparison")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_training_curves(history, out_path):
    epochs_range = range(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(10, 4))
    for i, (title, a, b) in enumerate([("Loss", "train_loss", "val_loss"), ("Accuracy", "train_acc", "val_acc")]):
        plt.subplot(1, 2, i + 1)
        plt.plot(epochs_range, history[a], "b-", label="Train")
        plt.plot(epochs_range, history[b], "r-", label="Val")
        plt.title(title); plt.xlabel("Epoch"); plt.legend(); plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# --------------------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------------------
def report_split(name, out_dir, labs, preds, probs, loss, classes, feats=None, tag=""):
    """Write report, confusion matrix, ROC, PR, metrics.json and the scatter plots for one split."""
    os.makedirs(out_dir, exist_ok=True)
    names = [c.capitalize() for c in classes]
    label_ids = list(range(len(classes)))
    acc = float((labs == preds).mean())

    rep_text = classification_report(labs, preds, labels=label_ids, target_names=names, zero_division=0)
    rep_dict = classification_report(labs, preds, labels=label_ids, target_names=names,
                                     zero_division=0, output_dict=True)
    print(f"\n=== {tag}{name.upper()} set: loss {loss:.4f}, acc {acc*100:.1f}% ({len(labs)} images) ===")
    print(rep_text)

    with open(os.path.join(out_dir, f"classification_report_{name}.txt"), "w") as f:
        f.write(f"{name.upper()} set | loss: {loss:.4f} | acc: {acc*100:.2f}% | n: {len(labs)}\n\n")
        f.write(rep_text)

    cm = confusion_matrix(labs, preds, labels=label_ids)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=names).plot(cmap="Blues", ax=ax, colorbar=False)
    plt.title(f"Confusion Matrix ({name}) {tag}".strip())
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"confusion_matrix_{name}.png"), dpi=200)
    plt.close()

    auc_scores = plot_and_save_roc(labs, probs, classes, os.path.join(out_dir, f"roc_curve_{name}.png"),
                                   title_prefix=f"({name}) {tag}".strip())
    ap_scores = plot_and_save_pr(labs, probs, classes, os.path.join(out_dir, f"pr_curve_{name}.png"),
                                 title_prefix=f"({name}) {tag}".strip())

    plot_probability_scatter(probs, labs, preds, classes,
                             os.path.join(out_dir, f"scatter_probs_{name}.png"),
                             title=f"Class probabilities ({name}) {tag}".strip())
    if feats is not None:
        plot_embedding_scatter(feats, labs, preds, classes,
                               os.path.join(out_dir, f"scatter_embedding_{name}.png"),
                               title=f"Feature space ({name}) {tag}".strip())

    with open(os.path.join(out_dir, f"metrics_{name}.json"), "w") as f:
        json.dump({"split": name, "loss": float(loss), "accuracy": acc, "num_samples": int(len(labs)),
                   "roc_auc": auc_scores, "pr_auc": ap_scores, "classification_report": rep_dict}, f, indent=2)
    return acc


def run_benchmark(arch, model, loader, val_loader, criterion, optimizer, device, epochs, steps=6):
    model.train()
    it, times = iter(loader), []
    for _ in range(steps):
        t0 = time.time()
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(loader)
            x, y = next(it)
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        criterion(model(x), y).backward()
        optimizer.step()
        times.append(time.time() - t0)
    per_batch = float(np.mean(times[1:]))
    bs = loader.batch_size
    t_train = per_batch * len(loader)

    t0, n_val = time.time(), 0
    model.eval()
    with torch.no_grad():
        for i, (x, _) in enumerate(val_loader):
            model(x.to(device))
            n_val += 1
            if i >= 2:
                break
    t_val = (time.time() - t0) / max(1, n_val) * len(val_loader)

    epoch_s = t_train + t_val
    print(f"\n=== BENCHMARK [{arch}] ===")
    print(f"{per_batch:.2f} s per batch of {bs}  ({per_batch / bs * 1000:.0f} ms per image)")
    print(f"~{epoch_s:.0f} s per epoch  ({len(loader.dataset)} train / {len(val_loader.dataset)} val images)")
    print(f"~{epoch_s * epochs / 60:.0f} min for {epochs} epochs")
    return epoch_s * epochs / 60


# --------------------------------------------------------------------------------------
# One model
# --------------------------------------------------------------------------------------
def make_criterion(args, train_loader, device):
    if args.weighted:
        weights = get_class_weights(train_loader.dataset, device)
        print(f"Class weights: {weights.tolist()}")
        return nn.CrossEntropyLoss(weight=weights)
    return nn.CrossEntropyLoss()


def train_one(arch, args, train_loader, val_loader, classes, device, criterion):
    """Train a single architecture. Returns {"best_path", "best_acc"} (None for --benchmark)."""
    arch_dir = os.path.join(args.out_dir, arch)
    os.makedirs(arch_dir, exist_ok=True)

    model = get_model(arch=arch, num_classes=len(classes)).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model: {DESCRIPTIONS.get(arch, arch)}  ({n_params:.1f} M parameters)")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    if args.benchmark:
        run_benchmark(arch, model, train_loader, val_loader, criterion, optimizer, device, args.epochs)
        return None

    best_path = os.path.join(arch_dir, "best_model.pth")
    last_path = os.path.join(arch_dir, "last_checkpoint.pth")
    best_acc, best_loss, bad_epochs, start_epoch = 0.0, float("inf"), 0, 1
    best_acc_loss = float("inf")
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    if not args.no_resume and os.path.exists(last_path):
        ck = torch.load(last_path, map_location=device, weights_only=False)
        if ck.get("class_names") == classes and ck.get("arch") == arch:
            model.load_state_dict(ck["model_state_dict"])
            optimizer.load_state_dict(ck["optimizer_state_dict"])
            scheduler.load_state_dict(ck["scheduler_state_dict"])
            start_epoch = ck["epoch"] + 1
            best_acc, best_loss = ck["best_acc"], ck["best_loss"]
            best_acc_loss = ck.get("best_acc_loss", float("inf"))
            bad_epochs, history = ck.get("bad_epochs", 0), ck["history"]
            print(f"Resumed from epoch {ck['epoch']} (best val acc {best_acc*100:.1f}%)")

    t_start = time.time()
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc, _, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step(va_loss)

        for k, v in zip(history, (tr_loss, tr_acc, va_loss, va_acc)):
            history[k].append(v)

        dt = time.time() - t0
        eta = dt * (args.epochs - epoch) / 60
        print(f"[{arch}] Epoch [{epoch:02d}/{args.epochs:02d}] "
              f"Train Loss: {tr_loss:.4f} Acc: {tr_acc*100:.1f}% | "
              f"Val Loss: {va_loss:.4f} Acc: {va_acc*100:.1f}% | {dt:.0f}s (ETA {eta:.0f} min)")

        if va_acc > best_acc or (va_acc == best_acc and va_loss < best_acc_loss):
            best_acc, best_acc_loss = va_acc, va_loss
            torch.save({
                "epoch": epoch, "model_state_dict": model.state_dict(), "arch": arch,
                "class_names": classes, "best_acc": best_acc, "img_size": args.img_size,
            }, best_path)
            print(f"  * New best model saved ({va_acc*100:.1f}%)")

        if va_loss < best_loss - 1e-4:
            best_loss, bad_epochs = va_loss, 0
        else:
            bad_epochs += 1

        torch.save({
            "epoch": epoch, "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "arch": arch, "class_names": classes, "img_size": args.img_size,
            "best_acc": best_acc, "best_loss": best_loss, "best_acc_loss": best_acc_loss,
            "bad_epochs": bad_epochs, "history": history,
        }, last_path + ".tmp")
        os.replace(last_path + ".tmp", last_path)

        if args.patience and bad_epochs >= args.patience:
            print(f"Early stopping: val loss did not improve for {args.patience} epochs.")
            break

    print(f"[{arch}] done in {(time.time() - t_start) / 60:.1f} min. Best validation accuracy: {best_acc*100:.1f}%")

    if history["train_loss"]:
        plot_training_curves(history, os.path.join(arch_dir, "training_curves.png"))
        plot_train_val_scatter(history, os.path.join(arch_dir, "scatter_train_val.png"), arch=arch)

    return {"best_path": best_path, "best_acc": best_acc}


# --------------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train Galaxy Classifiers (multi-model / multi-modal)")
    parser.add_argument("--data_dir", default="processed_dataset", help="Path to data")
    parser.add_argument("--out_dir", default="outputs", help="Output directory")
    parser.add_argument("--archs", nargs="+", default=DEFAULT_ARCHS, choices=ARCHS,
                        help="models to train (2-3 recommended), default: %(default)s")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--weighted", action="store_true", help="Use weighted loss for balance")
    parser.add_argument("--weight_by_val", action="store_true",
                        help="weight each model's vote in the ensemble by its validation accuracy")
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--no_resume", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    archs = list(dict.fromkeys(args.archs))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(args.threads or min(6, os.cpu_count() or 4))
    print(f"Device: {device}" + (f" ({torch.get_num_threads()} threads)" if device.type == "cpu" else ""))
    print(f"Models: {', '.join(archs)}")

    os.makedirs(args.out_dir, exist_ok=True)
    train_loader, val_loader, test_loader, classes = get_loaders(
        args.data_dir, batch_size=args.batch_size, img_size=args.img_size, num_workers=args.num_workers
    )
    print(f"Classes: {classes} | train {len(train_loader.dataset)}, val {len(val_loader.dataset)}, "
          f"test {len(test_loader.dataset) if test_loader else 0}")

    with open(os.path.join(args.out_dir, "classes.json"), "w") as f:
        json.dump(classes, f, indent=2)

    criterion = make_criterion(args, train_loader, device)

    # ---- 1. train every model -------------------------------------------------------
    results = {}
    for arch in archs:
        print(f"\n{'=' * 70}\n  {arch}\n{'=' * 70}")
        res = train_one(arch, args, train_loader, val_loader, classes, device, criterion)
        if res:
            results[arch] = res
    if args.benchmark:
        return

    # ---- 2. evaluate every model on val / test --------------------------------------
    split_loaders = {"val": val_loader}
    if test_loader is not None:
        split_loaders["test"] = test_loader
    else:
        print("\nNo test/ folder found - skipping test evaluation.")

    per_split = {s: {} for s in split_loaders}
    table = {}
    for arch in archs:
        ckpt = torch.load(results[arch]["best_path"], map_location=device, weights_only=False)
        model = get_model(arch=arch, num_classes=len(classes), pretrained=False).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        table[arch] = {}
        for s, loader in split_loaders.items():
            loss, acc, preds, labs, probs = evaluate(model, loader, criterion, device)
            feats = extract_embeddings(model, loader, device)
            report_split(s, os.path.join(args.out_dir, arch), labs, preds, probs, loss, classes,
                         feats=feats, tag=f"[{arch}] ")
            per_split[s][arch] = {"probs": probs, "labs": labs}
            table[arch][s] = acc
        del model

    # ---- 3. ensemble (soft voting) --------------------------------------------------
    w = np.array([results[a]["best_acc"] for a in archs], dtype=float) if args.weight_by_val \
        else np.ones(len(archs))
    w = w / w.sum()
    if len(archs) > 1:
        table["ensemble"] = {}
        for s in split_loaders:
            probs = sum(wi * per_split[s][a]["probs"] for wi, a in zip(w, archs))
            labs = per_split[s][archs[0]]["labs"]
            preds = probs.argmax(1)
            loss = float(-np.mean(np.log(np.clip(probs[np.arange(len(labs)), labs], 1e-8, 1.0))))
            table["ensemble"][s] = report_split(s, os.path.join(args.out_dir, "ensemble"), labs, preds,
                                                probs, loss, classes, tag="[ensemble] ")

    manifest = {
        "classes": classes,
        "members": [{"arch": a, "checkpoint": f"{a}/best_model.pth", "weight": float(wi),
                     "val_acc": float(results[a]["best_acc"])} for a, wi in zip(archs, w)],
    }
    with open(os.path.join(args.out_dir, "ensemble.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    # ---- 4. comparison + a single-file checkpoint for older tools -------------------
    best_arch = max(archs, key=lambda a: results[a]["best_acc"])
    shutil.copy(results[best_arch]["best_path"], os.path.join(args.out_dir, "best_galaxy_model.pth"))

    cmp_table = {k: {"val": v.get("val", 0.0), "test": v.get("test")} for k, v in table.items()}
    plot_model_comparison(cmp_table, os.path.join(args.out_dir, "model_comparison.png"))
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump({"classes": classes, "accuracy": table, "best_single_model": best_arch}, f, indent=2)

    print("\n" + "=" * 70)
    print(f"{'model':<18}{'val acc':>10}{'test acc':>10}")
    for name, v in table.items():
        t = f"{v['test'] * 100:.1f}%" if "test" in v else "-"
        print(f"{name:<18}{v['val'] * 100:>9.1f}%{t:>10}")
    print("=" * 70)
    print(f"Best single model on validation: {best_arch} (copied to {args.out_dir}/best_galaxy_model.pth)")
    print(f"Saved models, reports, scatter plots and the ensemble manifest to {args.out_dir}/")


if __name__ == "__main__":
    main()
