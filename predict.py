"""
Galaxy Classification Predictor
Run inference on a single image (.png) or a triplet of raw FITS files (IR, VIS, UV),
with every trained model plus the ensemble vote.
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from preprocess import align_bands


def plot_prediction(pil_img, classes, ens_probs, per_model, out_file=None):
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1, 1.6]})
    ax0.imshow(pil_img)
    ax0.axis("off")
    ax0.set_title("Galaxy Image (R=IR, G=VIS, B=UV)", fontsize=11, fontweight="bold")

    rows = list(per_model.items())
    if len(rows) > 1:
        rows.append(("ensemble", ens_probs))
    height = 0.8 / len(rows)
    cmap = plt.cm.tab10
    y = np.arange(len(classes))
    for j, (label, p) in enumerate(rows):
        is_ens = label == "ensemble"
        bars = ax1.barh(y + j * height, np.asarray(p) * 100, height, label=label,
                        color="black" if is_ens else cmap(j % 10), edgecolor="black", alpha=0.9)
        if is_ens or len(rows) == 1:
            for b in bars:
                ax1.text(b.get_width() + 1, b.get_y() + b.get_height() / 2, f"{b.get_width():.1f}%",
                         va="center", fontsize=9, fontweight="bold")
    ax1.set_yticks(y + height * (len(rows) - 1) / 2)
    ax1.set_yticklabels([c.capitalize() for c in classes], fontsize=11)
    ax1.set_xlabel("Confidence (%)")
    ax1.set_xlim(0, 108)
    ax1.set_title("Classification", fontsize=11, fontweight="bold")
    ax1.grid(axis="x", linestyle="--", alpha=0.6)
    ax1.legend(fontsize=8, loc="lower right")

    fig.tight_layout()
    if out_file:
        fig.savefig(out_file, dpi=300)
        print(f"Saved prediction plot to: {out_file}")
    else:
        plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Predict galaxy morphology")
    parser.add_argument("--image", help="Path to galaxy image (.png/.jpg)")
    parser.add_argument("--ir", help="Path to IR FITS file")
    parser.add_argument("--vis", help="Path to Visible FITS file")
    parser.add_argument("--uv", help="Path to UV FITS file")
    parser.add_argument("--checkpoint", default="outputs",
                        help="outputs folder (ensemble), ensemble.json, or a single .pth file")
    parser.add_argument("--out", default="prediction.png", help="Output plot filename")
    args = parser.parse_args()

    from inference import Ensemble, discover_checkpoints
    paths, weights = discover_checkpoints(args.checkpoint)
    if not paths:
        raise FileNotFoundError(f"No checkpoints found at '{args.checkpoint}'. Train the model first!")
    ens = Ensemble(paths, weights)
    print(f"Loaded {len(ens.members)} model(s): {', '.join(ens.names)}")

    if args.image:
        img = Image.open(args.image).convert("RGB")
    elif args.ir and args.vis and args.uv:
        print("Aligning raw FITS bands...")
        img = align_bands(args.ir, args.vis, args.uv)
    else:
        raise ValueError("Please provide either --image or all three FITS files (--ir, --vis, --uv)")

    probs, per_model = ens.predict_pil(img)
    classes = ens.classes

    print("\n" + "=" * 52)
    print("GALAXY PREDICTION RESULT")
    print("=" * 52)
    if len(per_model) > 1:
        for name, p in per_model.items():
            print(f"  {name:<16}: {classes[int(p.argmax())].upper():<11} ({p.max() * 100:5.1f}%)")
        print("-" * 52)
    top = int(probs.argmax())
    label = "Ensemble" if len(per_model) > 1 else "Prediction"
    print(f"{label:<18}: {classes[top].upper()} ({probs[top] * 100:.2f}%)")
    for c, p in zip(classes, probs):
        print(f"  * {c.capitalize():<12}: {p * 100:6.2f}%")
    print("=" * 52 + "\n")

    plot_prediction(img, classes, probs, per_model, out_file=args.out)


if __name__ == "__main__":
    main()
