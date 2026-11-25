"""
Shared inference helpers: load one checkpoint or the whole multi-model ensemble written by train.py.

    paths, weights = discover_checkpoints("outputs")
    ens = Ensemble(paths, weights)
    probs, per_model = ens.predict_pil(img)        # ensemble probabilities + each model's own
"""

import os
import json
import glob
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from model import get_model
from dataset import get_transforms

DEFAULT_CLASSES = ["anomaly", "elliptical", "spiral"]


def torch_load(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:                      # older torch without weights_only
        return torch.load(path, map_location=device)


def _read_manifest(manifest_path):
    base = os.path.dirname(manifest_path)
    with open(manifest_path) as f:
        man = json.load(f)
    paths, weights = [], []
    for m in man.get("members", []):
        p = os.path.join(base, m["checkpoint"])
        if os.path.exists(p):
            paths.append(p)
            weights.append(float(m.get("weight", 1.0)))
    return paths, weights


def discover_checkpoints(path="outputs"):
    """Return (checkpoint_paths, weights). Accepts a .pth file, an ensemble.json, or an outputs folder."""
    if os.path.isfile(path):
        if path.endswith(".json"):
            return _read_manifest(path)
        return [path], [1.0]
    if os.path.isdir(path):
        manifest = os.path.join(path, "ensemble.json")
        if os.path.exists(manifest):
            paths, weights = _read_manifest(manifest)
            if paths:
                return paths, weights
        per_arch = sorted(glob.glob(os.path.join(path, "*", "best_model.pth")))
        if per_arch:
            return per_arch, [1.0] * len(per_arch)
        for name in ("best_galaxy_model.pth", "best_model.pth"):
            p = os.path.join(path, name)
            if os.path.exists(p):
                return [p], [1.0]
    return [], []


class Member:
    """One trained model plus its own preprocessing."""

    def __init__(self, path, device):
        ckpt = torch_load(path, device)
        self.path = path
        self.arch = ckpt.get("arch", "resnet18")
        self.name = self.arch
        self.classes = list(ckpt.get("class_names", DEFAULT_CLASSES))
        self.img_size = int(ckpt.get("img_size", 224))
        self.model = get_model(arch=self.arch, num_classes=len(self.classes), pretrained=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(device).eval()
        _, self.tf = get_transforms(self.img_size)
        self.info = {"arch": self.arch, "img_size": self.img_size, "epoch": ckpt.get("epoch"),
                     "best_val_acc": ckpt.get("best_acc")}


class Ensemble:
    """Soft-voting ensemble (weighted mean of softmax outputs). Works with a single model too."""

    def __init__(self, paths, weights=None, device=None):
        if not paths:
            raise FileNotFoundError("No checkpoints given - train first with `python train.py`.")
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.members = [Member(p, self.device) for p in paths]
        names = {}
        for m in self.members:                       # keep names unique
            names[m.name] = names.get(m.name, 0) + 1
            if names[m.name] > 1:
                m.name = f"{m.name}_{names[m.name]}"
        self.classes = self.members[0].classes
        for m in self.members:
            if m.classes != self.classes:
                raise ValueError(f"Class order mismatch: {m.path} has {m.classes}, expected {self.classes}")
        w = np.asarray(weights if weights else [1.0] * len(self.members), dtype=float)
        self.weights = w / w.sum()

    @property
    def names(self):
        return [m.name for m in self.members]

    def _member_probs(self, member, imgs):
        x = torch.stack([member.tf(im.convert("RGB")) for im in imgs]).to(self.device)
        with torch.no_grad():
            return F.softmax(member.model(x), dim=1).cpu().numpy()

    def predict_images(self, imgs):
        """imgs: list of PIL images -> (ensemble_probs (N,C), {model_name: probs (N,C)})."""
        per = {m.name: self._member_probs(m, imgs) for m in self.members}
        ens = sum(w * per[m.name] for w, m in zip(self.weights, self.members))
        return ens, per

    def predict_pil(self, img):
        ens, per = self.predict_images([img])
        return ens[0], {k: v[0] for k, v in per.items()}

    def predict_paths(self, paths, batch_size=32):
        ens_out, per_out = [], {n: [] for n in self.names}
        for i in range(0, len(paths), batch_size):
            imgs = [Image.open(p).convert("RGB") for p in paths[i:i + batch_size]]
            ens, per = self.predict_images(imgs)
            ens_out.append(ens)
            for n in per_out:
                per_out[n].append(per[n])
        if not ens_out:
            empty = np.zeros((0, len(self.classes)))
            return empty, {n: empty for n in self.names}
        return np.concatenate(ens_out), {n: np.concatenate(v) for n, v in per_out.items()}
