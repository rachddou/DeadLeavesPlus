"""Utilities for dataset similarity metrics: FID, Inception Score, Improved Precision & Recall.

Precision & Recall follow the kNN manifold algorithm from:
  "Improved Precision and Recall Metric for Assessing Generative Models"
  Kynkäänniemi et al., NeurIPS 2019
  (see submodule: improved-precision-and-recall-metric/)
"""

import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import inception_v3, Inception_V3_Weights
from PIL import Image
from tqdm import tqdm
from torch_fidelity import calculate_metrics


# ---------------------------------------------------------------------------
# Image dataset
# ---------------------------------------------------------------------------

_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}


class ImageFolder(Dataset):
    def __init__(self, root, transform=None):
        self.paths = sorted(
            p for p in (os.path.join(root, f) for f in os.listdir(root))
            if os.path.splitext(p)[1].lower() in _EXTENSIONS
        )
        if not self.paths:
            raise ValueError(f"No images found in {root!r}")
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img


# ---------------------------------------------------------------------------
# InceptionV3 feature extraction
# ---------------------------------------------------------------------------

_INCEPTION_TRANSFORM = transforms.Compose([
    transforms.Resize((299, 299), antialias=True),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


def build_inception_feature_extractor(device):
    model = inception_v3(weights=Inception_V3_Weights.DEFAULT)
    model.fc = torch.nn.Identity()
    model.aux_logits = False
    model.AuxLogits = None
    model.eval()
    return model.to(device)


@torch.no_grad()
def extract_features(image_dir, model, device, batch_size=64):
    """Return (N, 2048) InceptionV3 pool features for all images in image_dir."""
    dataset = ImageFolder(image_dir, transform=_INCEPTION_TRANSFORM)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=4, pin_memory=True)
    features = []
    for batch in tqdm(loader, desc=f"  {os.path.basename(image_dir)}", leave=False):
        features.append(model(batch.to(device)).cpu())
    return torch.cat(features, dim=0).numpy()


# ---------------------------------------------------------------------------
# Improved Precision & Recall
# ---------------------------------------------------------------------------

def _knn_radii(features: np.ndarray, k: int, batch_size: int = 5000) -> np.ndarray:
    """Return the k-th nearest-neighbour distance for each point in features."""
    N = features.shape[0]
    feats = torch.from_numpy(features).float()
    radii = torch.zeros(N)
    for i in range(0, N, batch_size):
        batch = feats[i : i + batch_size]
        sq_dist = (batch.unsqueeze(1) - feats.unsqueeze(0)).pow(2).sum(dim=2)
        # k+1 because the point itself has distance 0
        kth_sq, _ = torch.kthvalue(sq_dist, k + 1, dim=1)
        radii[i : i + batch_size] = kth_sq.sqrt()
    return radii.numpy()


def _in_manifold(eval_features: np.ndarray, ref_features: np.ndarray,
                 ref_radii: np.ndarray, batch_size: int = 5000) -> np.ndarray:
    """Boolean mask: True if each eval point lies within any ref hypersphere."""
    N_eval = eval_features.shape[0]
    eval_t = torch.from_numpy(eval_features).float()
    ref_t = torch.from_numpy(ref_features).float()
    radii_t = torch.from_numpy(ref_radii).float()
    in_mf = torch.zeros(N_eval, dtype=torch.bool)
    for i in range(0, N_eval, batch_size):
        batch = eval_t[i : i + batch_size]
        dists = (batch.unsqueeze(1) - ref_t.unsqueeze(0)).pow(2).sum(dim=2).sqrt()
        in_mf[i : i + batch_size] = (dists <= radii_t.unsqueeze(0)).any(dim=1)
    return in_mf.numpy()


def improved_precision_recall(real_features: np.ndarray, fake_features: np.ndarray,
                               k: int = 3) -> tuple[float, float]:
    """Compute Improved Precision and Recall (Kynkäänniemi et al. 2019).

    Precision = fraction of generated samples inside the real manifold.
    Recall    = fraction of real samples inside the generated manifold.
    """
    print(f"  Building real manifold (k={k})...")
    real_radii = _knn_radii(real_features, k=k)
    print(f"  Building fake manifold (k={k})...")
    fake_radii = _knn_radii(fake_features, k=k)
    precision = _in_manifold(fake_features, real_features, real_radii).mean()
    recall = _in_manifold(real_features, fake_features, fake_radii).mean()
    return float(precision), float(recall)


def _pairwise_distances(a: np.ndarray, b: np.ndarray, batch_size: int = 2000) -> np.ndarray:
    """Return L2 distance matrix of shape (len(a), len(b)), computed in batches."""
    a_t = torch.from_numpy(a).float()
    b_t = torch.from_numpy(b).float()
    N = len(a)
    dists = torch.zeros(N, len(b))
    for i in range(0, N, batch_size):
        batch = a_t[i : i + batch_size]
        dists[i : i + batch_size] = (
            (batch.unsqueeze(1) - b_t.unsqueeze(0)).pow(2).sum(dim=2).sqrt()
        )
    return dists.numpy()


def precision_recall_curve(
    real_features: np.ndarray,
    fake_features: np.ndarray,
    k_max: int = 10,
) -> dict:
    """Compute Improved Precision & Recall for k in [1, k_max].

    Pairwise distances are computed once and reused across all k values.

    Returns a dict with keys 'k_values', 'precision', 'recall'.
    """
    print("  Computing real-real distances...")
    real_sorted = np.sort(_pairwise_distances(real_features, real_features), axis=1)[:, : k_max + 1]
    print("  Computing fake-fake distances...")
    fake_sorted = np.sort(_pairwise_distances(fake_features, fake_features), axis=1)[:, : k_max + 1]
    print("  Computing real-fake cross distances...")
    cross = _pairwise_distances(real_features, fake_features)
    # cross[i, j] = dist(real_i, fake_j)

    precisions, recalls = [], []
    for k in range(1, k_max + 1):
        # Precision: fake_j in real manifold iff ∃i: cross[i,j] ≤ real_sorted[i, k]
        precision = (cross <= real_sorted[:, k : k + 1]).any(axis=0).mean()
        # Recall: real_i in fake manifold iff ∃j: cross[i,j] ≤ fake_sorted[j, k]
        recall = (cross <= fake_sorted[:, k]).any(axis=1).mean()
        precisions.append(float(precision))
        recalls.append(float(recall))

    return {
        "k_values": list(range(1, k_max + 1)),
        "precision": precisions,
        "recall": recalls,
    }


def plot_precision_recall_curve(curve: dict, save_path: str = None) -> None:
    """Plot precision & recall vs k (left) and in P&R space (right)."""
    import matplotlib.pyplot as plt

    k = curve["k_values"]
    precision = curve["precision"]
    recall = curve["recall"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: both metrics vs k on the same axis
    axes[0].plot(k, precision, "b-o", label="Precision")
    axes[0].plot(k, recall, "r-o", label="Recall")
    axes[0].set_xlabel("k (neighbourhood size)")
    axes[0].set_ylabel("Score")
    axes[0].set_title("Precision & Recall vs k")
    axes[0].set_ylim(0, 1)
    axes[0].legend()
    axes[0].grid(True, alpha=0.4)

    # Right: P&R space curve with k labels
    axes[1].plot(recall, precision, "g-o")
    for i, ki in enumerate(k):
        axes[1].annotate(
            f"k={ki}", (recall[i], precision[i]),
            textcoords="offset points", xytext=(5, 5), fontsize=8,
        )
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall space")
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1)
    axes[1].grid(True, alpha=0.4)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Plot saved to {save_path}")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# FID & Inception Score
# ---------------------------------------------------------------------------

def compute_fid_and_is(real_dir: str, fake_dir: str) -> tuple[float, float, float]:
    """Return (FID, IS_mean, IS_std) via torch-fidelity."""
    metrics = calculate_metrics(
        input1=fake_dir,
        input2=real_dir,
        cuda=torch.cuda.is_available(),
        isc=True,
        fid=True,
        verbose=False,
    )
    return (
        metrics["frechet_inception_distance"],
        metrics["inception_score_mean"],
        metrics["inception_score_std"],
    )
