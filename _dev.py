"""Development script — staircase point cloud segmentation pipeline.

Validates all functions end-to-end before porting to the final notebook.
Outputs figures to ./figures/ and metrics to ./data/metrics.json.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN

HERE = Path(__file__).resolve().parent
FIG_DIR = HERE / "figures"
DATA_DIR = HERE / "data"
FIG_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

RNG = np.random.default_rng(42)

# ---------- Geometry constants ----------
N_STEPS = 4
WIDTH = 2.5          # x-axis  [m]
DEPTH = 0.30         # tread depth (y-axis per step) [m]
HEIGHT = 0.18        # rise per step (z) [m]
SIGMA_Z_TREAD = 0.02
SIGMA_Y_RISER = 0.005

LABEL_TREAD = 0   # dapat dipijak
LABEL_RISER = 1
LABEL_OTHER = 2


# =====================================================================
# 1. POINT CLOUD GENERATION
# =====================================================================
def gen_tread(i: int, n: int, rng: np.random.Generator) -> np.ndarray:
    """Tread (horizontal surface) for step i."""
    x = rng.uniform(0.0, WIDTH, n)
    y = rng.uniform(i * DEPTH, (i + 1) * DEPTH, n)
    z = i * HEIGHT + rng.normal(0.0, SIGMA_Z_TREAD, n)
    return np.column_stack([x, y, z])


def gen_riser(i: int, n: int, rng: np.random.Generator) -> np.ndarray:
    """Riser (vertical surface) at the front of step i (between step i-1 and i).

    For i = 0 we synthesize the floor-to-first-step riser of height HEIGHT.
    For i >= 1 the riser spans z in [(i-1)*H, i*H] at y = i*D.
    """
    x = rng.uniform(0.0, WIDTH, n)
    y = i * DEPTH + rng.normal(0.0, SIGMA_Y_RISER, n)
    z_lo = max(0.0, (i - 1) * HEIGHT) if i >= 1 else 0.0
    z_hi = i * HEIGHT if i >= 1 else HEIGHT
    z = rng.uniform(z_lo, z_hi, n)
    return np.column_stack([x, y, z])


def add_lidar_noise(pts: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Add isotropic Gaussian noise (sensor noise)."""
    return pts + rng.normal(0.0, sigma, pts.shape)


def add_outliers(pts: np.ndarray, ratio: float, rng: np.random.Generator) -> np.ndarray:
    """Random outliers inside an enlarged bounding volume."""
    n_out = int(len(pts) * ratio)
    if n_out == 0:
        return pts
    x = rng.uniform(-0.2, WIDTH + 0.2, n_out)
    y = rng.uniform(-0.2, (N_STEPS + 1) * DEPTH + 0.2, n_out)
    z = rng.uniform(-0.1, N_STEPS * HEIGHT + 0.2, n_out)
    out = np.column_stack([x, y, z])
    return np.vstack([pts, out]), n_out


def generate_staircase(
    n_per_tread: int = 3500,
    n_per_riser: int = 1800,
    lidar_sigma: float = 0.005,
    outlier_ratio: float = 0.02,
    rng: np.random.Generator | None = None,
):
    """Generate the 4-step staircase point cloud with ground-truth labels."""
    rng = rng or np.random.default_rng(0)
    pieces = []
    labels = []
    for i in range(N_STEPS):
        t = gen_tread(i, n_per_tread, rng)
        r = gen_riser(i, n_per_riser, rng)
        pieces.append(t); labels.append(np.full(len(t), LABEL_TREAD))
        pieces.append(r); labels.append(np.full(len(r), LABEL_RISER))

    pc = np.vstack(pieces)
    lab = np.concatenate(labels)

    # add LiDAR noise to surface points
    pc = add_lidar_noise(pc, lidar_sigma, rng)

    # add outliers (label OTHER)
    pc_out, n_out = add_outliers(pc, outlier_ratio, rng)
    lab_out = np.concatenate([lab, np.full(n_out, LABEL_OTHER)])

    # shuffle
    idx = rng.permutation(len(pc_out))
    return pc_out[idx], lab_out[idx]


# =====================================================================
# 2. SEGMENTATION METHODS
# =====================================================================

# ---- Method 1: iterative RANSAC plane fitting (horizontal / vertical planes) ----
def ransac_plane(points: np.ndarray, threshold: float, n_iter: int,
                 rng: np.random.Generator, target: str = "horizontal",
                 min_cos: float = 0.95) -> tuple[np.ndarray, np.ndarray]:
    """Constrained RANSAC.

    target = "horizontal": only consider planes with |n_z| >= min_cos.
    target = "vertical":   only consider planes with |n_z| <= 1 - min_cos.
    """
    best_mask = np.zeros(len(points), dtype=bool)
    best_plane = None
    for _ in range(n_iter):
        idx = rng.choice(len(points), 3, replace=False)
        p0, p1, p2 = points[idx]
        v1 = p1 - p0; v2 = p2 - p0
        n = np.cross(v1, v2)
        norm = np.linalg.norm(n)
        if norm < 1e-9:
            continue
        n = n / norm
        nz = abs(n[2])
        if target == "horizontal" and nz < min_cos:
            continue
        if target == "vertical" and nz > 1 - min_cos:
            continue
        d = -np.dot(n, p0)
        dist = np.abs(points @ n + d)
        mask = dist < threshold
        if mask.sum() > best_mask.sum():
            best_mask = mask
            best_plane = np.array([n[0], n[1], n[2], d])
    return best_plane, best_mask


def ransac_treads(points: np.ndarray, n_tread: int = 4, n_riser: int = 4,
                  thr_tread: float = 0.04, thr_riser: float = 0.025,
                  n_iter: int = 600, min_inliers: int = 200,
                  min_cos: float = 0.95) -> np.ndarray:
    """Extract horizontal tread planes, then vertical riser planes."""
    rng = np.random.default_rng(123)
    remaining_idx = np.arange(len(points))
    pred = np.full(len(points), LABEL_OTHER)

    for _ in range(n_tread):
        if len(remaining_idx) < min_inliers:
            break
        sub = points[remaining_idx]
        plane, mask = ransac_plane(sub, thr_tread, n_iter, rng,
                                   target="horizontal", min_cos=min_cos)
        if plane is None or mask.sum() < min_inliers:
            break
        pred[remaining_idx[mask]] = LABEL_TREAD
        remaining_idx = remaining_idx[~mask]

    for _ in range(n_riser):
        if len(remaining_idx) < min_inliers:
            break
        sub = points[remaining_idx]
        plane, mask = ransac_plane(sub, thr_riser, n_iter, rng,
                                   target="vertical", min_cos=min_cos)
        if plane is None or mask.sum() < min_inliers:
            break
        pred[remaining_idx[mask]] = LABEL_RISER
        remaining_idx = remaining_idx[~mask]
    return pred


# ---- Method 2: PCA + normal vector analysis ----
def estimate_normals(points: np.ndarray, k: int = 20) -> np.ndarray:
    """Per-point normal via local PCA (smallest eigenvector)."""
    tree = cKDTree(points)
    _, idx = tree.query(points, k=k)
    normals = np.zeros_like(points)
    for i in range(len(points)):
        nb = points[idx[i]]
        nb_c = nb - nb.mean(axis=0)
        cov = nb_c.T @ nb_c / max(len(nb) - 1, 1)
        vals, vecs = np.linalg.eigh(cov)
        normals[i] = vecs[:, 0]  # eigenvector of smallest eigenvalue
    # orient so that nz >= 0 (consistent sign)
    flip = normals[:, 2] < 0
    normals[flip] *= -1
    return normals


def pca_normal_classify(points: np.ndarray, k: int = 20,
                        tread_cos: float = 0.80,
                        riser_cos: float = 0.30) -> tuple[np.ndarray, np.ndarray]:
    """Classify based on |n_z|: tread if near 1, riser if near 0, else other."""
    n = estimate_normals(points, k=k)
    nz = np.abs(n[:, 2])
    pred = np.full(len(points), LABEL_OTHER)
    pred[nz >= tread_cos] = LABEL_TREAD
    pred[nz <= riser_cos] = LABEL_RISER
    return pred, n


# ---- Method 3: DBSCAN clustering + slope analysis ----
def dbscan_slope(points: np.ndarray, eps: float = 0.08, min_samples: int = 25,
                 nz_weight: float = 0.5, cluster_tread_cos: float = 0.70,
                 min_cluster: int = 300, k_normal: int = 20) -> np.ndarray:
    """DBSCAN in an augmented feature space (x, y, z, w*|n_z|).

    The normal magnitude |n_z| acts as an additional dimension that pulls
    same-orientation points together and separates tread/riser at the
    bridge, removing the connectivity that causes the staircase to collapse
    into a single 3D cluster.
    """
    normals = estimate_normals(points, k=k_normal)
    nz = np.abs(normals[:, 2])
    feat = np.column_stack([points, nz_weight * nz])

    pred = np.full(len(points), LABEL_OTHER)
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(feat)
    cl = db.labels_
    for c in np.unique(cl):
        if c == -1:
            continue
        m = cl == c
        if m.sum() < min_cluster:
            continue
        pts = points[m]
        pts_c = pts - pts.mean(axis=0)
        cov = pts_c.T @ pts_c / max(len(pts) - 1, 1)
        _, vecs = np.linalg.eigh(cov)
        n = vecs[:, 0]
        if n[2] < 0:
            n = -n
        cos_z = abs(n[2])
        pred[m] = LABEL_TREAD if cos_z >= cluster_tread_cos else LABEL_RISER
    return pred


# ---- Method 4 (bonus): height histogram analysis ----
def height_histogram(points: np.ndarray, bin_w: float = 0.01,
                     peak_quant: float = 0.85,
                     plateau_tol: float = 0.03) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Detect tread z-levels as histogram peaks; classify points within tolerance.

    Returns pred labels, bin centers, counts.
    """
    z = points[:, 2]
    nbins = int((z.max() - z.min()) / bin_w) + 1
    counts, edges = np.histogram(z, bins=nbins)
    centers = 0.5 * (edges[:-1] + edges[1:])

    thr = np.quantile(counts, peak_quant)
    peak_mask = counts >= thr
    # merge contiguous peak bins → plateau centers
    plateaus = []
    i = 0
    while i < len(peak_mask):
        if peak_mask[i]:
            j = i
            while j + 1 < len(peak_mask) and peak_mask[j + 1]:
                j += 1
            plateaus.append(centers[(i + j) // 2])
            i = j + 1
        else:
            i += 1

    pred = np.full(len(points), LABEL_OTHER)
    if plateaus:
        plateaus = np.array(plateaus)
        d = np.abs(z[:, None] - plateaus[None, :])
        near = d.min(axis=1) <= plateau_tol
        pred[near] = LABEL_TREAD
        # everything else with z above the first plateau → riser candidate
        pred[(~near) & (z > plateaus.min() - plateau_tol)] = LABEL_RISER
    return pred, centers, counts


# =====================================================================
# 3. METRICS
# =====================================================================
def metrics_traversable(gt: np.ndarray, pred: np.ndarray) -> dict:
    """Binary metrics for the tread (traversable) class."""
    y_t = (gt == LABEL_TREAD).astype(int)
    p_t = (pred == LABEL_TREAD).astype(int)
    tp = int(((y_t == 1) & (p_t == 1)).sum())
    fp = int(((y_t == 0) & (p_t == 1)).sum())
    fn = int(((y_t == 1) & (p_t == 0)).sum())
    tn = int(((y_t == 0) & (p_t == 0)).sum())
    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    iou = tp / max(tp + fp + fn, 1)
    overall_acc = (gt == pred).mean()
    return dict(accuracy=overall_acc, tread_accuracy=acc, precision=prec,
                recall=rec, f1=f1, iou=iou, tp=tp, fp=fp, fn=fn, tn=tn)


# =====================================================================
# 4. VISUALIZATION
# =====================================================================
COLORS = {LABEL_TREAD: "#2ecc71",  # green = traversable
          LABEL_RISER: "#e74c3c",  # red = riser
          LABEL_OTHER: "#888888"}  # gray = other


def scatter_labeled(points: np.ndarray, labels: np.ndarray, title: str,
                    out: Path, elev: float = 22, azim: float = -60):
    fig = plt.figure(figsize=(7, 5.2))
    ax = fig.add_subplot(111, projection="3d")
    for lab, color in COLORS.items():
        m = labels == lab
        if m.any():
            name = {LABEL_TREAD: "Tread (traversable)",
                    LABEL_RISER: "Riser",
                    LABEL_OTHER: "Other"}[lab]
            ax.scatter(points[m, 0], points[m, 1], points[m, 2],
                       s=1.0, c=color, label=name, depthshade=False)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_zlabel("z [m]")
    ax.set_title(title); ax.view_init(elev=elev, azim=azim)
    ax.legend(loc="upper left", markerscale=8, frameon=True)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


# =====================================================================
# 5. MAIN PIPELINE
# =====================================================================
def main():
    t0 = time.time()
    pc, gt = generate_staircase(rng=RNG)
    print(f"[gen] {len(pc)} points "
          f"(tread={int((gt==0).sum())}, riser={int((gt==1).sum())}, "
          f"other={int((gt==2).sum())})  in {time.time()-t0:.2f}s")

    np.savez_compressed(DATA_DIR / "staircase.npz", points=pc, labels=gt)

    # --- raw cloud viz
    scatter_labeled(pc, np.full(len(pc), LABEL_OTHER), "Raw staircase point cloud",
                    FIG_DIR / "fig_raw_cloud.png")
    scatter_labeled(pc, gt, "Ground truth labels", FIG_DIR / "fig_gt.png")

    methods = {}

    print("[ransac] running ...")
    t = time.time()
    pred_r = ransac_treads(pc)
    methods["RANSAC"] = (pred_r, time.time() - t)

    print("[pca] running ...")
    t = time.time()
    pred_p, normals = pca_normal_classify(pc, k=20)
    methods["PCA-Normal"] = (pred_p, time.time() - t)

    print("[dbscan] running ...")
    t = time.time()
    pred_d = dbscan_slope(pc)
    methods["DBSCAN-Slope"] = (pred_d, time.time() - t)

    print("[hist] running ...")
    t = time.time()
    pred_h, centers, counts = height_histogram(pc)
    methods["HeightHist"] = (pred_h, time.time() - t)

    results = {}
    for name, (pred, dt) in methods.items():
        m = metrics_traversable(gt, pred)
        m["runtime_s"] = round(dt, 3)
        results[name] = m
        print(f"  {name:12s} acc={m['accuracy']:.3f} P={m['precision']:.3f} "
              f"R={m['recall']:.3f} F1={m['f1']:.3f} IoU={m['iou']:.3f} t={dt:.2f}s")
        scatter_labeled(pc, pred, f"{name} segmentation",
                        FIG_DIR / f"fig_seg_{name.lower().replace('-', '_')}.png")

    # --- height histogram figure
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(centers, counts, width=(centers[1] - centers[0]) * 0.95,
           color="#3498db", edgecolor="none")
    ax.set_xlabel("z [m]"); ax.set_ylabel("point count")
    ax.set_title("Height histogram (tread plateaus)")
    fig.tight_layout(); fig.savefig(FIG_DIR / "fig_height_hist.png", dpi=160); plt.close(fig)

    # --- noise sweep
    print("[noise sweep] ...")
    sigmas = [0.0, 0.005, 0.01, 0.02, 0.03, 0.05]
    sweep = {n: [] for n in methods}
    for s in sigmas:
        pc_s, gt_s = generate_staircase(lidar_sigma=s, rng=np.random.default_rng(7))
        sweep["RANSAC"].append(metrics_traversable(gt_s, ransac_treads(pc_s))["f1"])
        sweep["PCA-Normal"].append(
            metrics_traversable(gt_s, pca_normal_classify(pc_s, k=20)[0])["f1"])
        sweep["DBSCAN-Slope"].append(
            metrics_traversable(gt_s, dbscan_slope(pc_s))["f1"])
        sweep["HeightHist"].append(
            metrics_traversable(gt_s, height_histogram(pc_s)[0])["f1"])
        print(f"   sigma={s}")
    fig, ax = plt.subplots(figsize=(6, 3.8))
    for name, vals in sweep.items():
        ax.plot(sigmas, vals, marker="o", label=name)
    ax.set_xlabel("LiDAR noise σ [m]"); ax.set_ylabel("F1 (tread)")
    ax.set_title("Effect of LiDAR noise on tread segmentation")
    ax.grid(True, alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(FIG_DIR / "fig_noise_sweep.png", dpi=160); plt.close(fig)

    with open(DATA_DIR / "metrics.json", "w") as f:
        json.dump({"per_method": results,
                   "noise_sweep": {"sigmas": sigmas, **sweep},
                   "geometry": dict(width=WIDTH, depth=DEPTH, height=HEIGHT,
                                    n_steps=N_STEPS)}, f, indent=2)
    print(f"[done] total {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
