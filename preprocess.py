"""
FITS Preprocessor & WCS Aligner
Aligns multi-band (IR, VIS, UV) FITS images into normalized RGB composites.

-----------------------
* INTENDED GALAXY IS CENTRED. The catalog position is only a starting guess. Extended sources are detected
  (stars are suppressed by heavy smoothing + clipping) and the one at / nearest the target position wins -
  so a brighter star or a neighbouring galaxy elsewhere in the field does not steal the centre. For merging
  pairs the connected system is centred. The crop is centred on the galaxy (never silently shifted).
* NO FOOTPRINT EDGES. Pixels outside any band's survey footprint (NaN or big runs of exact zeros) are masked;
  the crop shrinks to avoid them, and the galaxy is skipped if it cannot (see preprocess_report.csv).
* EMPTY FIELDS ARE DROPPED. If nothing is detected near the target the galaxy is skipped
  (override with --keep_undetected; curated galaxies are never dropped for this).
* STRETCH IS NOISE-AWARE and scaled from the intended galaxy, not from stars or noise.
* Optional --exclude / --relabel lists to remove junk or correct labels (curated test galaxies are never excluded).
"""

import os
import shutil
import zipfile
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import gaussian_filter, median_filter, binary_opening, binary_dilation, label

try:
    from astropy.io import fits
    from astropy.wcs import WCS
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    from reproject import reproject_interp
    ASTRO_READY = True
except ImportError:
    ASTRO_READY = False


def norm_path(p):
    """CSV files written on Windows contain backslashes; this makes them portable."""
    return os.path.normpath(str(p).replace("\\", "/"))


def check_libs():
    if not ASTRO_READY:
        raise ImportError("Please install astropy and reproject: pip install astropy reproject")


def load_fits(path):
    """Loads 2D image data and WCS from a FITS file."""
    check_libs()
    with fits.open(path) as hdul:
        data, header = None, None
        for hdu in hdul:
            if hdu.data is not None:
                arr = np.squeeze(hdu.data)
                if arr.ndim == 2:
                    data = arr.astype(np.float32)
                    header = hdu.header
                    break
                elif arr.ndim > 2:
                    data = np.mean(arr, axis=0).astype(np.float32)
                    header = hdu.header
                    break

        if data is None:
            raise ValueError(f"No 2D image found in {path}")

        try:
            wcs = WCS(header)
            if wcs.naxis > 2:
                wcs = wcs.celestial
        except Exception:
            wcs = None

        return data, wcs


# --------------------------------------------------------------------------------------
# Statistics, masks, stretch
# --------------------------------------------------------------------------------------
def sky_stats(img, valid=None):
    """Robust sky level and noise. Only BRIGHT outliers are clipped (galaxies, stars), and the noise is
    measured from the faint side of the distribution, so a large galaxy filling the frame cannot bias it."""
    ok = np.isfinite(img) if valid is None else (valid & np.isfinite(img))
    v = np.asarray(img)[ok].ravel()
    if v.size == 0:
        return 0.0, 1.0
    m, s = float(np.median(v)), 1.0
    for _ in range(8):
        m = float(np.median(v))
        low = v[v < m]
        s = 1.4826 * float(np.median(m - low)) if low.size else 0.0
        keep = v < m + 3 * max(s, 1e-12)
        if keep.all() or keep.sum() < 50:
            break
        v = v[keep]
    return m, max(s, 1e-6 + 1e-4 * abs(m))


def valid_mask(*bands):
    """True where ALL bands have real data. Invalid = NaN, or a big block of exact zeros
    (what SkyView returns outside a survey footprint). Isolated zero pixels are kept."""
    ok = np.ones(bands[0].shape, dtype=bool)
    for b in bands:
        bad = ~np.isfinite(b)
        zero = (b == 0)
        if zero.any():
            zero = binary_opening(zero, structure=np.ones((5, 5), bool))
        ok &= ~(bad | zero)
    return ok


def asinh_stretch(image, factor=0.05, p_high=99.7, clip_sigma=1.5, smooth=1.0, valid=None, ref_mask=None):
    """Noise-aware asinh stretch. Sky is subtracted, noise is not amplified, empty bands stay dark.
    The white point comes from the intended galaxy (`ref_mask`) after a median filter, so stars do not set the scale.
    Invalid (outside-footprint) pixels are returned black."""
    img = np.asarray(image, dtype=np.float32)
    ok = np.isfinite(img) if valid is None else (valid & np.isfinite(img))
    if not ok.any():
        return np.zeros_like(img)
    img = np.where(ok, img, np.median(img[ok]))
    if smooth:
        img = gaussian_filter(img, smooth)

    sky, sigma = sky_stats(img, ok)
    h, w = img.shape
    # white point from a median-filtered copy: removes point sources (stars) but keeps smooth galaxy cores
    med = median_filter(img, size=15)
    if ref_mask is not None and (ref_mask & ok).sum() >= 50:
        ref = med[ref_mask & ok]
    else:
        ref = med[h // 4:3 * h // 4, w // 4:3 * w // 4].ravel()
    vmin = sky + clip_sigma * sigma
    vmax = max(float(np.percentile(ref, p_high)), vmin + 6 * sigma)

    norm = np.clip((img - vmin) / (vmax - vmin), 0.0, 1.0)
    out = np.arcsinh(norm / factor) / np.arcsinh(1.0 / factor)
    return np.where(ok, out, 0.0).astype(np.float32)


# --------------------------------------------------------------------------------------
# Finding the intended galaxy
# --------------------------------------------------------------------------------------
def _remove_stars(z):
    """Replace point sources (compact excess over a median-filtered copy, incl. diffraction spikes) with the
    local background so they cannot be mistaken for the target. Smooth galaxies survive the median filter."""
    med = median_filter(z, size=11)
    excess = gaussian_filter(z, 1.0) - med
    star = binary_dilation(excess > 8.0, iterations=3) | binary_dilation(excess > 100.0, iterations=9)
    out = np.where(star, med, z)
    return np.clip(out, -5, 30)


def locate_galaxy(vis, ir, valid, expected_xy=None, search_frac=0.30, max_shift_frac=0.20,
                  ext_sigma=5.0, min_area_frac=0.0015):
    """
    Find the extended source belonging to the target.

    Detection uses the visible + infrared bands (galaxies of every type are bright there; ellipticals are
    faint in UV). Each band is turned into sky-noise units, clipped (so saturated stars cannot dominate) and
    heavily smoothed (stars are compact and wash out, galaxies do not). Connected regions above a threshold
    are candidates; the one containing / nearest the expected position wins, with large regions preferred.

    Returns dict: center (x, y), mask (region of the chosen galaxy or None), status, n_extended, extent_px.
      status: "ok"            centroid refined
              "clamped"       refined centre was far from the catalog position, shift limited
              "undetected"    no extended source near the target centre = expected position
    """
    h, w = vis.shape
    S = min(h, w)
    ex, ey = (w / 2.0, h / 2.0) if expected_xy is None else expected_xy

    z = 0.0
    for b in (vis, ir):
        sky, sig = sky_stats(b, valid)
        zb = np.where(valid & np.isfinite(b), (b - sky) / sig, 0.0)
        z = z + _remove_stars(zb)
    D = gaussian_filter(np.clip(z, -10, 60), ext_sigma)
    mD, sD = sky_stats(D, valid)

    min_area = max(30.0, min_area_frac * h * w)
    min_half_diam = 1.6 * 2.355 * ext_sigma                    # wider than a smoothed point source
    R = search_frac * S
    yy, xx = np.mgrid[:h, :w]
    chosen, n_ext, thr_used = None, 0, None

    for k_sigma in (2.5, 4.0, 7.0, 12.0):                      # raise the bar if everything merges into one blob
        thr = mD + k_sigma * sD
        mask = (D > thr) & valid
        lab, n = label(binary_dilation(mask, iterations=2))
        cands = []
        for k in range(1, n + 1):
            m = (lab == k) & mask
            area = int(m.sum())
            if area < min_area:
                continue
            # a star or noise blob is only as wide as the smoothing kernel itself; a galaxy is wider
            level = thr + 0.5 * (float(D[m].max()) - thr)
            half_diam = 2 * np.sqrt(float(((D > level) & m).sum()) / np.pi)
            if half_diam < min_half_diam:
                continue
            d = float(np.min(np.hypot(xx[m] - ex, yy[m] - ey)))
            cands.append({"m": m, "area": area, "d": d, "score": d - 0.3 * np.sqrt(area)})
        near = [c for c in cands if c["d"] <= R]
        n_ext = len([c for c in cands if c["d"] <= 0.5 * S])
        if not near:
            continue
        best = min(near, key=lambda c: c["score"])
        chosen, thr_used = best, thr
        if best["area"] <= 0.4 * h * w:
            break

    if chosen is None:
        return {"center": (ex, ey), "mask": None, "status": "undetected", "n_extended": n_ext, "extent_px": 0.0}

    m = chosen["m"]
    Dp = np.clip(D - thr_used, 0, None)

    # Start point: the catalog position if it lies on the detected source (trust it over brighter
    # neighbours); otherwise the strongest point of the chosen region near the catalog position.
    ix, iy = int(round(min(max(ex, 0), w - 1))), int(round(min(max(ey, 0), h - 1)))
    if Dp[iy, ix] > 0 and m[iy, ix]:
        cx, cy = float(ex), float(ey)
    else:
        prior = np.exp(-0.5 * (np.hypot(xx - ex, yy - ey) / (0.15 * S)) ** 2)
        score = np.where(m, Dp * prior, -1.0)
        py, px = np.unravel_index(int(np.argmax(score)), score.shape)
        cx, cy = float(px), float(py)

    # Local mean-shift: move to the centre of light of the galaxy under the start point. The Gaussian window
    # is small compared with the field, so a neighbouring galaxy does not pull the centre away, while the
    # two members of a close merging pair are both inside it and the centre lands between them.
    sk, c0 = 0.07 * S, (cx, cy)
    for _ in range(20):
        K = Dp * np.exp(-0.5 * ((xx - cx) ** 2 + (yy - cy) ** 2) / sk ** 2)
        tot = K.sum()
        if tot <= 0:
            break
        nx, ny = float((xx * K).sum() / tot), float((yy * K).sum() / tot)
        d0 = np.hypot(nx - c0[0], ny - c0[1])
        if d0 > 0.15 * S:                                      # never wander far from the start point
            nx, ny = c0[0] + (nx - c0[0]) * 0.15 * S / d0, c0[1] + (ny - c0[1]) * 0.15 * S / d0
        moved = np.hypot(nx - cx, ny - cy)
        cx, cy = nx, ny
        if moved < 0.2:
            break

    status, lim = "ok", max_shift_frac * S
    shift = np.hypot(cx - ex, cy - ey)
    if shift > lim:                                           # trust the catalog position partly
        cx, cy = ex + (cx - ex) * lim / shift, ey + (cy - ey) * lim / shift
        status = "clamped"
    local = m & (np.hypot(xx - cx, yy - cy) < 0.35 * S)       # this galaxy only, not far neighbours
    return {"center": (cx, cy), "mask": local if local.sum() >= 50 else m, "status": status,
            "n_extended": n_ext, "extent_px": float(2 * np.sqrt(max(local.sum(), 1) / np.pi))}


def choose_crop(cx, cy, valid, frac=0.7, min_frac=0.40, max_invalid=0.02, extent_px=0.0, auto_zoom=False):
    """Square crop CENTRED on (cx, cy). If it would leave the image or include out-of-footprint pixels the
    side is shrunk (down to 40% of the short side) rather than shifting the galaxy off-centre."""
    h, w = valid.shape
    S = min(h, w)
    side = frac * S
    if auto_zoom and extent_px > 0:
        side = float(np.clip(2.0 * extent_px, min_frac * S, 0.9 * S))
    min_side = min_frac * S

    while True:
        s = int(round(side))
        x1, y1 = int(round(cx - s / 2)), int(round(cy - s / 2))
        if x1 >= 0 and y1 >= 0 and x1 + s <= w and y1 + s <= h:
            if 1.0 - valid[y1:y1 + s, x1:x1 + s].mean() <= max_invalid:
                return x1, y1, s, "centered"
        if side * 0.92 < min_side:
            break
        side *= 0.92

    # could not centre cleanly: take the nominal window pushed inside the image and check it
    s = int(round(frac * S))
    x1 = int(min(max(round(cx - s / 2), 0), w - s))
    y1 = int(min(max(round(cy - s / 2), 0), h - s))
    if 1.0 - valid[y1:y1 + s, x1:x1 + s].mean() > 0.10:
        raise ValueError("survey footprint edge inside the crop (galaxy too close to the edge of the data)")
    return x1, y1, s, "shifted"


def build_composite(vis, ir, uv, expected_xy=None, target_size=(224, 224), frac=0.7,
                    auto_zoom=False, keep_undetected=True):
    """Pure-numpy core: aligned bands -> centred RGB PIL image (R = IR, G = VIS, B = UV) + info dict."""
    valid = valid_mask(vis, ir, uv)
    if valid.mean() < 0.5:
        raise ValueError("less than half of the field has data in all three bands")

    loc = locate_galaxy(vis, ir, valid, expected_xy)
    if loc["status"] == "undetected" and not keep_undetected:
        raise ValueError("no galaxy detected near the target position (empty field)")

    ref = binary_dilation(loc["mask"], iterations=3) if loc["mask"] is not None else None
    rgb = np.dstack([asinh_stretch(ir, valid=valid, ref_mask=ref),
                     asinh_stretch(vis, valid=valid, ref_mask=ref),
                     asinh_stretch(uv, valid=valid, ref_mask=ref)])

    cx, cy = loc["center"]
    x1, y1, s, crop_status = choose_crop(cx, cy, valid, frac=frac, extent_px=loc["extent_px"], auto_zoom=auto_zoom)
    cropped = rgb[y1:y1 + s, x1:x1 + s, :]
    img = Image.fromarray((np.clip(cropped, 0, 1) * 255).astype(np.uint8)).resize(target_size, Image.Resampling.BILINEAR)

    ex, ey = (vis.shape[1] / 2.0, vis.shape[0] / 2.0) if expected_xy is None else expected_xy
    info = {"detect": loc["status"], "crop": crop_status, "shift_px": round(float(np.hypot(cx - ex, cy - ey)), 1),
            "n_galaxies": loc["n_extended"], "crop_frac": round(s / min(vis.shape), 2)}
    return img, info


def crop_center(composite, center_xy=None, target_size=(224, 224), frac=0.7):
    """Kept for backwards compatibility: plain square crop around a centre (no detection)."""
    h, w, _ = composite.shape
    side = max(2, int(min(h, w) * frac))
    cx, cy = (w // 2, h // 2) if center_xy is None else (int(round(center_xy[0])), int(round(center_xy[1])))
    x1 = min(max(cx - side // 2, 0), w - side)
    y1 = min(max(cy - side // 2, 0), h - side)
    img = Image.fromarray((composite[y1:y1 + side, x1:x1 + side, :] * 255).astype(np.uint8))
    return img.resize(target_size, Image.Resampling.BILINEAR)


def align_bands(ir_path, vis_path, uv_path, ra_dec=None, target_size=(224, 224), return_info=False,
                auto_zoom=False, keep_undetected=True):
    """
    Aligns IR and UV onto the Visible reference WCS and returns an RGB PIL Image centred on the target galaxy.
    R = Infrared, G = Visible, B = Ultraviolet
    """
    data_vis, wcs_vis = load_fits(norm_path(vis_path))
    data_ir, wcs_ir = load_fits(norm_path(ir_path))
    data_uv, wcs_uv = load_fits(norm_path(uv_path))

    if wcs_vis and wcs_ir and wcs_uv:
        aligned_ir, _ = reproject_interp((data_ir, wcs_ir), wcs_vis, shape_out=data_vis.shape)
        aligned_uv, _ = reproject_interp((data_uv, wcs_uv), wcs_vis, shape_out=data_vis.shape)
    else:
        from scipy.ndimage import zoom
        aligned_ir = zoom(data_ir, (data_vis.shape[0] / data_ir.shape[0], data_vis.shape[1] / data_ir.shape[1]))
        aligned_uv = zoom(data_uv, (data_vis.shape[0] / data_uv.shape[0], data_vis.shape[1] / data_uv.shape[1]))

    # expected target pixel: catalog position through the WCS (the cutout is also requested centred on it)
    expected = None
    if ra_dec and wcs_vis:
        try:
            coord = SkyCoord(ra=ra_dec[0] * u.deg, dec=ra_dec[1] * u.deg)
            px, py = wcs_vis.world_to_pixel(coord)
            px, py = float(px), float(py)
            h, w = data_vis.shape
            if np.isfinite(px) and np.isfinite(py) and 0 <= px < w and 0 <= py < h:
                expected = (px, py)
        except Exception:
            expected = None

    img, info = build_composite(data_vis, aligned_ir, aligned_uv, expected, target_size,
                                auto_zoom=auto_zoom, keep_undetected=keep_undetected)
    return (img, info) if return_info else img


# --------------------------------------------------------------------------------------
# Dataset building
# --------------------------------------------------------------------------------------
def assign_splits(df_cls, split_ratio, curated_test=False):
    """Returns a list of split names ('train'/'val'/'test'), one per row of df_cls."""
    n = len(df_cls)
    if curated_test and "source" in df_cls.columns:
        is_cur = (df_cls["source"] == "curated").to_numpy()
        out = ["test"] * n
        rest = [i for i in range(n) if not is_cur[i]]
        n_val = max(1, int(len(rest) * 0.2)) if len(rest) >= 3 else 0
        for k, i in enumerate(rest):
            out[i] = "train" if k < len(rest) - n_val else "val"
        return out

    n_train = int(n * split_ratio[0])
    n_val = int(n * split_ratio[1])
    if n >= 3:
        n_val = max(1, n_val)
        n_train = min(n_train, n - n_val - 1)
    return ["train" if i < n_train else "val" if i < n_train + n_val else "test" for i in range(n)]


def load_exclude_ids(path, reasons=None):
    """Galaxy ids to drop: a .txt (one id per line) or a .csv with an 'id' / 'galaxy_id' column
    (optionally filtered by its 'reason' column)."""
    if str(path).lower().endswith(".csv"):
        d = pd.read_csv(path)
        col = "galaxy_id" if "galaxy_id" in d.columns else "id"
        if reasons and "reason" in d.columns:
            d = d[d["reason"].isin(reasons)]
        return set(d[col].astype(str))
    with open(path) as f:
        return {ln.strip() for ln in f if ln.strip() and not ln.startswith("#")}


def process_dataset(csv_path="galaxies.csv", out_dir="processed_dataset",
                    split_ratio=(0.7, 0.2, 0.1), curated_test=False, keep_undetected=False,
                    auto_zoom=False, exclude=None, exclude_reasons=None, relabel=None,
                    report_csv="preprocess_report.csv"):
    df = pd.read_csv(csv_path)

    if relabel:
        fix = pd.read_csv(relabel)
        mapping = dict(zip(fix["galaxy_id"].astype(str), fix["label"]))
        hit = df["galaxy_id"].astype(str).isin(mapping)
        df.loc[hit, "label"] = df.loc[hit, "galaxy_id"].astype(str).map(mapping)
        print(f"Relabelled {int(hit.sum())} galaxies from {relabel}")

    if exclude:
        ids = load_exclude_ids(exclude, exclude_reasons)
        drop = df["galaxy_id"].astype(str).isin(ids)
        if "source" in df.columns:
            drop &= (df["source"] != "curated")               # never touch the trusted test galaxies
        print(f"Excluding {int(drop.sum())} galaxies listed in {exclude}")
        df = df[~drop].reset_index(drop=True)

    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)

    splits = ["train", "val", "test"]
    labels = df["label"].unique()

    for s in splits:
        for lbl in labels:
            os.makedirs(os.path.join(out_dir, s, str(lbl)), exist_ok=True)

    counts = {s: {l: 0 for l in labels} for s in splits}
    skipped, report = [], []

    for lbl in labels:
        df_cls = df[df["label"] == lbl].sample(frac=1.0, random_state=42).reset_index(drop=True)
        n = len(df_cls)
        assigned = assign_splits(df_cls, split_ratio, curated_test)

        print(f"Processing {lbl} ({n} total)...")
        for idx, row in df_cls.iterrows():
            sp = assigned[idx]
            gid = str(row["galaxy_id"])
            out_file = os.path.join(out_dir, sp, str(lbl), f"{gid}.png")

            ra_dec = (float(row["ra"]), float(row["dec"])) if ("ra" in row and not pd.isna(row["ra"])) else None

            try:
                trusted = ("source" in row and row["source"] == "curated")      # hand-picked, often huge: never drop
                img, info = align_bands(row["ir_path"], row["vis_path"], row["uv_path"], ra_dec=ra_dec,
                                        return_info=True, auto_zoom=auto_zoom,
                                        keep_undetected=keep_undetected or trusted)
                img.save(out_file)
                counts[sp][lbl] += 1
                report.append({"galaxy_id": gid, "label": lbl, "split": sp, "status": "ok", **info})
            except Exception as e:
                skipped.append(f"{gid} ({lbl}): {e}")
                report.append({"galaxy_id": gid, "label": lbl, "split": sp, "status": "skipped", "reason": str(e)})

    print("\nDataset Summary:")
    for s in splits:
        sub = ", ".join([f"{l}: {counts[s][l]}" for l in labels])
        print(f"  {s.upper()}: {sum(counts[s].values())} ({sub})")

    if report:
        rep = pd.DataFrame(report)
        rep.to_csv(report_csv, index=False)
        ok = rep[rep["status"] == "ok"]
        if len(ok):
            print(f"\nCentring: {int((ok['detect'] == 'ok').sum())} refined, "
                  f"{int((ok['detect'] == 'clamped').sum())} shift-limited, "
                  f"{int((ok['detect'] == 'undetected').sum())} kept without a detection; "
                  f"median shift from catalog position {ok['shift_px'].median():.0f} px; "
                  f"{int((ok['n_galaxies'] > 1).sum())} fields contain more than one extended source")
        print(f"Per-galaxy details: {report_csv}")

    if skipped:
        print(f"\nSkipped {len(skipped)} galaxies (reasons in {report_csv}), first few:")
        for s in skipped[:8]:
            print("  ", s)

    return out_dir


def zip_dataset(dataset_dir="processed_dataset", zip_name="galaxy_dataset.zip"):
    print(f"Zipping {dataset_dir} -> {zip_name}...")
    if os.path.exists(zip_name):
        os.remove(zip_name)

    with zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(dataset_dir):
            for f in files:
                full = os.path.join(root, f)
                rel = os.path.relpath(full, start=dataset_dir)
                zf.write(full, arcname=rel)

    size_mb = os.path.getsize(zip_name) / (1024 * 1024)
    print(f"Done! {zip_name} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Preprocess and align FITS dataset")
    parser.add_argument("--csv", default="galaxies.csv", help="Catalog CSV")
    parser.add_argument("--out_dir", default="processed_dataset", help="Output directory")
    parser.add_argument("--zip", action="store_true", help="Compress dataset into zip")
    parser.add_argument("--curated_test", action="store_true",
                        help="Use the hand-curated galaxies as the test split (catalog galaxies -> train/val)")
    parser.add_argument("--keep_undetected", action="store_true",
                        help="keep galaxies where no extended source is found near the target (default: skip them)")
    parser.add_argument("--auto_zoom", action="store_true",
                        help="size the crop from the detected galaxy extent (small galaxies fill more of the frame)")
    parser.add_argument("--exclude", help="txt (one id per line) or csv (id/galaxy_id column) of galaxies to drop")
    parser.add_argument("--exclude_reasons", nargs="*",
                        help="with a csv that has a 'reason' column (suspect_galaxies.csv): only drop these reasons")
    parser.add_argument("--relabel", help="csv with columns galaxy_id,label to correct labels")
    parser.add_argument("--ir", help="Single IR FITS path")
    parser.add_argument("--vis", help="Single VIS FITS path")
    parser.add_argument("--uv", help="Single UV FITS path")
    parser.add_argument("--out", default="aligned.png", help="Single output image")

    args = parser.parse_args()

    if args.ir and args.vis and args.uv:
        img = align_bands(args.ir, args.vis, args.uv, auto_zoom=args.auto_zoom)
        img.save(args.out)
        print(f"Saved aligned composite to {args.out}")
    else:
        process_dataset(args.csv, out_dir=args.out_dir, curated_test=args.curated_test,
                        keep_undetected=args.keep_undetected, auto_zoom=args.auto_zoom,
                        exclude=args.exclude, exclude_reasons=args.exclude_reasons, relabel=args.relabel)
        if args.zip:
            zip_dataset(args.out_dir)
