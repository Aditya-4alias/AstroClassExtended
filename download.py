"""
Download multi-wavelength FITS images from NASA SkyView (IR, VIS, UV).

* Merges the hand-curated list with the catalog targets in targets_extra.csv (made by build_targets.py).
* A galaxy is only kept if ALL THREE bands (IR, VIS, UV) are available AND each band passes a quality check
  (valid pixels, inside the survey footprint, real dynamic range, target centred, visible source).
  If a survey fails the check the next fallback survey for that band is tried.
* galaxies.csv only ever contains complete, quality-checked 3-band galaxies; everything else is listed in
  rejected_targets.csv together with the reason.
"""

import os
import re
import ssl
import json
import math
import time
import warnings
import urllib.parse
import urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd

SPEC = {
    "spiral": [
        ("M51", 0.30, 202.46957, 47.19526),
        ("M31", 2.50, 10.68471, 41.26875),
        ("M101", 0.70, 210.80227, 54.34895),
        ("M81", 0.65, 148.88822, 69.06529),
        ("M33", 1.20, 23.46210, 30.65994),
        ("M63", 0.35, 198.95554, 42.02937),
        ("M74", 0.30, 24.17400, 15.78300),
        ("M83", 0.35, 204.25396, -29.86576),
        ("NGC1365", 0.30, 53.40152, -36.14066),
        ("NGC1300", 0.20, 49.92082, -19.41108),
        ("NGC2997", 0.25, 146.41170, -31.19125),
        ("NGC6946", 0.32, 308.71800, 60.15390),
        ("NGC1232", 0.22, 47.45830, -20.57920),
        ("NGC4535", 0.22, 188.58460, 8.19780),
        ("NGC4321", 0.22, 185.72890, 15.82230),
        ("NGC3344", 0.22, 160.84630, 24.92280),
        ("NGC3627", 0.28, 170.06670, 12.99170),
        ("NGC3184", 0.22, 154.57080, 41.42440),
        ("NGC4826", 0.30), ("NGC3623", 0.30), ("NGC1068", 0.20), ("NGC4736", 0.30),
        ("NGC3351", 0.25), ("NGC3368", 0.25), ("NGC4254", 0.22), ("NGC4192", 0.30),
        ("NGC4303", 0.22), ("NGC4501", 0.25), ("NGC253", 0.80), ("NGC300", 0.80),
        ("NGC2403", 0.65), ("NGC7331", 0.30), ("NGC2841", 0.28),
    ],
    "elliptical": [
        ("M87", 0.25, 187.70593, 12.39112),
        ("M84", 0.16, 186.26560, 12.88698),
        ("M86", 0.22, 186.54900, 12.94600),
        ("M49", 0.28, 187.44365, 8.00039),
        ("M60", 0.22, 190.91684, 11.55269),
        ("M89", 0.15, 188.91586, 12.55634),
        ("M105", 0.16, 161.95670, 12.58160),
        ("NGC1399", 0.20, 54.62094, -35.45163),
        ("NGC1404", 0.12, 54.71764, -35.59020),
        ("NGC4473", 0.14, 187.45380, 13.42940),
        ("NGC4697", 0.22, 192.14950, -5.80070),
        ("NGC5846", 0.14, 226.62260, 1.60560),
        ("NGC4261", 0.14, 184.84670, 5.82470),
        ("NGC3377", 0.16, 161.92880, 13.98610),
        ("NGC4621", 0.16, 190.50830, 11.64720),
        ("NGC4365", 0.20, 186.11830, 7.32080),
        ("NGC4636", 0.22), ("NGC4660", 0.10), ("NGC4494", 0.16), ("NGC4125", 0.20),
        ("NGC4291", 0.10), ("NGC3608", 0.12), ("NGC5322", 0.14), ("NGC1407", 0.16),
        ("NGC720", 0.18), ("NGC1600", 0.14), ("NGC5813", 0.16), ("NGC5982", 0.10),
        ("NGC821", 0.12), ("NGC584", 0.14), ("NGC1052", 0.16), ("NGC1549", 0.18),
        ("NGC1374", 0.10), ("NGC1427", 0.12),
    ],
    "anomaly": [
        ("NGC4038", 0.15, 180.47100, -18.86800),
        ("NGC4676", 0.10, 191.54220, 30.72890),
        ("M82", 0.25, 148.96846, 69.67970),
        ("NGC520", 0.12, 21.14480, 3.79370),
        ("NGC2623", 0.08, 129.60000, 25.75300),
        ("NGC7252", 0.12),
        ("Arp220", 0.06, 233.73850, 23.50320),
        ("NGC6240", 0.08, 253.24540, 2.40080),
        ("NGC3256", 0.10, 156.96470, -43.90380),
        ("NGC4449", 0.15, 187.04630, 44.09360),
        ("NGC4485", 0.15, 187.63580, 41.70030),
        ("NGC7714", 0.10, 354.05770, 2.15540),
        ("NGC6052", 0.06, 241.30460, 20.54220),
        ("NGC34", 0.08, 2.77580, -12.10720),
        ("NGC3994", 0.08, 179.40080, 32.27780),
        ("NGC2782", 0.14),
        ("NGC3310", 0.10, 159.69120, 53.50360),
        ("NGC4656", 0.30, 190.99040, 32.20140),
        ("NGC1487", 0.08),
        ("NGC1316", 0.30), ("NGC5128", 0.50), ("NGC3690", 0.08), ("NGC2207", 0.12),
        ("NGC1275", 0.10), ("NGC1614", 0.06), ("NGC660", 0.20), ("NGC4631", 0.40),
        ("NGC3077", 0.15), ("NGC7469", 0.08), ("NGC3921", 0.08), ("NGC4194", 0.06),
        ("IC1623", 0.08), ("NGC5256", 0.08), ("NGC2992", 0.15),
    ],
}

BANDS = ["ir", "vis", "uv"]

# Broader per-band survey fallbacks so all 3 bands are found whenever coverage exists
SURVEYS = {
    "ir": [
        "WISE 3.4", "2MASS-K", "WISE 4.6",
        "2MASS-H", "2MASS-J", "WISE 12", "WISE 22", "DSS2 IR",
    ],
    "vis": [
        "DSS2 Red", "SDSSr", "DSS2 Blue", "DSS", "SDSSg", "SDSSi", "SDSSz",
    ],
    "uv": [
        "GALEX Near UV", "GALEX Far UV", "SDSSu",
    ],
}

UA = {"User-Agent": "Mozilla/5.0 AstroResearcher/1.0"}
SESAME_URLS = [
    "https://cds.unistra.fr/cgi-bin/nph-sesame/-oI/A?{q}",
    "https://vizier.cfa.harvard.edu/viz-bin/nph-sesame/-oI/A?{q}",
]

UV_SNR_RELAX = 0.5   # galaxies (esp. ellipticals) are faint in the UV, so the S/N bar is halved there


def make_ssl_context(insecure=False):
    if insecure:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        return ctx
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


SSL_CTX = make_ssl_context(insecure=False)


# --------------------------------------------------------------------------------------
# Targets
# --------------------------------------------------------------------------------------
def build_targets():
    """Hand-curated galaxies (trusted labels)."""
    targets, seen = [], set()
    for cls, entries in SPEC.items():
        for e in entries:
            gid, size = e[0], e[1]
            ra, dec = (e[2], e[3]) if len(e) == 4 else (None, None)
            if gid not in seen:
                seen.add(gid)
                targets.append({"id": gid, "class": cls, "ra": ra, "dec": dec, "size": size,
                                "source": "curated"})
    return targets


def load_extra_targets(csv_path):
    """Read targets_extra.csv written by build_targets.py (columns: id,class,ra,dec,size,source,...)."""
    if not csv_path:
        return []
    if not os.path.exists(csv_path):
        print(f"[WARN] {csv_path} not found - using the curated list only. "
              f"Create it with:  python build_targets.py --per_class 170")
        return []
    df = pd.read_csv(csv_path)
    need = {"id", "class", "ra", "dec", "size"}
    if not need.issubset(df.columns):
        raise SystemExit(f"{csv_path} must contain the columns {sorted(need)} (found {list(df.columns)})")

    df = df.dropna(subset=["id", "class", "ra", "dec", "size"])
    df = df[df["class"].isin(SPEC.keys())]
    items = []
    for rec in df.to_dict("records"):
        items.append({"id": str(rec["id"]), "class": str(rec["class"]), "ra": float(rec["ra"]),
                      "dec": float(rec["dec"]), "size": float(rec["size"]),
                      "source": str(rec.get("source", "catalog") or "catalog")})
    print(f"Loaded {len(items)} extra targets from {csv_path}: "
          f"{pd.Series([i['class'] for i in items]).value_counts().to_dict()}")
    return items


def merge_targets(curated, extra):
    """Curated first (so it wins on duplicates); drop extra targets whose id is already used."""
    seen = {t["id"] for t in curated}
    merged, dup_ids = list(curated), 0
    for t in extra:
        if t["id"] in seen:
            dup_ids += 1
            continue
        seen.add(t["id"])
        merged.append(t)
    if dup_ids:
        print(f"  {dup_ids} extra targets skipped (same id as a curated galaxy)")
    return merged


def resolve_name(gid):
    name = re.sub(r"^([A-Za-z]+)(\d)", r"\1 \2", gid)
    q = urllib.parse.quote(name)
    for tmpl in SESAME_URLS:
        try:
            req = urllib.request.Request(tmpl.format(q=q), headers=UA)
            with urllib.request.urlopen(req, timeout=30, context=SSL_CTX) as resp:
                text = resp.read().decode("utf-8", errors="ignore")
            m = re.search(r"^%J\s+([-+]?\d+\.?\d*)\s+([-+]?\d+\.?\d*)", text, re.MULTILINE)
            if m:
                return float(m.group(1)), float(m.group(2))
        except Exception:
            continue
    return None


# --------------------------------------------------------------------------------------
# Download + quality checks
# --------------------------------------------------------------------------------------
def _valid_fits(data, min_bytes=10000):
    return (
        data.startswith(b"SIMPLE")
        and len(data) >= min_bytes
        and b"No data found" not in data[:5000]
    )


def check_band_quality(path, band, ra, dec, pixels, min_valid=0.9, max_zero=0.25, min_snr=3.0):
    """
    Decide whether one downloaded band is usable. Returns (ok, reason).

    Checks: readable 2-D image of the requested size, mostly finite pixels, not mostly zero-filled
    (outside the survey footprint), real dynamic range, target position near the image centre (WCS),
    and a source at the centre that stands out from the sky background.
    Needs astropy; without it the content checks are skipped (and a note is returned).
    """
    try:
        from astropy.io import fits
    except ImportError:
        return True, "astropy missing - content checks skipped"

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with fits.open(path) as hdul:
                data = header = None
                for hdu in hdul:
                    if hdu.data is None:
                        continue
                    arr = np.squeeze(hdu.data)
                    if arr.ndim == 2:
                        data, header = arr.astype(np.float32), hdu.header
                        break
                    if arr.ndim > 2:
                        data, header = np.mean(arr, axis=0).astype(np.float32), hdu.header
                        break
    except Exception as e:
        return False, f"unreadable FITS ({e})"

    if data is None or data.ndim != 2:
        return False, "no 2-D image data"
    h, w = data.shape
    if min(h, w) < 0.9 * pixels:
        return False, f"image too small ({w}x{h})"

    finite = np.isfinite(data)
    if finite.mean() < min_valid:
        return False, f"{100 * (1 - finite.mean()):.0f}% NaN pixels"
    vals = data[finite]
    if (vals == 0).mean() > max_zero:
        return False, f"{100 * (vals == 0).mean():.0f}% zero pixels (outside survey footprint)"
    if np.unique(vals[::5]).size < 20:
        return False, "flat or heavily quantised image"
    lo, hi = np.percentile(vals, [1, 99.5])
    if not hi > lo:
        return False, "no dynamic range"

    # target should sit near the centre of the cutout
    if ra is not None and dec is not None and header is not None:
        try:
            from astropy.wcs import WCS
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                wcs = WCS(header).celestial
            if wcs.has_celestial:
                x, y = wcs.world_to_pixel_values(ra, dec)
                x, y = float(x), float(y)
                if np.isfinite(x) and np.isfinite(y) and not (0.25 * w <= x <= 0.75 * w and 0.25 * h <= y <= 0.75 * h):
                    return False, f"target off-centre (pixel {x:.0f},{y:.0f})"
        except Exception:
            pass

    # a real source at the centre (99th percentile of the central box vs. the sky in the outer frame)
    filled = np.where(finite, data, np.median(vals))
    center = filled[int(h * 0.3):int(h * 0.7), int(w * 0.3):int(w * 0.7)]
    fh, fw = max(1, int(h * 0.2)), max(1, int(w * 0.2))
    border = np.concatenate([filled[:fh].ravel(), filled[-fh:].ravel(),
                             filled[:, :fw].ravel(), filled[:, -fw:].ravel()])
    bg = float(np.median(border))
    sigma = max(1.4826 * float(np.median(np.abs(border - bg))), 0.01 * float(hi - lo))
    snr = (float(np.percentile(center, 99)) - bg) / sigma
    need = min_snr * (UV_SNR_RELAX if band == "uv" else 1.0)
    if snr < need:
        return False, f"no clear source at target (S/N {snr:.1f} < {need:.1f})"
    return True, f"S/N {snr:.1f}"


def download_fits(position, survey, dest, size=0.1, pixels=256, min_bytes=10000,
                  retries=2, timeout=90, verbose=False):
    url = "https://skyview.gsfc.nasa.gov/current/cgi/runquery.pl?" + urllib.parse.urlencode({
        "Survey": survey,
        "Position": str(position),
        "Pixels": str(pixels),
        "Size": str(size),
        "Return": "FITS",
        "Coordinates": "J2000",
        "Projection": "Tan",
    })

    def _save(data):
        tmp = dest + ".tmp"
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, dest)

    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(
                urllib.request.Request(url, headers=UA), timeout=timeout, context=SSL_CTX
            ) as resp:
                data = resp.read()
            if _valid_fits(data, min_bytes=min_bytes):
                _save(data)
                return True

            text = data.decode("utf-8", errors="ignore")
            links = re.findall(r'href=[\'"]?([^\'" >]+\.fits)', text, re.IGNORECASE)
            if links:
                link = links[0]
                if link.startswith("/"):
                    link = "https://skyview.gsfc.nasa.gov" + link
                elif not link.startswith("http"):
                    link = "https://skyview.gsfc.nasa.gov/current/cgi/" + link
                with urllib.request.urlopen(
                    urllib.request.Request(link, headers=UA), timeout=timeout, context=SSL_CTX
                ) as r2:
                    f_data = r2.read()
                if _valid_fits(f_data, min_bytes=min_bytes):
                    _save(f_data)
                    return True

            if verbose and attempt == retries:
                print(f"    [NO-FITS] {survey} @ {position} ({len(data)} bytes): {data[:120]!r}")
            return False
        except Exception as e:
            if verbose and attempt == retries:
                print(f"    [ERR] {survey} @ {position}: {e}")
            time.sleep(2 * (attempt + 1))
    return False


def get_galaxy(item, out_dir="./data/raw_fits", pixels=256, min_bytes=10000,
               retries=2, timeout=90, verbose=False, qc=None):
    """
    Returns (row, reason). `row` is a dict only when all three bands were found AND passed the
    quality check, otherwise None and `reason` says why the galaxy was rejected.
    """
    gid, cls, size = item["id"], item["class"], item.get("size", 0.1)
    ra, dec = item["ra"], item["dec"]

    if ra is None or dec is None:
        res = resolve_name(gid)
        if res is None:
            return None, "position could not be resolved"
        ra, dec = res

    target_dir = os.path.join(out_dir, cls, gid)
    os.makedirs(target_dir, exist_ok=True)

    meta = {"ra": round(ra, 5), "dec": round(dec, 5), "size": size, "pixels": pixels, "qc": bool(qc)}
    meta_path = os.path.join(target_dir, "meta.json")
    cache_ok, cached_surveys = False, {}
    if os.path.exists(meta_path):
        try:
            with open(meta_path) as f:
                stored = json.load(f)
            cache_ok = all(stored.get(k) == v for k, v in meta.items())
            cached_surveys = stored.get("surveys", {})
        except Exception:
            cache_ok = False

    def band_ok(path, band):
        if not qc:
            return True, ""
        return check_band_quality(path, band, ra, dec, pixels, **qc)

    pos = f"{ra:.5f},{dec:.5f}"
    paths, used = {}, {}

    for band in BANDS:
        dest = os.path.join(target_dir, f"{gid}_{band}.fits")
        if cache_ok and os.path.exists(dest) and os.path.getsize(dest) >= min_bytes:
            ok, _ = band_ok(dest, band)
            if ok:
                paths[band], used[band] = dest, cached_surveys.get(band, "CACHED")
                continue
            os.remove(dest)

        rejects = []
        for sname in SURVEYS[band]:
            if not download_fits(pos, sname, dest, size=size, pixels=pixels, min_bytes=min_bytes,
                                 retries=retries, timeout=timeout, verbose=verbose):
                continue
            ok, why = band_ok(dest, band)
            if ok:
                paths[band], used[band] = dest, sname
                break
            rejects.append(f"{sname}: {why}")
            os.remove(dest)

        if band not in paths:
            detail = "; ".join(rejects[:3]) if rejects else "no survey coverage"
            reason = f"no usable {band.upper()} image ({detail})"
            print(f"    [REJECT] {gid} ({cls}): {reason}")
            return None, reason          # a galaxy missing any band is useless - stop early

    with open(meta_path, "w") as f:
        json.dump({**meta, "surveys": used}, f)

    return {
        "galaxy_id": gid,
        "label": cls,
        "ir_path": Path(paths["ir"]).as_posix(),
        "vis_path": Path(paths["vis"]).as_posix(),
        "uv_path": Path(paths["uv"]).as_posix(),
        "ra": ra,
        "dec": dec,
        "size_deg": size,
        "source": item.get("source", "curated"),
        "ir_survey": used["ir"],
        "vis_survey": used["vis"],
        "uv_survey": used["uv"],
    }, ""


def _sep(a, b):
    return math.hypot((a["ra"] - b["ra"]) * math.cos(math.radians(a["dec"])), a["dec"] - b["dec"])


def dedupe_targets(items):
    """Drop targets whose field overlaps an earlier one (earlier = curated wins)."""
    kept = []
    for it in items:
        if not any(_sep(it, k) < 0.35 * max(it["size"], k["size"]) for k in kept):
            kept.append(it)
    return kept


def resolve_positions(items, threads):
    todo = [it for it in items if it["ra"] is None]
    if todo:
        with ThreadPoolExecutor(max_workers=threads) as pool:
            for it, res in zip(todo, pool.map(lambda t: resolve_name(t["id"]), todo)):
                if res:
                    it["ra"], it["dec"] = res
    return [it for it in items if it["ra"] is not None]


def download_all(out_dir="./data/raw_fits", csv_file="galaxies.csv", extra_csv="targets_extra.csv",
                 limit=None, threads=2, pixels=256, min_bytes=10000, retries=2, timeout=90,
                 verbose=False, qc=None):
    os.makedirs(out_dir, exist_ok=True)

    curated = build_targets()
    extra = load_extra_targets(extra_csv)
    items = merge_targets(curated, extra)
    before = len(items)
    items = dedupe_targets(resolve_positions(items, threads))
    print(f"Targets: {len(curated)} curated + {len(extra)} extra -> {len(items)} after removing "
          f"duplicates / overlapping fields ({before - len(items)} dropped)")

    if limit:
        counts = {c: 0 for c in SPEC}
        filtered = []
        for it in items:
            if counts[it["class"]] < limit:
                counts[it["class"]] += 1
                filtered.append(it)
        items = filtered

    print(f"Downloading {len(items)} galaxies (pixels={pixels}, threads={threads}, "
          f"quality check={'ON' if qc else 'OFF'})...")
    results, rejected = [], []

    with ThreadPoolExecutor(max_workers=threads) as pool:
        futures = {
            pool.submit(get_galaxy, it, out_dir, pixels, min_bytes, retries, timeout, verbose, qc): it
            for it in items
        }
        for n, f in enumerate(as_completed(futures), 1):
            it = futures[f]
            try:
                res, reason = f.result()
            except Exception as e:
                res, reason = None, f"error: {e}"
            if res:
                results.append(res)
                print(f"  [{n}/{len(items)}] OK   {it['id']:<14} ({it['class']}, {it.get('source')}) -> "
                      f"IR {res['ir_survey']} | VIS {res['vis_survey']} | UV {res['uv_survey']}")
            else:
                rejected.append({"id": it["id"], "class": it["class"], "source": it.get("source"),
                                 "reason": reason})
                print(f"  [{n}/{len(items)}] SKIP {it['id']:<14} ({it['class']})")

    if rejected:
        pd.DataFrame(rejected).to_csv("rejected_targets.csv", index=False)
        print(f"\n{len(rejected)} galaxies rejected (reasons in rejected_targets.csv)")

    if results:
        results.sort(key=lambda r: (r["label"], r["galaxy_id"]))
        df = pd.DataFrame(results)
        assert df[["ir_path", "vis_path", "uv_path"]].notna().all().all()   # only complete 3-band rows
        df.to_csv(csv_file, index=False)
        print(f"Saved {len(results)} complete 3-band galaxies to {csv_file}")
        print(pd.crosstab(df["label"], df["source"]).to_string())
        c = df["label"].value_counts()
        if c.max() > 1.5 * c.min():
            print(f"[NOTE] classes are imbalanced ({c.to_dict()}); train with --weighted or raise --per_class "
                  f"in build_targets.py")
    else:
        print("No complete multi-band galaxies downloaded.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download galaxy FITS images (3 bands, quality-checked)")
    parser.add_argument("--out_dir", default="./data/raw_fits")
    parser.add_argument("--csv", default="galaxies.csv")
    parser.add_argument("--extra", default="targets_extra.csv",
                        help='extra targets from build_targets.py; use --extra "" for the curated list only')
    parser.add_argument("--limit", type=int, default=None, help="max galaxies per class (curated come first)")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--pixels", type=int, default=256, help="Cutout width/height in pixels")
    parser.add_argument("--min_bytes", type=int, default=10000, help="Minimum valid FITS file size in bytes")
    parser.add_argument("--retries", type=int, default=2, help="Retries per survey request")
    parser.add_argument("--timeout", type=int, default=90, help="Request timeout in seconds")
    parser.add_argument("--min_snr", type=float, default=3.0,
                        help="min centre-vs-sky S/N per band (halved for UV)")
    parser.add_argument("--no_quality_check", action="store_true",
                        help="only require that all 3 bands exist (skip the content checks)")
    parser.add_argument("--verbose", action="store_true", help="Print per-survey error/response details")
    parser.add_argument("--insecure", action="store_true",
                        help="Disable SSL certificate verification if Windows CA bundle fails")
    args = parser.parse_args()
    SSL_CTX = make_ssl_context(insecure=args.insecure)
    qc_cfg = None if args.no_quality_check else {"min_snr": args.min_snr}
    download_all(
        out_dir=args.out_dir,
        csv_file=args.csv,
        extra_csv=args.extra,
        limit=args.limit,
        threads=args.threads,
        pixels=args.pixels,
        min_bytes=args.min_bytes,
        retries=args.retries,
        timeout=args.timeout,
        verbose=args.verbose,
        qc=qc_cfg,
    )
