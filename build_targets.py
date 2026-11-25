"""
Build a large, class-balanced list of extra galaxy targets from the SIMBAD catalog.
"""

import io
import re
import ssl
import argparse
import urllib.parse
import urllib.request
import numpy as np
import pandas as pd
import urllib.error

TAP_URLS = [
    "https://simbad.cds.unistra.fr/simbad/sim-tap/sync",
    "https://simbad.cfa.harvard.edu/simbad/sim-tap/sync",
    "https://simbad.idia.ac.za/simbad/sim-tap/sync",
]
UA = {"User-Agent": "Mozilla/5.0 AstroResearcher/1.0"}

Q_MORPH = ("SELECT TOP 40000 main_id, ra, dec, galdim_majaxis, morph_type, otype FROM basic "
           "WHERE morph_type IS NOT NULL AND galdim_majaxis BETWEEN {lo} AND {hi}")
Q_INTER = ("SELECT TOP 8000 main_id, ra, dec, galdim_majaxis, morph_type, otype FROM basic "
           "WHERE otype IN ('IG','PaG') AND galdim_majaxis BETWEEN {lo} AND {hi}")

SPIRAL_RE = re.compile(r"^S(?:AB|A|B)?(?:\([a-z]+\))?(?:[abcd]m?|m)")
ELLIP_RE = re.compile(r"^E[0-7]?\+?$")

HELP = """

If the error is 'getaddrinfo failed' (Windows error 11001) your PC cannot resolve the host name (DNS):
  * open https://simbad.idia.ac.za in a browser - if that fails too, it is a network problem, not the script
  * switch DNS to a public one (Windows: Settings > Network > adapter > DNS = 1.1.1.1 / 8.8.8.8), or try another
    network / turn a VPN off or on; school, office and some ISP networks block astronomy sites
  * check proxy settings (urllib uses the system proxy; set HTTPS_PROXY if you need one)
  * no luck? run without SIMBAD:   python download.py --extra ""     (the curated 100 galaxies)
    or use a catalog file exported on another machine:  python build_targets.py --local_csv catalog.csv
"""


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


def tap(query, ctx):
    body = urllib.parse.urlencode({"REQUEST": "doQuery", "LANG": "ADQL",
                                   "FORMAT": "csv", "QUERY": query}).encode()
    errors = []
    for url in TAP_URLS:
        host = urllib.parse.urlparse(url).netloc
        try:
            req = urllib.request.Request(url, data=body, headers=UA)
            with urllib.request.urlopen(req, timeout=300, context=ctx) as r:
                text = r.read().decode("utf-8", errors="ignore")
            if text.lstrip().startswith("<"):
                errors.append(f"{host}: server error {text[:600]!r}")
                continue
            return pd.read_csv(io.StringIO(text))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="ignore")[:600]   # the server explains what is wrong
            errors.append(f"{host}: HTTP {e.code} {detail!r}")
        except Exception as e:
            errors.append(f"{host}: {e}")
    raise RuntimeError("Could not query SIMBAD:\n  " + "\n  ".join(errors) + HELP)

def classify(morph, otype):
    m = "" if pd.isna(morph) else str(morph).strip()
    if "pec" in m:
        return "anomaly"
    if not m or ":" in m or "?" in m:
        return None
    if ELLIP_RE.match(m):
        return "elliptical"
    if SPIRAL_RE.match(m):
        return "spiral"
    return None


def safe_id(main_id, used):
    s = re.sub(r"\s+", "", str(main_id)).replace("+", "p")
    s = re.sub(r"[^A-Za-z0-9_.\-]", "", s) or "obj"
    base, k = s, 2
    while s in used:
        s, k = f"{base}_{k}", k + 1
    used.add(s)
    return s


def field_size(major_arcmin):
    return round(float(np.clip(2.0 * major_arcmin / 60.0, 0.06, 1.2)), 3)


def main():
    ap = argparse.ArgumentParser(description="Build extra galaxy targets from SIMBAD")
    ap.add_argument("--per_class", type=int, default=170, help="catalog galaxies per class")
    ap.add_argument("--max_spiral", type=int, default=400, help="cap for the (plentiful) spiral class")
    ap.add_argument("--min_arcmin", type=float, default=2.0, help="min apparent major axis")
    ap.add_argument("--max_arcmin", type=float, default=40.0, help="max apparent major axis")
    ap.add_argument("--out", default="targets_extra.csv")
    ap.add_argument("--local_csv", help="use a catalog CSV instead of querying SIMBAD")
    ap.add_argument("--insecure", action="store_true", help="Disable SSL certificate verification if Windows CA bundle fails")
    args = ap.parse_args()
    ctx = make_ssl_context(insecure=args.insecure)

    if args.local_csv:
        print(f"Reading {args.local_csv} ...")
        morph = pd.read_csv(args.local_csv)
        if "otype" not in morph.columns:
            morph["otype"] = ""
        morph = morph[morph["galdim_majaxis"].between(args.min_arcmin, args.max_arcmin)]
        inter = morph[morph["otype"].isin(["IG", "PaG"])]
        print(f"  {len(morph)} rows")
    else:
        print("Querying SIMBAD (morphological types)...")
        morph = tap(Q_MORPH.format(lo=args.min_arcmin, hi=args.max_arcmin), ctx)
        print(f"  {len(morph)} rows")
        print("Querying SIMBAD (interacting galaxies / pairs)...")
        try:
            inter = tap(Q_INTER.format(lo=args.min_arcmin, hi=args.max_arcmin), ctx)
        except RuntimeError as e:
            print(f"  [WARN] interacting-galaxy query failed, continuing without it:\n{e}")
            inter = morph.iloc[0:0]
        print(f"  {len(inter)} rows")

    morph = morph.dropna(subset=["ra", "dec", "galdim_majaxis"]).copy()
    inter = inter.dropna(subset=["ra", "dec", "galdim_majaxis"]).copy()
    morph["cls"] = [classify(m, o) for m, o in zip(morph["morph_type"], morph["otype"])]
    inter["cls"] = "anomaly"

    pool = pd.concat([inter, morph[morph["cls"].notna()]], ignore_index=True)
    pool = pool.drop_duplicates(subset="main_id")

    selected_xy = np.empty((0, 2))
    selected_sz = np.empty(0)
    rows, used = [], set()

    for cls in ["anomaly", "elliptical", "spiral"]:
        cand = pool[pool["cls"] == cls].sort_values("galdim_majaxis", ascending=False)
        print(f"{cls}: {len(cand)} candidates")
        taken = 0
        for r in cand.itertuples():
            limit = args.max_spiral if cls == "spiral" else args.per_class
            if taken >= limit:
                break
            size = field_size(r.galdim_majaxis)
            if len(selected_sz):
                dx = (selected_xy[:, 0] - r.ra) * np.cos(np.radians(r.dec))
                dy = selected_xy[:, 1] - r.dec
                sep = np.hypot(dx, dy)
                if np.any(sep < 0.35 * np.maximum(selected_sz, size)):
                    continue
            selected_xy = np.vstack([selected_xy, [r.ra, r.dec]])
            selected_sz = np.append(selected_sz, size)
            rows.append({"id": safe_id(r.main_id, used), "class": cls, "ra": round(r.ra, 5),
                         "dec": round(r.dec, 5), "size": size, "source": "catalog",
                         "morph": "" if pd.isna(r.morph_type) else r.morph_type,
                         "main_id": re.sub(r"\s+", " ", str(r.main_id))})
            taken += 1
        print(f"  selected {taken}")
        if taken < args.per_class:
            print(f"  [NOTE] only {taken}/{args.per_class} available for '{cls}' with these size limits "
                  f"(try --min_arcmin < 2.0)")

    if not rows:
        raise SystemExit("No targets selected - check the SIMBAD query output above.")

    pd.DataFrame(rows).to_csv(args.out, index=False)
    counts = pd.DataFrame(rows)["class"].value_counts().to_dict()
    print(f"\nSaved {len(rows)} targets to {args.out}  {counts}")
    print("Next: python download.py")


if __name__ == "__main__":
    main()
