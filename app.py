"""
Streamlit front-end for the galaxy classifier (multi-model ensemble).

    streamlit run app.py
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

from inference import Ensemble, discover_checkpoints

st.set_page_config(page_title="Galaxy Classifier", page_icon="🌌", layout="wide")
torch.set_num_threads(min(4, os.cpu_count() or 2))

SPLITS = ["train", "val", "test"]
PALETTE = {"spiral": "#2b5c8f", "elliptical": "#d97724", "anomaly": "#bd3333"}
BANDS = ["Infrared (R)", "Visible (G)", "Ultraviolet (B)"]
IMG_EXT = (".png", ".jpg", ".jpeg")
ENS = "Ensemble (all models)"


@st.cache_resource(show_spinner="Loading models...")
def load_ens(ckpts, weights, stamp):
    return Ensemble(list(ckpts), weights=list(weights), device=torch.device("cpu"))


@st.cache_data(show_spinner=False)
def scan_dataset(data_dir, stamp):
    rows = []
    for split in SPLITS:
        sdir = os.path.join(data_dir, split)
        if not os.path.isdir(sdir):
            continue
        for cls in sorted(os.listdir(sdir)):
            cdir = os.path.join(sdir, cls)
            if not os.path.isdir(cdir):
                continue
            for fn in sorted(os.listdir(cdir)):
                if fn.lower().endswith(IMG_EXT):
                    rows.append({"split": split, "label": cls, "galaxy_id": os.path.splitext(fn)[0],
                                 "path": os.path.join(cdir, fn)})
    return pd.DataFrame(rows, columns=["split", "label", "galaxy_id", "path"])


@st.cache_data(show_spinner=False)
def load_catalog(csv_path, stamp):
    try:
        df = pd.read_csv(csv_path)
        return df.drop_duplicates("galaxy_id").set_index("galaxy_id")
    except Exception:
        return None


@st.cache_data(show_spinner="Running the models...")
def predict_paths(ckpts, weights, stamp, paths):
    ens = load_ens(ckpts, weights, stamp)
    return ens.predict_paths(list(paths))


def pick(ens_probs, per_model, source):
    return ens_probs if source == ENS else per_model[source]


def prob_chart(classes, probs):
    fig, ax = plt.subplots(figsize=(4.8, 2.0 + 0.2 * len(classes)))
    bars = ax.barh(range(len(classes)), probs * 100, edgecolor="black",
                   color=[PALETTE.get(c, "#777777") for c in classes])
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels([c.capitalize() for c in classes])
    ax.set_xlim(0, 105)
    ax.set_xlabel("Confidence (%)")
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    for b in bars:
        ax.text(b.get_width() + 1, b.get_y() + b.get_height() / 2, f"{b.get_width():.1f}%", va="center", fontsize=9)
    fig.tight_layout()
    return fig


def show_fig(fig):
    st.pyplot(fig)
    plt.close(fig)


def verdict_banner(true, pred, conf):
    if true == pred:
        st.success(f"Correct: {pred.capitalize()} ({conf:.1f}% confidence)")
    else:
        st.error(f"Wrong: predicted {pred.capitalize()} ({conf:.1f}%), true class is {true.capitalize()}")


def agreement_table(classes, ens_probs, per_model, i=0):
    rows = [{"model": n, "predicted": classes[int(p[i].argmax())], "confidence %": round(100 * float(p[i].max()), 1)}
            for n, p in per_model.items()]
    if len(per_model) > 1:
        rows.append({"model": "ensemble", "predicted": classes[int(ens_probs[i].argmax())],
                     "confidence %": round(100 * float(ens_probs[i].max()), 1)})
    return pd.DataFrame(rows)


st.sidebar.title("🌌 Galaxy Classifier")
models_path = st.sidebar.text_input("Models folder or checkpoint", "outputs")
data_dir = st.sidebar.text_input("Dataset folder", "processed_dataset")
csv_path = st.sidebar.text_input("Catalog CSV (optional)", "galaxies.csv")

ckpts, ens_weights = discover_checkpoints(models_path)
ens, source = None, ENS
if ckpts:
    stamp = float(sum(os.path.getmtime(p) for p in ckpts))
    ckpts_t, weights_t = tuple(ckpts), tuple(ens_weights)
    try:
        ens = load_ens(ckpts_t, weights_t, stamp)
    except Exception as e:
        st.sidebar.error(f"Could not load models: {e}")
else:
    st.sidebar.warning("No checkpoints found - you can browse images, but predictions are disabled. "
                       "Train first with `python train.py`.")

if ens:
    for m in ens.members:
        i = m.info
        st.sidebar.caption(f"**{m.name}** · {i['img_size']}px · epoch {i['epoch']} · "
                           f"val acc {100 * (i['best_val_acc'] or 0):.1f}%")
    if len(ens.members) > 1:
        source = st.sidebar.selectbox("Prediction source", [ENS] + ens.names)
    else:
        source = ens.names[0]

if not os.path.isdir(data_dir):
    st.error(f"Dataset folder '{data_dir}' not found. Run `python preprocess.py` first "
             f"(or fix the path in the sidebar).")
    st.stop()

df = scan_dataset(data_dir, os.path.getmtime(data_dir))
if df.empty:
    st.error(f"No images found in {data_dir}/(train|val|test)/<class>/*.png")
    st.stop()

catalog = load_catalog(csv_path, os.path.getmtime(csv_path)) if os.path.exists(csv_path) else None

st.sidebar.subheader("Dataset")
st.sidebar.dataframe(pd.crosstab(df["label"], df["split"]).reindex(columns=[s for s in SPLITS if s in set(df["split"])]),
                     use_container_width=True)
st.sidebar.caption("Predictions on **train** images are optimistic: the model has already seen them.")

classes = ens.classes if ens else sorted(df["label"].unique())
tab_browse, tab_eval, tab_upload = st.tabs(["🔭 Browse & predict", "📊 Evaluate split", "📤 Upload image"])

with tab_browse:
    c1, c2, c3 = st.columns([1.2, 1.2, 1.6])
    avail_splits = [s for s in SPLITS if s in set(df["split"])]
    sel_splits = c1.multiselect("Split", avail_splits, default=avail_splits)
    all_labels = sorted(df["label"].unique())
    sel_labels = c2.multiselect("True class", all_labels, default=all_labels)
    query = c3.text_input("Search galaxy id", "", placeholder="e.g. NGC4 or M8")

    view = df[df["split"].isin(sel_splits) & df["label"].isin(sel_labels)]
    if query:
        view = view[view["galaxy_id"].str.contains(query, case=False, regex=False)]
    view = view.sort_values(["label", "galaxy_id"]).reset_index(drop=True)

    if view.empty:
        st.info("No galaxy matches the current filters.")
    else:
        options = [f"{r.galaxy_id}  ·  {r.label}  ·  {r.split}" for r in view.itertuples()]
        if st.session_state.get("galaxy_choice") not in options:
            st.session_state["galaxy_choice"] = options[0]

        def _step(delta):
            cur = options.index(st.session_state["galaxy_choice"])
            st.session_state["galaxy_choice"] = options[(cur + delta) % len(options)]

        nav1, nav2, nav3 = st.columns([1, 6, 1])
        nav1.button("◀ Prev", on_click=_step, args=(-1,), use_container_width=True)
        nav3.button("Next ▶", on_click=_step, args=(1,), use_container_width=True)
        choice = nav2.selectbox(f"Galaxy ({len(options)} match)", options, key="galaxy_choice",
                                label_visibility="collapsed")

        row = view.iloc[options.index(choice)]
        img = Image.open(row["path"]).convert("RGB")

        left, right = st.columns([1, 1])
        with left:
            st.subheader(row["galaxy_id"])
            st.image(img, caption="Composite  R = infrared · G = visible · B = ultraviolet", width=400)
            arr = np.array(img)
            cols = st.columns(3)
            for i, col in enumerate(cols):
                col.image(arr[:, :, i], caption=BANDS[i], clamp=True, width=125)

        with right:
            st.subheader("Result")
            m1, m2 = st.columns(2)
            m1.metric("True class", row["label"].capitalize())
            m2.metric("Split", row["split"])

            if ens is None:
                st.info("Load a trained model (see sidebar) to see predictions.")
            else:
                ens_p, per = predict_paths(ckpts_t, weights_t, stamp, (row["path"],))
                probs = pick(ens_p, per, source)[0]
                pred = classes[int(np.argmax(probs))]
                verdict_banner(row["label"], pred, 100 * probs.max())
                if row["split"] == "train":
                    st.warning("This galaxy was in the **training** set, so the result is not a fair test.")
                show_fig(prob_chart(classes, probs))
                if len(per) > 1:
                    st.caption("What each model says")
                    st.dataframe(agreement_table(classes, ens_p, per), use_container_width=True, hide_index=True)

            if catalog is not None and row["galaxy_id"] in catalog.index:
                meta = catalog.loc[row["galaxy_id"]]
                bits = [f"RA {meta['ra']:.4f}°, Dec {meta['dec']:.4f}°"]
                if "size_deg" in meta and not pd.isna(meta["size_deg"]):
                    bits.append(f"field {meta['size_deg']:.2f}°")
                if "source" in meta and not pd.isna(meta["source"]):
                    bits.append(f"label source: {meta['source']}")
                st.caption(" · ".join(bits))

with tab_eval:
    if ens is None:
        st.info("Train a model first (`python train.py`), then evaluate it here.")
    else:
        order = [s for s in ["test", "val", "train"] if s in set(df["split"])]
        split = st.radio("Split to evaluate", order, horizontal=True)
        sub = df[(df["split"] == split) & df["label"].isin(classes)].reset_index(drop=True)
        if split == "train":
            st.warning("Train-split accuracy is optimistic. Use **test** for an honest number.")

        if not sub.empty and st.button(f"Evaluate {len(sub)} images", type="primary"):
            st.session_state["eval_split"] = split
        if not sub.empty and st.session_state.get("eval_split") == split:
            ens_p, per = predict_paths(ckpts_t, weights_t, stamp, tuple(sub["path"]))
            probs = pick(ens_p, per, source)
            y_true = np.array([classes.index(l) for l in sub["label"]])
            y_pred = probs.argmax(1)
            labels = list(range(len(classes)))
            names = [c.capitalize() for c in classes]

            st.caption(f"Showing: **{source}**")
            a, b, c = st.columns(3)
            a.metric("Accuracy", f"{100 * (y_true == y_pred).mean():.1f}%")
            b.metric("Images", len(sub))
            c.metric("Misclassified", int((y_true != y_pred).sum()))

            if len(per) > 1:
                cmp_rows = [{"model": n, "accuracy %": round(100 * float((p.argmax(1) == y_true).mean()), 1)}
                            for n, p in per.items()]
                cmp_rows.append({"model": "ensemble",
                                 "accuracy %": round(100 * float((ens_p.argmax(1) == y_true).mean()), 1)})
                st.dataframe(pd.DataFrame(cmp_rows), use_container_width=True, hide_index=True)

            g1, g2 = st.columns([1.2, 1])
            with g1:
                rep = classification_report(y_true, y_pred, labels=labels, target_names=names,
                                            zero_division=0, output_dict=True)
                rep_df = pd.DataFrame(rep).T.round(3)
                rep_df["support"] = rep_df["support"].astype(int)
                st.dataframe(rep_df, use_container_width=True)
            with g2:
                fig, ax = plt.subplots(figsize=(4.2, 3.6))
                ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred, labels=labels),
                                       display_labels=names).plot(cmap="Blues", ax=ax, colorbar=False)
                ax.set_title(f"Confusion matrix ({split})")
                fig.tight_layout()
                show_fig(fig)

            wrong = np.where(y_true != y_pred)[0]
            st.subheader(f"Misclassified galaxies ({len(wrong)})")
            if len(wrong) == 0:
                st.success("No mistakes on this split.")
            else:
                table = pd.DataFrame({
                    "galaxy": sub["galaxy_id"].to_numpy()[wrong],
                    "true": [classes[i] for i in y_true[wrong]],
                    "predicted": [classes[i] for i in y_pred[wrong]],
                    "confidence %": (100 * probs[wrong].max(1)).round(1),
                }).sort_values("confidence %", ascending=False)
                st.dataframe(table, use_container_width=True, hide_index=True)

                shown = wrong[:12]
                cols = st.columns(4)
                for k, i in enumerate(shown):
                    cols[k % 4].image(Image.open(sub["path"][i]).convert("RGB"), width=170,
                                      caption=f"{sub['galaxy_id'][i]}: {classes[y_true[i]]} → "
                                              f"{classes[y_pred[i]]} ({100 * probs[i].max():.0f}%)")
                if len(wrong) > len(shown):
                    st.caption(f"Showing {len(shown)} of {len(wrong)}; the table lists all of them.")

with tab_upload:
    if ens is None:
        st.info("Train a model first (`python train.py`).")
    else:
        st.caption("Use a composite made by `preprocess.py` (R = infrared, G = visible, B = ultraviolet). "
                   "An ordinary colour photo will give meaningless results.")
        up = st.file_uploader("Galaxy composite image", type=["png", "jpg", "jpeg"])
        if up is not None:
            img = Image.open(up).convert("RGB")
            ens_p, per = ens.predict_pil(img)
            probs = ens_p if source == ENS else per[source]
            pred = classes[int(np.argmax(probs))]
            u1, u2 = st.columns(2)
            u1.image(img, caption=up.name, width=380)
            with u2:
                st.success(f"Prediction ({source}): {pred.capitalize()} ({100 * probs.max():.1f}%)")
                show_fig(prob_chart(classes, probs))
                if len(per) > 1:
                    st.dataframe(agreement_table(classes, ens_p[None, :], {k: v[None, :] for k, v in per.items()}),
                                 use_container_width=True, hide_index=True)
