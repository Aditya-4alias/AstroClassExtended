# Galaxy Morphology Classifier (Spiral, Elliptical, Anomaly)

Deep learning computer vision project to classify galaxies into three morphology classes (**Spiral**,
**Elliptical**, **Anomaly/Merger**) from co-registered multi-wavelength (Infrared, Visible, Ultraviolet)
telescope imagery. Three different models vote as an ensemble.

---

## Multi-modal design

Each galaxy has three **modalities** (wavelength bands). `preprocess.py` stores them as the R, G, B channels
of one PNG (R = IR, G = VIS, B = UV). Three models look at that data differently:

| Model | Fusion | What it does |
|---|---|---|
| `resnet18` | early | the 3 bands enter one ResNet-18 as a 3-channel image |
| `multiband` | **late** | one ResNet-18 **stream per band**; the 3x512 features are concatenated and classified together. Band dropout during training stops it relying on a single wavelength |
| `efficientnet_b0` | early | same input, different architecture family |

Their softmax outputs are averaged (by soft voting). 
`python train.py --archs resnet18 multiband` trains stated models.
`multiband` is ~3x the cost of `resnet18` .Use a GPU (Colab T4) or  `--img_size 160` argument when using CPU.

---

## Project Structure

```
Image_Classification/
├── data/raw_fits/             # Raw downloaded FITS images.
├── processed_dataset/         # Aligned 3-channel composites (train / val / test).
├── outputs/                   # Written by train.py (contains models, reports, graphs, etc.).
├── build_targets.py           # ~400 extra balanced targets from SIMBAD -> targets_extra.csv.
├── download.py                # NASA SkyView download, merges curated + extra targets, quality-checks bands.
├── preprocess.py              # WCS alignment, asinh stretch, dataset zipping.
├── dataset.py                 # DataLoaders, flips / 90-degree rotation augmentation.
├── model.py                   # resnet18, efficientnet_b0, multiband (late fusion), resnet50, simple.
├── inference.py               # loads one checkpoint or the whole ensemble.
├── train.py                   # trains the models, scatter plots, ensemble, comparison.
├── predict.py                 # command-line prediction (all models + ensemble).
├── app.py                     # Streamlit UI.
├── galaxies.csv, rejected_targets.csv
└── README.md
```

---

## Workflow

### 1. Install
```bash
pip install -r requirements.txt     # torch, torchvision, scikit-learn, astropy, reproject, streamlit, ...
```

### 2. Choose targets and download (3 bands, quality-checked)
```bash
python build_targets.py --per_class 400      # writes targets_extra.csv
python download.py --threads 4               # curated (~100) + targets_extra.csv
```
* `download.py` reads `targets_extra.csv` (`--extra path`, or `--extra ""` for curated only). This puts the curated  galaxies first, drops duplicate ids and overlapping fields, then downloads IR, VIS and UV for each.
* **Only complete 3-band galaxies are kept.** A band is accepted only if it passes a content check: image is
the right size, <=10% NaN, <=25% zero pixels (outside the survey footprint), real dynamic range, target is
near the image centre (WCS), and a source at the centre stands out from the sky (S/N >= 3, halved for UV
because galaxies are faint there; tune with `--min_snr`). If a survey fails, the next fallback survey for
that band is tried. Galaxies that end up missing a band are skipped and listed with the reason in
`rejected_targets.csv`.
* `--no_quality_check` only requires that all three bands exist. The check needs `astropy` (already required
  by `preprocess.py`).
* `galaxies.csv` also has `source` (curated/catalog) and the survey used for each band.
* Resumable: finished galaxies are cached, just re-run after an interruption.

### 3. Preprocess & Align
```bash
python preprocess.py --csv galaxies.csv --zip --curated_test --exclude exclude_ids.txt --relabel labels_fix.csv
```
* **Centres the intended galaxy.** The catalog position is a starting point; extended sources are detected
  (stars subtracted first) and the one at/nearest the target is centred, so a brighter star or neighbouring
  galaxy doesn't decides the crop. A merging pair is centred as one system. Shifts are capped (20% of the
  field) so a wrong detection cannot throw the crop far away.
* **No footprint edges:** pixels outside any band's coverage are masked; the crop shrinks to avoid them, and
the galaxy is skipped if it cannot.
* **Empty fields are skipped** (nothing detected near the target). `--keep_undetected` keeps them; curated
galaxies are never skipped for this.
* The stretch is scaled from the target galaxy, avoiding stars.
* `--auto_zoom` sizes the crop from the detected galaxy (small galaxies fill more of the frame; off by default).
* `--exclude` takes a `.txt` of ids or `suspect_galaxies.csv` (+ `--exclude_reasons "blank/faint centre" ...`);
  `--relabel` takes a csv `galaxy_id,label`. Curated galaxies are never excluded.
* Writes `preprocess_report.csv` (centring status, shift in px, number of extended sources, skip reasons)
`--curated_test` puts the hand-picked galaxies (trusted labels) in **test** and splits the catalog galaxies
80/20 into train/val.

### 4. Train
```bash
python train.py --benchmark                      # times per model,i.e. benchmark
python train.py --epochs 25 --weighted           # all 3 models + ensemble, auto-resumes
python train.py --archs resnet18 multiband       # choose 2-3 models
python train.py --weight_by_val                  # weight ensemble votes by validation accuracy
```
For Colab: upload the project and the dataset zip, then `!python train.py --epochs 25 --weighted`
(**Note**: earlier `train_colab.ipynb`  contains only *resnet18* single-model training code for testing).

### Outputs (`outputs/`)
```
<arch>/best_model.pth, last_checkpoint.pth
<arch>/classification_report_{val,test}.txt, confusion_matrix_*.png, roc_curve_*.png, pr_curve_*.png, metrics_*.json
<arch>/training_curves.png
<arch>/scatter_train_val.png          # scatter: train vs val accuracy / loss, one point per epoch
<arch>/scatter_embedding_*.png        # scatter: PCA + t-SNE of learned features, misclassified circled
<arch>/scatter_probs_*.png            # scatter: class-probability triangle (corners = confident)
ensemble/                              # same reports + scatter_probs for the averaged ensemble
ensemble.json, model_comparison.png, summary.json
best_galaxy_model.pth                  # best single model by val accuracy (for older tools)
```

### 5. Streamlit UI
```bash
streamlit run app.py
```
Browse the dataset, see the IR/VIS/UV channels, the true label vs. the prediction, **what each model says**,
evaluate a split (per-model and ensemble accuracy, report, confusion matrix, errors) or upload a composite.
Use "Prediction source" in the sidebar to pick the ensemble or a single model.

### 6. Command-line prediction
```bash
python predict.py --image "processed_dataset/test/spiral/M74.png"
python predict.py --ir ".../M51_ir.fits" --vis ".../M51_vis.fits" --uv ".../M51_uv.fits"
python predict.py --image x.png --checkpoint outputs/multiband/best_model.pth   # single model
```

---

##  Notes

* **Class order** is alphabetical (`anomaly, elliptical, spiral`); it is stored in every checkpoint.
* **Evaluation**: quote the **test** numbers; val is used to pick the best epoch. The ensemble is not
  guaranteed to beat its best member on a tiny test set, do compare in `model_comparison.png`.
* **Label quality**: curated labels are reliable; SIMBAD labels are noisier and 'anomaly' is loose.
* **Quality filter vs. balance**: the S/N filter can remove more ellipticals (faint in UV) than spirals.
Check the class counts printed by `download.py`; use `--weighted` when training or lower `--min_snr`.
* **Dataset size**: with ~100 galaxies one mistake moves accuracy by ~5 points; use `build_targets.py`.