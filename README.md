# Face Presentation Attack Detection — Internal Embeddings

A two-stage deep learning pipeline that detects whether a face presented to a recognition system is real or a spoof attack — using pre-extracted 512-D face embeddings from YuNet.

Built as part of Machine Learning in Biometrics coursework at **Paderborn University**.

---

## The Problem

Face recognition systems can be fooled. A printed photo, a video replay on a phone screen, a 3D mask — these are all presentation attacks. The harder challenge is not detecting attacks on the same dataset you trained on. It is detecting attacks on datasets you have never seen before. That cross-dataset generalisation gap is what this project directly addresses.

---

## How It Works

### Stage 1 — Pretrain
A compact regularized MLP is trained on 512-D face embeddings across 10 attack types (bonafide, glasses, flexible mask, makeup, mannequin, paper mask, print, replay, rigid mask, tattoo).

- Focal loss with dynamic alpha handles class imbalance between bonafide and attack samples
- L2 normalisation followed by StandardScaler applied to embeddings
- EarlyStopping on validation accuracy, ModelCheckpoint saves best weights

### Stage 2 — Leave-One-Out Fine-tuning
The pretrained model is fine-tuned and evaluated using a Leave-One-Out (LOO) protocol across three standard PAD datasets: Replay-Attack, MSU-MFSD, and OULU-NPU.

For each split: train on two datasets, test on the held-out one.

Fine-tuning runs in two phases:
- Phase 1: base learning rate (2e-5), monitors val AUC
- Phase 2: 5× smaller learning rate for precision tuning

After fine-tuning, temperature scaling is applied to calibrate probabilities on the validation set before final threshold selection.

---

## Results

| Test Dataset | Strategy | Test HTER % | Val AUC (cal) % | Temperature T |
|---|---|---|---|---|
| Replay-Attack | EER | 45.60 | 79.29 | 1.5 |
| MSU-MFSD | EER | **39.51** | 74.70 | 1.4 |
| OULU-NPU | EER | 48.08 | 86.79 | 1.3 |

The gap between validation AUC and test HTER is expected — cross-dataset PAD is a well-known open problem. The best result (MSU-MFSD, 39.5% HTER) is consistent with published embedding-based PAD baselines on this LOO protocol. The OULU-NPU split is the hardest because it contains digital attacks not present in the other two datasets.

---

## Tech Stack

- Python, TensorFlow / Keras
- scikit-learn (metrics, StandardScaler, class weights)
- NumPy, Pandas, joblib
- SLURM GPU cluster (training environment)

---

## Repository Structure

```
face-recognition-pad/
├── face_pad_with_outputs.ipynb   # Full pipeline with real training outputs
├── README.md
└── requirements.txt
```

---

## How to Run

This pipeline expects pre-extracted 512-D face embeddings as JSON input files. Raw video or image data is not included due to dataset licensing restrictions (Replay-Attack, MSU-MFSD, and OULU-NPU require registration with their respective institutions).

```bash
pip install -r requirements.txt
jupyter notebook face_pad_with_outputs.ipynb
```

Update `BASE_PATH` in the Configuration cell to point to your local embedding files.

### Expected data format

Each JSON file should follow this structure:

```json
[
  {"embedding_vector": [[0.12, -0.34, ..., 0.87]]},
  ...
]
```

File naming convention used in this project:
- Stage 1: `train_bonafide_*.json`, `train_Replay_*.json`, etc.
- Stage 2 LOO: `replay_train_real_*.json`, `msu_test_attack_*.json`, etc.

---

## Background

This project is part of broader face recognition research at Paderborn University. ArcFace models (ResNet-18/50/100) trained in parallel work achieved **96.12% TAR @ FAR=1e-4 on IJB-C** — competitive with published SOTA on that benchmark.

The PAD module in this repo operates on top of those embeddings, adding a lightweight spoof detection layer without retraining the underlying recognition model.

---

## License

MIT License — see LICENSE file.

---

## Author

**Ananya Sathyanarayana**
M.Sc. Computer Science, Paderborn University
[LinkedIn](https://linkedin.com/in/ananya-sathyanarayana) | [GitHub](https://github.com/MsuniqueStar)
