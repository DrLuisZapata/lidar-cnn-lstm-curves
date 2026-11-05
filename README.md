# lidar-cnn-lstm-curves

Code for the article **"A Physics-Informed CNN-LSTM for Speed and Heading-Rate Prediction from LiDAR Road Geometry toward Safe Curve Negotiation"** (submitted to *Sensors*, MDPI).

Preprint: https://doi.org/10.20944/preprints202611.0570.v1

Authors: Luis Miguel Zapata Alvarado, Hamurabi Gamboa Rosales, José Ismael De la Rosa Vargas
(Universidad Autónoma de Zacatecas and Universidad Tecnológica del Estado de Zacatecas, Mexico)

## Overview

A multi-task CNN-LSTM predicts the change of the vehicle heading rate and speed over horizons of 1, 2, and 3 s from a 2 s window of features: a LiDAR-based preview of the road geometry (lateral offset of the road at 10–40 m ahead), banking, grade, speed, and heading rate. A one-sided physics-informed penalty discourages predicted speeds above the maximum cornering speed of a banked curve with friction. The model is trained on PandaSet and evaluated on held-out sequences over five sequence-wise splits and, without retraining, on KITTI raw (Road category), against a persistence baseline and an ablation without the physics term.

## Repository contents

| File | Description |
|---|---|
| `PandaSet_Data_Exploration.ipynb` | Coordinate-frame check, world-to-vehicle transform, LiDAR point-cloud visualization, vehicle speed from poses, and bird's-eye view of the road (Figures 2–4 of the article). |
| `CNN_LSTM_PIDL_PandaSet_KITTI.ipynb` | Feature extraction for PandaSet and KITTI, CNN-LSTM with physics-informed loss, five sequence-wise splits, three horizons, persistence baseline, ablation, results tables, and result figures. |

Run the notebooks in this order: first `PandaSet_Data_Exploration.ipynb`, then `CNN_LSTM_PIDL_PandaSet_KITTI.ipynb`.

## Requirements

- Google Colab with a GPU runtime (tested on an NVIDIA T4).
- Python libraries available by default in Colab (NumPy, pandas, scikit-learn, TensorFlow/Keras, Matplotlib). The exploration notebook also installs Open3D.

## Data

- **PandaSet**: https://scale.com/open-datasets/pandaset
  Place the sequences in Google Drive under `/content/drive/MyDrive/pandaset/<sequence>/` (e.g., `001/lidar/00.pkl` and `001/lidar/poses.json`). Change `RUTA_PANDASET` / `PANDASET_ROOT` in the notebooks if you use another location.
- **KITTI raw (Road category)**: https://www.cvlibs.net/datasets/kitti/raw_data.php
  The CNN-LSTM notebook downloads drives `2011_09_26_drive_0015`, `0027`, and `0028` automatically from the official mirror.

## Kaggle credentials (optional)

Only needed if you download PandaSet through the Kaggle API.

1. In Kaggle, go to **Settings → API → Create New Token**.
2. In Colab, open the **Secrets** panel (key icon) and add `KAGGLE_USERNAME` and `KAGGLE_KEY`.
3. Enable notebook access for both secrets.

**Never commit your `kaggle.json` file or your credentials to this repository.**

## Outputs

The CNN-LSTM notebook saves:

- `results_all_splits.csv`, `results_summary_mean_std.csv`, and `results_vs_persistence.csv` (all metrics per split, mean ± std, and comparison with persistence);
- the result figures in PDF and 300-dpi PNG in the `figures/` folder.

Random seeds are fixed. Small differences in the last decimals may occur between runs because GPU training is not fully deterministic.

## Citation

If you use this code, please cite the preprint (the citation will be updated when the peer-reviewed article is published):

Zapata Alvarado, L.M.; Gamboa Rosales, H.; De la Rosa Vargas, J.I. A Physics-Informed CNN-LSTM for Speed and Heading-Rate Prediction from LiDAR Road Geometry Toward Safe Curve Negotiation. *Preprints* **2026**. https://doi.org/10.20944/preprints202611.0570.v1

```bibtex
@article{202611.0570,
  doi       = {10.20944/preprints202611.0570.v1},
  url       = {https://doi.org/10.20944/preprints202611.0570.v1},
  year      = 2026,
  month     = {November},
  publisher = {Preprints},
  author    = {Zapata Alvarado, Luis Miguel and Gamboa Rosales, Hamurabi and De la Rosa Vargas, José Ismael},
  title     = {A Physics-Informed CNN-LSTM for Speed and Heading-Rate Prediction from LiDAR Road Geometry Toward Safe Curve Negotiation},
  journal   = {Preprints}
}
```

## License

This project is released under the MIT License (see `LICENSE`).
