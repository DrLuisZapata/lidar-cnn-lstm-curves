# lidar-cnn-lstm-curves

Code, notebooks and results of the article:

> **Short-Horizon Speed and Heading-Rate Prediction from LiDAR Road Geometry and Vehicle Dynamics: A Cross-Validated Benchmark with a Physics-Based Curve-Speed Constraint**<br>
> Luis Miguel Zapata Alvarado, Hamurabi Gamboa Rosales, José Ismael De la Rosa Vargas<br>
> Universidad Tecnológica del Estado de Zacatecas · Universidad Autónoma de Zacatecas<br>
> *Sensors* (MDPI), manuscript sensors-4700056 — under review

## Summary

The study asks whether a LiDAR preview of the road geometry improves the prediction of a vehicle's mean speed and heading rate over the next 1, 2 and 3 s, beyond the recent vehicle dynamics, and whether a physics-based curve-speed constraint (maximum speed on a banked curve) helps training.

- **Features:** road preview offsets at 10–40 m, road curvature, turn-relative banking and grade extracted from LiDAR, with two variants: a label-free height-band ground filter and the PandaSet semantic road labels.
- **Predictors (identical inputs):** persistence, constant-trend extrapolation, ridge regression, MLP, LSTM, GRU, CNN-LSTM, and CNN-LSTM with a physics penalty on speeds above the banked-curve limit.
- **Evaluation:** five-fold sequence-wise cross-validation on 39 PandaSet sequences, per-sequence Wilcoxon tests with Holm correction and bootstrap 95% CIs, input ablations, sensitivity to the friction coefficient μ and the penalty weight λ, curve-stratified errors, validation of the LiDAR geometry against the vehicle's future path, and zero-shot transfer to the 12 KITTI raw Road sequences.

**Main findings:** all learned predictors improved on persistence, and lightweight models were the most accurate (at 1 s, ridge regression reduced the speed and heading-rate MSE by 79% and 38%, against 63% and 20% for the CNN-LSTM). Removing the LiDAR geometry improved accuracy, and the extracted geometry correlated weakly with the path driven. The physics term was inactive for dry-road friction and, for low friction, traded accuracy for compliance with the limit.

## Repository contents

| File / folder | Content |
|---|---|
| `CNN_LSTM_PIDL_PandaSet_KITTI_v5_revision.ipynb` | **Main results** of the revised article: height-band geometry, 39 PandaSet sequences, KITTI zero-shot, all models, ablations, physics sensitivity, curve strata, geometry validation |
| `CNN_LSTM_PIDL_PandaSet_KITTI_v6_semseg.ipynb` | Semantic-road variant (30 PandaSet sequences with labels), with decision criteria fixed before the run |
| `PandaSet_Data_Exploration.ipynb` | Data exploration, coordinate check and world-to-vehicle transformation |
| `results/v5/`, `results/v6/` | Tables (CSV) and figures (PNG/PDF) produced by the two notebooks |
| `make_article_figures.py` | Redraws the result figures of the article from the CSV files in `results/` |
| `make_article_diagrams.py` | Draws the methodology and validation-protocol diagrams of the article |
| `CNN_LSTM_PIDL_PandaSet_KITTI.ipynb` | Notebook of the original submission (superseded by v5 and v6; kept for traceability) |

## Data

- **PandaSet:** https://scale.com/open-datasets/pandaset (Kaggle mirror used in the notebooks). Expected in Google Drive at `MyDrive/pandaset/<sequence>/lidar/` (and `annotations/semseg/` for v6).
- **KITTI raw, Road category:** https://www.cvlibs.net/datasets/kitti/raw_data.php — downloaded automatically by the v5 notebook (12 drives).

The datasets are not redistributed in this repository; please follow their licenses.

## How to run

1. Open the notebook in Google Colab and select a GPU runtime (T4 is enough).
2. If PandaSet is downloaded through Kaggle, add `KAGGLE_USERNAME` and `KAGGLE_KEY` in the Colab **Secrets** panel. Credentials are never written in the notebooks.
3. Run all cells. Features and every finished fold are saved to Google Drive (`pidl_revision_v4/`, `pidl_revision_v5/`, `pidl_revision_v6/`); if Colab disconnects, run all cells again and finished folds are loaded instead of retrained.
4. Run v5 before v6: v6 reuses the KITTI features cached by the v5 pipeline.
5. Each notebook ends by downloading `revision_results.zip` with all tables and figures.

To redraw the figures of the article from the stored results:

```bash
pip install pandas numpy matplotlib
python make_article_figures.py
python make_article_diagrams.py
```

## Citation

If you use this code, please cite the article (the reference will be updated after publication):

```bibtex
@article{zapata2026curves,
  author  = {Zapata Alvarado, Luis Miguel and Gamboa Rosales, Hamurabi and De la Rosa Vargas, Jos{\'e} Ismael},
  title   = {Short-Horizon Speed and Heading-Rate Prediction from {LiDAR} Road Geometry and Vehicle Dynamics: A Cross-Validated Benchmark with a Physics-Based Curve-Speed Constraint},
  journal = {Sensors},
  year    = {2026},
  note    = {Under review}
}
```

## License

MIT License — see `LICENSE`.
