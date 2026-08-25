# oceanembed

**Goal:** oceanembed learns compact, reusable embeddings of ocean state (temperature,
salinity, sea surface height, and currents) from Copernicus Marine GLORYS reanalysis
data over the North Indian Ocean (5–30°N, 45–105°E). These embeddings are trained on
gridded, depth-resolved ocean fields and are meant to serve as a foundation for
downstream tasks like anomaly detection, forecasting, or regional ocean analysis. A
lightweight backend and frontend are included to serve trained embeddings/predictions
and visualize them interactively.

## Project structure

```
oceanembed/
├── configs/         # config.yaml — region, resolution, depth levels, date range
├── data/            # raw and processed ocean datasets (not committed)
├── preprocessing/   # download, regrid, and clean GLORYS data
├── models/          # model architectures
├── training/         # training loops and experiment scripts
├── validation/       # evaluation metrics and validation notebooks/scripts
├── backend/          # FastAPI service to serve embeddings/predictions
├── frontend/         # UI for exploring results
└── notebooks/        # exploratory analysis
```

## Setup

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

You'll also need a free [Copernicus Marine](https://data.marine.copernicus.eu/register)
account. Log in once locally so `copernicusmarine` can authenticate:

```bash
copernicusmarine login
```

Edit `configs/config.yaml` to set your desired `date_range.start` and `date_range.end`
before running any stage.

## Running each stage

1. **Preprocessing** — download GLORYS variables for the configured region/dates and
   regrid them to a common 0.25° grid across the specified depth levels:
   ```bash
   python preprocessing/download_and_regrid.py --config configs/config.yaml
   ```

2. **Training** — train the embedding model on the preprocessed data:
   ```bash
   python training/train.py --config configs/config.yaml
   ```

3. **Validation** — evaluate the trained model against held-out data or downstream tasks:
   ```bash
   python validation/evaluate.py --config configs/config.yaml --checkpoint models/checkpoint.pt
   ```

4. **Backend** — serve the trained model via a REST API:
   ```bash
   uvicorn backend.main:app --reload
   ```

5. **Frontend** — run the UI (see `frontend/README.md` once a framework is chosen) to
   query the backend and visualize embeddings/predictions.

6. **Notebooks** — open `notebooks/` in Jupyter for exploratory analysis at any stage:
   ```bash
   jupyter lab notebooks/
   ```
