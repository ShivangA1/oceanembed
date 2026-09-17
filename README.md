# oceanembed

**Goal:** oceanembed learns compact, reusable embeddings of ocean state (temperature, salinity, sea surface height, and currents) from Copernicus Marine GLORYS reanalysis data over the North Indian Ocean (5–30°N, 45–105°E). These embeddings are trained on gridded, depth-resolved ocean fields and serve as a foundation for downstream tasks like anomaly detection, forecasting, or regional ocean analysis. A lightweight AI service and frontend are included to serve trained embeddings/predictions and visualize them interactively.

Built for **Smart India Hackathon (SIH) Problem Statement #26066**, under INCOIS/MoES: *"Satellite Embedding-Based Subsurface Ocean Temperature Reconstruction."*

## Project Structure

```
oceanembed/
├── ai-service/          # Data pipeline, model, and API — preprocessing, training,
│                        # validation, and the FastAPI backend that serves predictions
├── frontend/            # UI for exploring results (map + depth-temperature profiles)
├── requirements.txt     # Python dependencies for ai-service
├── run_ai.bat            # Windows quick-start script for the AI service
├── run_frontend.bat       # Windows quick-start script for the frontend
└── .gitignore
```

## Approach

- A **CNN Satellite Encoder** compresses a 7-channel input patch (surface satellite observations) into a latent embedding.
- An **MLP Temperature Decoder** maps that embedding to a 15-depth temperature profile.

## Region & Depth Coverage

| | Value |
|---|---|
| Latitude range | 5°N – 30°N |
| Longitude range | 45°E – 105°E |
| Spatial resolution | 0.25° |
| Depth levels (m) | 0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000 |

## Tech Stack

- **Data & ML:** `copernicusmarine`, `xarray`, `xarray-regrid`, `netCDF4`, `h5netcdf`, `dask`, `torch`, `numpy`, `pandas`, `scipy`, `scikit-learn`, `matplotlib`
- **API:** `fastapi`, `uvicorn`, `pydantic`
- **Frontend:** React + Leaflet (clickable map with depth-temperature profile charts)

## Setup

```bash
git clone https://github.com/ShivangA1/oceanembed.git
cd oceanembed
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

You'll also need a free [Copernicus Marine](https://data.marine.copernicus.eu/register) account. Log in once locally so `copernicusmarine` can authenticate:

```bash
copernicusmarine login
```

## Running

**On Windows**, the included batch scripts handle setup + launch:

```bash
run_ai.bat         # starts the AI service (preprocessing/training/API, as configured)
run_frontend.bat    # starts the frontend
```

**Manually:**

```bash
# AI service (from ai-service/)
uvicorn main:app --reload

# Frontend (from frontend/)
npm install
npm start
```

## Validation

Model predictions are validated against held-out **ARGO float** profiles (via `argopy`), not used during training, comparing correlation, RMSE, and bias at each depth level.

## License

TBD