## Wind Power Generation Forecasting

Forecast wind power generation (**`Power`**) from weather measurements (temperature, humidity, wind speed/direction, gusts) across **4 locations** using classical ML regression models.

This project is primarily implemented as a Jupyter notebook: `Wind_Power_Generation_Forecasting.ipynb`.

### What’s inside

- **Data**: 4 CSV files (`Location1.csv` … `Location4.csv`) with hourly weather + power.
- **Data merge**: adds a `Location` column and concatenates all locations into `Merged_locations.csv`.
- **EDA**: summary stats, missing/duplicate checks, distributions, scatter plots, and correlation heatmap.
- **Modeling**:
  - Linear Regression
  - Random Forest Regressor
  - XGBoost Regressor (+ GridSearchCV hyperparameter tuning)

### Dataset

Each per-location CSV contains the following columns:

- **Time**: timestamp (string in the raw CSV; the notebook drops this column for modeling)
- **temperature_2m**
- **relativehumidity_2m**
- **dewpoint_2m**
- **windspeed_10m**
- **windspeed_100m**
- **winddirection_10m**
- **winddirection_100m**
- **windgusts_10m**
- **Power**: target variable to predict

The merged dataset (`Merged_locations.csv`) contains all the above plus:

- **Location**: one of `Location1` … `Location4`

In the notebook, `Location` is one-hot encoded into:

- `Location_Location2`, `Location_Location3`, `Location_Location4` (with `drop_first=True`)

### Results (from the notebook)

Models are evaluated on an **80/20 random train-test split** with **StandardScaler** applied to features.

| Model | MAE | MSE | R² |
|---|---:|---:|---:|
| Linear Regression | 0.1377 | 0.0325 | 0.5128 |
| Random Forest Regressor | **0.1066** | **0.0215** | **0.6775** |
| XGBoost Regressor (default) | 0.1157 | 0.0249 | 0.6265 |
| XGBoost (tuned, GridSearchCV) | 0.1133 | 0.0239 | 0.6421 |

**Best-performing model in this notebook run**: Random Forest Regressor (highest R² and lowest MAE/MSE among tested models).

#### What the notebook outputs

- Writes/overwrites **`Merged_locations.csv`** in the project root after concatenating the 4 location files.
- Produces EDA plots (histograms, boxplots, scatter plots, correlation heatmap).
- Prints metrics (MAE/MSE/R²) for each model and prints the best GridSearchCV parameters for tuned XGBoost.

### Project structure

```text
Wind Power Generation Forecasting/
  Wind_Power_Generation_Forecasting.ipynb
  Location1.csv
  Location2.csv
  Location3.csv
  Location4.csv
  Merged_locations.csv
  (supporting PDFs / PPT templates)
```

