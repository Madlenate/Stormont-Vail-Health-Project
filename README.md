# Emergency Department Admissions Analysis

Analysis of hospital encounter data to understand patterns in Emergency Department (ED) visits — focused on **abdominal/pelvic pain** and **throat/chest pain** cases — and to predict which ED visits result in hospital admission.

## Project Structure

| Notebook | Purpose |
|---|---|
| `Data_Cleaning.ipynb` | Loads raw CSVs, explores each table, de-duplicates keys, and joins everything into one combined dataset |
| `Data_visualizing.ipynb` | Filters to the two ED diagnosis groups of interest, visualizes demographic/seasonal trends, forecasts admission volume, and trains classifiers to predict hospital admission |

## Data Sources

Raw tables (CSV), joined via DuckDB SQL:

- `encounters.csv` — 7,675,801 rows; core visit-level data (dates, admission/discharge info, visit type, ED/inpatient flags)
- `patients.csv` — demographics (race, ethnicity, sex, marital status, smoking status, vital status, birth year bin)
- `diagnosis.csv` — 1,531,262 rows; ICD-10 diagnosis names/values, grouped by `GroupName`/`GroupCode`
- `providers.csv` — clinician/department assignment info
- `departments.csv` — 11,597 rows; department specialty and location info
- `social_determinants.csv`, `tigercensuscodes.csv` — loaded but not used in the final analysis

## Data Cleaning (`Data_Cleaning.ipynb`)

1. Loaded each CSV into DuckDB and inspected shape/schema/nulls for every table.
2. Identified duplicate keys in `diagnosis`, `providers`, `patients`, and `departments`; de-duplicated each with a `ROW_NUMBER()` partition before joining.
3. Left-joined `encounters` → `patients`, `diagnosis`, `providers`, `departments` into a single **`combined_data`** table (7,675,801 rows).
4. Explored ED visit volume by diagnosis `GroupName` and found the two largest non-generic categories were:
   - **Abdominal and pelvic pain**
   - **Pain in throat and chest**

   (General examination visits were excluded as uninformative.)

## Key Findings (`Data_visualizing.ipynb`)

### 1. Diagnosis volume (ED visits)
Top ED diagnosis groups by encounter count include Pain in throat and chest (12,023), Abdominal and pelvic pain (11,956), Dorsalgia, Other sepsis, Nausea and vomiting, and COVID-19, among others. Filtering to the two focus groups leaves **23,979 encounters** after column cleanup (dropping fully-null and redundant time/key columns).

### 2. Demographic breakdown
Visit counts for the two diagnosis groups were compared across `FirstRace`, `SexAssignedAtBirth`, and `SmokingStatus` to see whether either condition skews toward a particular population.

### 3. Seasonality
Monthly ED admission counts (2022–2025) show **two recurring peaks — around June, and again in October/November** — for both diagnosis groups, suggesting a seasonal pattern worth further investigation (e.g., PCA was run to see which variables associate most with these spikes).

### 4. Admission volume & rate over time
A monthly time series of total ED visits vs. hospital admission rate reveals a **COVID-era effect**: admission *rate* spiked during 2022 (COVID-19 wave) and has since receded, but the **total number of admitted patients has not returned to pre-COVID levels** — overall visit volume grew even as the rate normalized. A 6-month forecast was produced using Holt-Winters Exponential Smoothing.

### 5. Class imbalance
Among all ED visits: **54,135 admitted vs. 207,323 not admitted** (~79% not admitted). This heavy imbalance was addressed with **SMOTE** oversampling on the training set before modeling.

### 6. Predicting hospital admission
Two classifiers were trained (post-SMOTE) to predict `IsHospitalAdmission` from visit/demographic features (diagnosis group, race, ethnicity, sex, smoking status, marital status, birth-year bin, month, year):

| Model | 5-fold CV Accuracy (train) | Test Accuracy | Admitted Recall | Admitted Precision |
|---|---|---|---|---|
| Logistic Regression | 0.685 ± 0.003 | 0.65 | 0.72 | 0.34 |
| **Random Forest** | **0.876 ± 0.032** | **0.82** | 0.64 | 0.57 |

Random Forest clearly outperforms Logistic Regression overall, though **recall on the "Admitted" class remains the harder problem** (57–60% F1) — expected given only ~21% of ED visits result in admission. A feature importance chart from the Random Forest highlights which variables (e.g., diagnosis group, birth year bin, month) drive the prediction most.

## Tech Stack

- **DuckDB** — fast SQL querying directly on CSVs / in-memory joins
- **pandas / numpy** — data wrangling
- **matplotlib / seaborn** — visualization
- **scikit-learn** — LabelEncoder, StandardScaler, PCA, train/test split, Logistic Regression, Random Forest, XGBoost (imported), classification metrics
- **imbalanced-learn (SMOTE)** — class imbalance correction
- **statsmodels** — Holt-Winters Exponential Smoothing for forecasting

## How to Reproduce

1. Place the raw CSVs (`departments.csv`, `encounters.csv`, `patients.csv`, `diagnosis.csv`, `providers.csv`, `social_determinants.csv`, `tigercensuscodes.csv`) in the working directory.
2. Run `Data_Cleaning.ipynb` first to explore and validate the joined `combined_data` table.
3. Run `Data_visualizing.ipynb` to reproduce the ED-focused visualizations, seasonal/time-series analysis, and admission-prediction models.

## Limitations & Next Steps

- Admission-prediction recall for the minority ("Admitted") class is still moderate — could improve with additional features (vitals, prior visit history, department type), hyperparameter tuning, or trying the imported-but-unused XGBoost model.
- `social_determinants.csv` and `tigercensuscodes.csv` were loaded but never joined in — incorporating them could add socioeconomic/geographic context to the admission model.
- Seasonal peaks (June, Oct/Nov) are observed but not yet statistically explained — worth cross-referencing with respiratory illness season, school calendars, or other external seasonal factors.
