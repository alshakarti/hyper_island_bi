# Getting Started

## Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

The project defines its dependencies in `pyproject.toml` (and a `requirements.txt` for quick installs). Using `pip install -e .` keeps the `src/` package importable while you work in Streamlit.

## Create data repository


1. Request access to the shared Google Drive folder that contains the latest CSV export bundle (ask your project lead if you don’t already have it).
2. Download the ZIP archive (e.g., `hyper_island_bi_data.zip`) to your machine.
3. Extract the contents into the project’s `data/` directory so the individual `.csv` files sit directly under `data/`.

NOTE: The datasets contain sensitive information and must never be committed to Git. The `.gitignore` already excludes the `data/` folder—leave the files locally only.

## Run dashboard

```bash
streamlit run Key_metrics.py 
```

Run the command from the repository root (after activating the virtual environment and installing dependencies). Streamlit will open the dashboard in your browser; if it doesn’t launch automatically, copy the local URL from the terminal into your browser.

## Data modeling schema

The analytics layer in `src/data_processing.py` reshapes a handful of raw CSV extracts into the datasets used by the Streamlit pages. Each derived dataset traces back to the original file names listed below so it stays clear which upstream exports feed which part of the dashboard.

### Source → derived mapping

| Derived dataset (`process_data`) | Purpose | Source files | Key transformations |
| --- | --- | --- | --- |
| `sales_pipeline` | Enrich HubSpot deals with pipeline stage metadata for sales forecasting. | `fct__hubspot_deals__anonymized.csv`, `dim__hubspot_sales_pipeline_stages.csv` | Join `deal_stage` → `pipeline_stage_id`; drop archival metadata; coerce `create_date`/`close_date` to timezone-aware timestamps. |
| `invoices` | Clean customer invoice ledger used for revenue tracking. | `fct__fortnox_invoices__anonymized.csv` | Normalize year prefixes in date columns, fill missing `broker` as "Direct", drop accounting summary columns. |
| `payments` | Supplier payment facts for cost tracking. | `fct__fortnox_supplier_invoices.csv` | Convert `final_pay_date` to datetime; keep payment amount, category, and final settlement date. |
| `time_reporting` | Unified consultant time ledger across systems. | `fct__time_entries.csv`, `stg_qbis__activity_time.csv` | Split billable vs non-billable hours, normalize timestamps, append QBIS rows with `employee_id` alignment. |
| `monthly_totals` | Rolling consultant capacity valuation by month. | `dim__notion_roles__anonymized.csv`, `z.csv` | Filter rows with `startdate` and `hourly_rate`, back-fill open `enddate` with today, expand to month periods, aggregate `hourly_rate * 32`. |