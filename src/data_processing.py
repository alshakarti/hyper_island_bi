import pandas as pd
import os
from pathlib import Path
import io
import streamlit as st

def load_all_csv_files(data_dir='data', show_rows=5):
    if not os.path.exists(data_dir):
        print(f"Directory '{data_dir}' not found!")
        return {}, {}
    
    dataframes = {}
    file_mappings = {}
    csv_files = list(Path(data_dir).glob('*.csv'))
    csv_files.sort() 
    
    print(f"Found {len(csv_files)} CSV files in '{data_dir}' directory.\n")
    
    for i, file_path in enumerate(csv_files, 1):
        df_name = f"df{i}"
        file_mappings[df_name] = str(file_path)
        
        print(f"Loaded {file_path} as {df_name}")
        
        df = pd.read_csv(file_path)
        dataframes[df_name] = df

        print(f"File: {file_path}")
        print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        print(f"\nFirst {show_rows} rows of {df_name}:")
        print(df.head(show_rows))
        print("-" * 80 + "\n")
    
    return dataframes, file_mappings

# combine and process raw data into datasets for analysis 
def process_data(df1, df9, df7, df8, df10, df11):
    datasets = {}

    # --- Sales pipeline ---
    sales_columns_order = [
        'deal_id', 'owner_id', 'deal_amount', 'weighted_deal_amount', 'create_date', 'close_date',
        'deal_name_anon', 'pipeline_stage', 'pipeline_stage_order'
    ]
    sales_pipeline = (df9.merge(
        df1[['pipeline_stage_id', 'pipeline_stage','pipeline_stage_order']],
        left_on='deal_stage',
        right_on='pipeline_stage_id',
        how='left'
    )
    .drop(columns=['deal_stage', 'pipeline_stage_id', 'is_archived', 'last_modified_date'])
    .reindex(columns=sales_columns_order)
    )
    date_columns = ['create_date', 'close_date']
    for col in date_columns:
        sales_pipeline[col] = pd.to_datetime(sales_pipeline[col])
    datasets['sales_pipeline'] = sales_pipeline
    
    # --- Invoices ---
    date_columns = ['due_date', 'invoice_date', 'final_pay_date', 'accounting_year_date'] 
    for col in date_columns:
        df7[col] = df7[col].astype(str).apply(
            lambda x: '20' + x[2:] if len(x) >= 2 and x[:2] in ['21','22','23','24','25','26','27','28','29'] else x
        )
        df7[col] = pd.to_datetime(df7[col])
    df7['broker'] = df7['broker'].fillna('Direct')
    invoices = df7.drop(columns=['month_name', 'accounting_month', 'accounting_year'])
    datasets['invoices'] = invoices
    
    # --- Payments ---
    for col in ['final_pay_date']:
        df8[col] = pd.to_datetime(df8[col])
    payments = df8.drop(columns=['invoice_date', 'due_date']) 
    datasets['payments'] = payments

    # --- Time reporting (combine df10 + df11) ---
    records = []

    # df10 = clockify
    for _, row in df10.iterrows():
        date = pd.to_datetime(row['dt'])
        is_billable = bool(row['billable'])
        billable_hours = row['hours'] if is_billable else 0
        non_billable_hours = row['hours'] if not is_billable else 0
        total_hours = row['hours']
        records.append({
            'date': date,
            'billable_hours': billable_hours,
            'non_billable_hours': non_billable_hours,
            'total_hours': total_hours,
        })

    # df11 = qbis
    for _, row in df11.iterrows():
        date = pd.to_datetime(row['activity_date'])
        hours = row['minutes'] / 60.0
        is_billable = row['factor_value'] == 1.0
        records.append({
            'employee_id': row.get('employee_id', None),
            'date': date,
            'billable_hours': hours if is_billable else 0,
            'non_billable_hours': hours if not is_billable else 0,
            'total_hours': hours,
        })

    time_reporting = pd.DataFrame(records)
    datasets['time_reporting'] = time_reporting
    
    # --- Consultant capacity (from HR) ---
    hr_path = Path("data/dim__notion_hr__anonymized.csv")
    if hr_path.exists():
        df3 = pd.read_csv(hr_path, parse_dates=["startdate", "enddate"])
        active_consultants = df3[df3['active'] == "Yes"]['consultant_id'].nunique()
    else:
        active_consultants = 0

    # --- Monthly aggregation with consultant capacity ---
    time_reporting['month'] = pd.to_datetime(time_reporting['date']).dt.to_period('M')
    monthly_hours = (
        time_reporting.groupby('month')[['billable_hours','non_billable_hours','total_hours']]
        .sum()
        .reset_index()
    )
    monthly_hours['consultant_hours_total'] = active_consultants * 160
    monthly_hours['utilization_pct'] = (
        monthly_hours['billable_hours'] / monthly_hours['consultant_hours_total'] * 100
    ).round(1)
    datasets['monthly_hours'] = monthly_hours
    
    # --- Info printout ---
    print('-'*100)
    print("DATASETS CREATED")
    print('-'*100)
    for name, dataset in datasets.items():
        print(f"\n{name.upper()} DATASET:")
        print(f"Shape: {dataset.shape[0]} rows × {dataset.shape[1]} columns")
        buffer = io.StringIO()
        dataset.info(buf=buffer)
        print(buffer.getvalue())
        print('-'*100)
        
    return sales_pipeline, invoices, payments, time_reporting, monthly_hours

# --- Helper functions for date ranges etc. (unchanged) ---
def get_common_date_range(df, df2, df3):
    invoice_df = df.copy().dropna(subset=['final_pay_date'])
    if invoice_df.empty:
        return None, None
    invoice_dates = invoice_df['final_pay_date'].dt.to_period('M').unique()
    
    payment_df = df2.copy().dropna(subset=['final_pay_date'])
    if payment_df.empty:
        return None, None
    payment_dates = payment_df['final_pay_date'].dt.to_period('M').unique()
    
    common_dates = set(invoice_dates).intersection(set(payment_dates))
    if df3 is not None:
        time_df = df3.copy().dropna(subset=['date'])
        if time_df.empty:
            return None, None
        time_dates = pd.to_datetime(time_df['date']).dt.to_period('M').unique()
        common_dates = common_dates.intersection(set(time_dates))
    
    if not common_dates:
        return None, None
    
    common_dates = sorted(list(common_dates))
    first_month = common_dates[0]
    last_month = common_dates[-1]
    start_date = first_month.start_time
    end_date = last_month.end_time
    return start_date.date(), end_date.date()

def get_month_start_date(date):
    if date is None:
        return None
    if isinstance(date, str):
        date = pd.to_datetime(date)
    return pd.Timestamp(date.year, date.month, 1).date()

def get_month_end_date(date):
    if date is None:
        return None
    if isinstance(date, str):
        date = pd.to_datetime(date)
    next_month = date.replace(day=28) + pd.DateOffset(days=4)
    return (next_month - pd.DateOffset(days=next_month.day)).date()

def get_month_options(start_date, end_date):
    if start_date is None or end_date is None:
        return []
    start = pd.Timestamp(start_date.year, start_date.month, 1)
    end = pd.Timestamp(end_date.year, end_date.month, 1)
    months = []
    current = start
    while current <= end:
        months.append(current)
        current = current + pd.DateOffset(months=1)
    return months

def get_month_filter_data(start_date, end_date, months_back=12):
    default_end = get_month_end_date(end_date)
    if default_end:
        default_start_date = pd.Timestamp(default_end) - pd.DateOffset(months=months_back-1)
        default_start = get_month_start_date(default_start_date)
    else:
        default_start = get_month_start_date(start_date)
    all_month_options = get_month_options(get_month_start_date(start_date), get_month_end_date(end_date))
    all_month_labels = [d.strftime('%B %Y') for d in all_month_options]
    all_month_values = [d.strftime('%Y-%m-01') for d in all_month_options]
    if default_start and default_end and all_month_options:
        try:
            default_start_idx = all_month_values.index(default_start.strftime('%Y-%m-01'))
        except ValueError:
            default_start_idx = 0
        try:
            default_end_idx = all_month_values.index(pd.Timestamp(default_end).strftime('%Y-%m-01'))
        except ValueError:
            default_end_idx = len(all_month_values) - 1
    else:
        default_start_idx = 0
        default_end_idx = len(all_month_options) - 1 if all_month_options else 0
    return {
        'month_options': all_month_options,
        'month_labels': all_month_labels,
        'month_values': all_month_values,
        'default_start_idx': default_start_idx,
        'default_end_idx': default_end_idx,
        'default_start': default_start,
        'default_end': default_end
    }

def load_process_and_store():
    if 'data_loaded' not in st.session_state or not st.session_state.data_loaded:
        dataframes, file_mappings = load_all_csv_files()
        for name, df in dataframes.items():
            globals()[name] = df
        print(f"df1 is from file: {file_mappings['df1']}")
        sales_pipeline, invoices, payments, time_reporting, monthly_hours = process_data(df1, df9, df7, df8, df10, df11)
        start_date, end_date = get_common_date_range(invoices, payments, time_reporting)
        st.session_state.sales_pipeline = sales_pipeline
        st.session_state.invoices = invoices
        st.session_state.payments = payments
        st.session_state.time_reporting = time_reporting
        st.session_state.monthly_hours = monthly_hours
        st.session_state.start_date = start_date
        st.session_state.end_date = end_date
        st.session_state.data_loaded = True
    
    return (
        st.session_state.sales_pipeline,
        st.session_state.invoices,
        st.session_state.payments,
        st.session_state.time_reporting,
        st.session_state.monthly_hours,
        st.session_state.start_date,
        st.session_state.end_date
    )
