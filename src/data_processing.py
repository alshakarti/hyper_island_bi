import pandas as pd
import os
from pathlib import Path
import io
import streamlit as st

def load_all_csv_files(data_dir='data', show_rows=5):
    # check if the directory exists
    if not os.path.exists(data_dir):
        print(f"Directory '{data_dir}' not found!")
        return {}, {}

    # store dataframes, file mappings and get csv files
    dataframes = {}
    file_mappings = {}
    csv_files = list(Path(data_dir).glob('*.csv'))
    csv_files.sort()

    print(f"Found {len(csv_files)} CSV files in '{data_dir}' directory.\n")

    # load each CSV file into a dataframe
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
def process_data(df1, df9, df7, df8, df10, df11, df4, df13):
    # store datasets
    datasets = {}

    # create a sales pipeline dataset
    sales_columns_order = [
        'deal_id', 'owner_id', 'deal_amount', 'weighted_deal_amount',
        'create_date', 'close_date', 'deal_name_anon',
        'pipeline_stage', 'pipeline_stage_order'
    ]
    sales_pipeline = (
        df9.merge(
            df1[['pipeline_stage_id', 'pipeline_stage', 'pipeline_stage_order']],
            left_on='deal_stage',
            right_on='pipeline_stage_id',
            how='left'
        )
        .drop(columns=['deal_stage', 'pipeline_stage_id', 'is_archived', 'last_modified_date'])
        .reindex(columns=sales_columns_order)
    )
    for col in ['create_date', 'close_date']:
        sales_pipeline[col] = pd.to_datetime(sales_pipeline[col])
    datasets['sales_pipeline'] = sales_pipeline

    # create invoices dataset
    date_columns = ['due_date', 'invoice_date', 'final_pay_date', 'accounting_year_date']
    for col in date_columns:
        df7[col] = df7[col].astype(str).apply(
            lambda x: '20' + x[2:] if len(x) >= 2 and x[:2] in
            ['21', '22', '23', '24', '25', '26', '27', '28', '29'] else x
        )
        df7[col] = pd.to_datetime(df7[col])
    df7['broker'] = df7['broker'].fillna('Direct')
    invoices = df7.drop(columns=['month_name', 'accounting_month', 'accounting_year'])
    datasets['invoices'] = invoices

    # create payments dataset
    df8['final_pay_date'] = pd.to_datetime(df8['final_pay_date'])
    payments = df8.drop(columns=['invoice_date', 'due_date'])
    datasets['payments'] = payments

    # create time reporting dataset
    records = []
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

    # ✅ restored correct loop from v1: use df11, not df12
    for _, row in df11.iterrows():
        date = pd.to_datetime(row['activity_date'])
        hours = row['minutes'] / 60.0
        is_billable = row['factor_value'] == 1.0
        records.append({
            'employee_id': row['employee_id'],
            'date': date,
            'billable_hours': hours if is_billable else 0,
            'non_billable_hours': hours if not is_billable else 0,
            'total_hours': hours,
        })

    time_reporting = pd.DataFrame(records)
    datasets['time_reporting'] = time_reporting

    # ✅ merge restored to df13 (z.csv) for role enddates — df15 removed
    consultants = df4.merge(df13, on='role_id', how='left')

    today = pd.Timestamp.today().normalize()
    consultants['startdate'] = pd.to_datetime(consultants['startdate'])
    consultants['enddate'] = pd.to_datetime(consultants['enddate']).fillna(today)
    consultants = consultants.dropna(subset=['startdate', 'hourly_rate'])
    consultants.loc[consultants['enddate'] < consultants['startdate'], 'enddate'] = consultants['startdate']

    records = []
    for row in consultants.itertuples():
        months = pd.period_range(row.startdate, row.enddate, freq='M')
        for month in months:
            records.append({
                'month': month.to_timestamp(),
                'role_id': row.role_id,
                'consultant_value': row.hourly_rate * 32
            })

    consultant_monthly = pd.DataFrame(records)
    monthly_totals = (
        consultant_monthly.groupby('month', as_index=False)['consultant_value']
        .sum()
        .rename(columns={'consultant_value': 'total_consultant_value'})
    )
    datasets['monthly_totals'] = monthly_totals

    # print info for each dataset
    print('-' * 100)
    print("DATASETS CREATED")
    print('-' * 100)
    for name, dataset in datasets.items():
        print(f"\n{name.upper()} DATASET:")
        print(f"Shape: {dataset.shape[0]} rows × {dataset.shape[1]} columns")
        print("\nColumn information:")
        buffer = io.StringIO()
        dataset.info(buf=buffer)
        print(buffer.getvalue())
        print('-' * 100)

    return sales_pipeline, invoices, payments, time_reporting, monthly_totals


# date utility functions
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
    first_month, last_month = common_dates[0], common_dates[-1]
    return first_month.start_time.date(), last_month.end_time.date()


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
        current += pd.DateOffset(months=1)
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
        default_start_str = default_start.strftime('%Y-%m-01')
        default_end_str = pd.Timestamp(default_end).strftime('%Y-%m-01')
        default_start_idx = all_month_values.index(default_start_str) if default_start_str in all_month_values else 0
        default_end_idx = all_month_values.index(default_end_str) if default_end_str in all_month_values else len(all_month_values) - 1
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

        # ✅ restored correct process_data call
        sales_pipeline, invoices, payments, time_reporting, monthly_totals = process_data(
            df1, df9, df7, df8, df10, df11, df4, df13
        )
        start_date, end_date = get_common_date_range(invoices, payments, time_reporting)

        st.session_state.sales_pipeline = sales_pipeline
        st.session_state.invoices = invoices
        st.session_state.payments = payments
        st.session_state.time_reporting = time_reporting
        try:
            st.session_state.employees = df6
        except NameError:
            st.session_state.employees = None
        st.session_state.start_date = start_date
        st.session_state.end_date = end_date
        st.session_state.monthly_totals = monthly_totals
        st.session_state.data_loaded = True

    return (
        st.session_state.sales_pipeline,
        st.session_state.invoices,
        st.session_state.payments,
        st.session_state.time_reporting,
        st.session_state.start_date,
        st.session_state.end_date,
        st.session_state.monthly_totals
    )