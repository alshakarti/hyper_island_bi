import pandas as pd
import os
from pathlib import Path
import io

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
def process_data(df1, df9, df7, df8, df10, df11):
    
    # store datasets 
    datasets = {}

    # create a sales pipeline dataset
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
    
    # create invoices dataset
    date_columns = ['due_date', 'invoice_date', 'final_pay_date', 'accounting_year_date'] 
    # check for dates with years starting with 21-29 (2100s-2900s) and convert to datetime
    for col in date_columns:
        df7[col] = df7[col].astype(str).apply(
            lambda x: '20' + x[2:] if len(x) >= 2 and x[:2] in ['21', '22', '23', '24', '25', '26', '27', '28', '29'] else x
        )
        df7[col] = pd.to_datetime(df7[col])
    df7['broker'] = df7['broker'].fillna('Direct')
    invoices = df7.drop(columns=['month_name', 'accounting_month', 'accounting_year'])
    datasets['invoices'] = invoices
    
    # create payments dataset
    date_columns = ['final_pay_date']
    for col in date_columns:
        df8[col] = pd.to_datetime(df8[col])
    payments = df8.drop(columns=['invoice_date', 'due_date']) 
    datasets['payments'] = payments

    # create time reporting dataset
    records = []

    # iterate though df10 and add hours to master df based on is_billable flag 
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

    # iterate though df11 and add hours to master df based on factor_value flag 
    for _, row in df11.iterrows():
        date = pd.to_datetime(row['activity_date'])
        hours = row['minutes'] / 60.0
        is_billable = row['factor_value'] == 1.0
        billable_hours = hours if is_billable else 0
        non_billable_hours = hours if not is_billable else 0
        total_hours = hours
        records.append({
            'date': date,
            'billable_hours': billable_hours,
            'non_billable_hours': non_billable_hours,
            'total_hours': total_hours,
        })

    time_reporting = pd.DataFrame(records)
    datasets['time_reporting'] = time_reporting
    
    # print info for each dataset
    print('-'*100)
    print("DATASETS CREATED")
    print('-'*100)
    for name, dataset in datasets.items():
        print(f"\n{name.upper()} DATASET:")
        print(f"Shape: {dataset.shape[0]} rows × {dataset.shape[1]} columns")
        print("\nColumn information:")
        buffer = io.StringIO()
        dataset.info(buf=buffer)
        info_output = buffer.getvalue()
        print(info_output)
        print('-'*100)
        
    # return all datasets 
    return sales_pipeline, invoices, payments, time_reporting

# get max and min dates from invoices and payments dataframes where we have data from all tables
def get_common_date_range(df, df2, df3):
    """
    Identify the common date range where all dataframes have data.
    
    Parameters:
    -----------
    df : pandas DataFrame
        The invoice dataframe
    df2 : pandas DataFrame
        The payments dataframe
    df3 : pandas DataFrame, optional
        The time reports dataframe
    
    Returns:
    --------
    tuple: (start_date, end_date) as datetime objects or strings in 'YYYY-MM-DD' format
           Returns (None, None) if there's no common period with data
    """
    # get date range for invoice data
    invoice_df = df.copy()
    invoice_df = invoice_df.dropna(subset=['final_pay_date'])
    if invoice_df.empty:
        return None, None
    
    invoice_dates = invoice_df['final_pay_date'].dt.to_period('M').unique()
    
    # get date range for payments data
    payment_df = df2.copy()
    payment_df = payment_df.dropna(subset=['final_pay_date'])
    if payment_df.empty:
        return None, None
    
    payment_dates = payment_df['final_pay_date'].dt.to_period('M').unique()
    
    # get common months between invoices and payments
    common_dates = set(invoice_dates).intersection(set(payment_dates))
    
    # if df3 (time reports) is provided, include it in the common dates
    if df3 is not None:
        time_df = df3.copy()
        time_df = time_df.dropna(subset=['date'])
        if time_df.empty:
            return None, None
        
        time_dates = pd.to_datetime(time_df['date']).dt.to_period('M').unique()
        common_dates = common_dates.intersection(set(time_dates))
    
    if not common_dates:
        return None, None
    
    # convert to list and sort
    common_dates = sorted(list(common_dates))
    
    # get first and last date in the common range
    first_month = common_dates[0]
    last_month = common_dates[-1]
    
    # format as YYYY-MM-DD (first day of first month, last day of last month)
    start_date = first_month.start_time
    end_date = last_month.end_time
    
    return start_date.date(), end_date.date()

# Get the first day of the month for start_date and last day of the month for end_date
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

# get list of all months between start and end date for the selectbox
def get_month_options(start_date, end_date):
    if start_date is None or end_date is None:
        return []
    
    # Convert to timestamps for calculation
    start = pd.Timestamp(start_date.year, start_date.month, 1)
    end = pd.Timestamp(end_date.year, end_date.month, 1)
    
    months = []
    current = start
    while current <= end:
        months.append(current)
        current = current + pd.DateOffset(months=1)
    
    return months