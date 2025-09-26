import pandas as pd
import os
from pathlib import Path
import io
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

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