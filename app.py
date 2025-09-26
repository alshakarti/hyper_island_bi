# setting up environment
import streamlit as st
import pandas as pd
import os
from pathlib import Path
import io
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

from src.data_processing import (
    load_all_csv_files,
    process_data
)
from src.visualisation import (
    sales_funnel_viz,
    sales_funnel_viz2,
    plot_invoice_amounts,
    key_metrics_monthly,
    highlight_revenue_trend
)
# customizing the page
st.set_page_config(
    page_title="Key Metrics Dashboard",
    page_icon=":guitar:",
    layout="wide", 
    initial_sidebar_state="expanded"
)

# loading the data
@st.cache_data   
def load_process_and_cache(): 

    dataframes, file_mappings = load_all_csv_files()
    for name, df in dataframes.items():
        globals()[name] = df
    print(f"df1 is from file: {file_mappings['df1']}")

    # creating reporting tables  
    sales_pipeline, invoices, payments, time_reporting = process_data(df1, df9, df7, df8, df10, df11)
    
    return sales_pipeline, invoices, payments, time_reporting

sales_pipeline, invoices, payments, time_reporting = load_process_and_cache()

# dashboard items 
st.header("Key Metric")
st.markdown("---")

st.subheader("Financial")
financial_data = key_metrics_monthly(invoices, payments, time_reporting, start_date=None, end_date=None)
# Filter to include only the specified rows
filtered_financial_data = financial_data.loc[
    [
        #'Total Invoice Amount',
        'Total Net Amount',
        'Payments',
        'Revenue',
        'Broker % of Total Net',
        'Direct % of Total Net', 
        'Partner % of Total Net',
        'Billable Hours',
        'Non-Billable Hours',
        'Total Hours',
        'Utilization Percentage'
        
    ]
]

styled_financial_data = filtered_financial_data.style.apply(highlight_revenue_trend, axis=1)
st.dataframe(
    styled_financial_data,
    use_container_width=True,
    hide_index=False
)

# test chart - filters have to be moved to sidebar and linked to the chart function though variable/argument 
st.subheader("Monthly Trends")

amount_type = st.selectbox(
    "Select Amount Type",
    options=['net', 'total', 'payments', 'revenue'],
    index=0
)

fig = plot_invoice_amounts(
    invoices,
    payments,
    amount_type='net',
    hue=True
)

if fig:
    st.plotly_chart(fig, use_container_width=True)
    
    
# revenue and net invoice trends: color text based on positive or negative trend
# utilization goal: 80% of hours should be billed for