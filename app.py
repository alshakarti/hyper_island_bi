# setting up environment
import streamlit as st
import pandas as pd
from src.data_processing import (
    load_all_csv_files,
    process_data,
    get_common_date_range,
    get_month_start_date,
    get_month_end_date,
    get_month_options
)
from src.visualisation import (
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

    sales_pipeline, invoices, payments, time_reporting = process_data(df1, df9, df7, df8, df10, df11)
    start_date, end_date = get_common_date_range(invoices, payments, time_reporting)
    
    return sales_pipeline, invoices, payments, time_reporting, start_date, end_date

sales_pipeline, invoices, payments, time_reporting, start_date, end_date = load_process_and_cache()

# global date filter 
st.sidebar.header("Date Filters")

# calculate most recent 12 months as default
default_end = get_month_end_date(end_date)
if default_end:
    default_start_date = pd.Timestamp(default_end) - pd.DateOffset(months=11)
    default_start = get_month_start_date(default_start_date)
else:
    default_start = get_month_start_date(start_date)

# get all available months in the data
all_month_options = get_month_options(get_month_start_date(start_date), get_month_end_date(end_date))
all_month_labels = [d.strftime('%B %Y') for d in all_month_options]
all_month_values = [d.strftime('%Y-%m-01') for d in all_month_options]

# find the default month indices (for most recent 12 months)
if default_start and default_end and all_month_options:
    default_start_str = default_start.strftime('%Y-%m-01')
    default_end_str = pd.Timestamp(default_end).strftime('%Y-%m-01')
    
    try:
        default_start_idx = all_month_values.index(default_start_str)
    except ValueError:
        default_start_idx = 0
        
    try:
        default_end_idx = all_month_values.index(default_end_str)
    except ValueError:
        default_end_idx = len(all_month_values) - 1
else:
    default_start_idx = 0
    default_end_idx = len(all_month_options) - 1 if all_month_options else 0

# create select boxes for start and end month
selected_start_month = st.sidebar.selectbox(
    "Start Month",
    options=all_month_labels,
    index=default_start_idx,
    key="start_month",
    help="Filter data from this month"
)
selected_end_month = st.sidebar.selectbox(
    "End Month",
    options=all_month_labels,
    index=default_end_idx,
    key="end_month", 
    help="Filter data up to this month"
)
# get indices of selected months
start_idx = all_month_labels.index(selected_start_month)
end_idx = all_month_labels.index(selected_end_month)

# make sure end month is not before start month
if end_idx < start_idx:
    st.sidebar.error("End month cannot be before start month.")
    end_idx = start_idx
    selected_end_month = selected_start_month

# convert selections back to dates
start_date_str = all_month_values[start_idx]
end_date_obj = pd.to_datetime(all_month_values[end_idx])
end_date_obj = get_month_end_date(end_date_obj)
end_date_str = end_date_obj.strftime('%Y-%m-%d')

# table filters
st.sidebar.header("Table Filters")

trend_color = st.sidebar.selectbox(
    "Highlight MoM Trend",
    options=["Yes", "No"],
    index=0,
    help="Show green/red color for positive/negative trends"
)
trend_color_bool = (trend_color == "Yes")

# graf filters
st.sidebar.header("Graph Filters") 
    
amount_type = st.sidebar.selectbox(
    "Select Amount Type",
    options=['net', 'payments', 'revenue'],
    index=0
)
show_broker = st.sidebar.selectbox(
    "Amount breakdown by broker",
    options=['Yes', 'No'],
    index=0,
    help="Show separate bars for each broker type"
)
show_broker_bool = (show_broker == 'Yes')

# dashboard items 
st.header("Key Metric")
st.markdown("---")

st.subheader("Financial")
st.caption(f"Showing data from {selected_start_month} to {selected_end_month}")
financial_data = key_metrics_monthly(invoices, payments, time_reporting, start_date=start_date_str, end_date=end_date_str)

# Define all potential rows we want to display
desired_rows = [
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

# Filter to include only the rows that actually exist in the data
filtered_financial_data = financial_data.loc[
    [row for row in desired_rows if row in financial_data.index]
]

# ...existing code...
styled_financial_data = filtered_financial_data.style.apply(
    lambda row: highlight_revenue_trend(row, color=trend_color_bool), axis=1
)
st.dataframe(
    styled_financial_data,
    use_container_width=True,
    hide_index=False
)

# Connect date filters to the chart
st.subheader("Monthly Trends")
st.caption(f"Showing data from {selected_start_month} to {selected_end_month}")
fig = plot_invoice_amounts(
    invoices,
    payments,
    start_date=start_date_str,  
    end_date=end_date_str,     
    amount_type=amount_type,
    hue=show_broker_bool
)

if fig:
    st.plotly_chart(fig, use_container_width=True)