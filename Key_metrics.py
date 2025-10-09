# setting up environment
import streamlit as st
import pandas as pd
from src.data_processing import (
    get_month_end_date,
    get_month_filter_data,
    load_process_and_store
)
from src.visualization import (
    key_metrics_monthly,
    highlight_revenue_trend,
    plot_net_amount_mom_growth,
)
# customizing the page
st.set_page_config(
    page_title="Key Metrics Dashboard",
    page_icon=":guitar:",
    layout="wide", 
    initial_sidebar_state="expanded"
)

sales_pipeline, invoices, payments, time_reporting, start_date, end_date, monthly_totals = load_process_and_store()

# global filter 
st.sidebar.subheader("Filter")

# get datetime for filter
month_data = get_month_filter_data(start_date, end_date)
all_month_labels = month_data['month_labels']
all_month_values = month_data['month_values']
default_start_idx = month_data['default_start_idx']
default_end_idx = month_data['default_end_idx']

# reset button - still work in progress 
if st.sidebar.button("Reset to Default Date Range", help="Reset date filter to default range"):
    st.session_state.start_month = all_month_labels[default_start_idx]
    st.session_state.end_month = all_month_labels[default_end_idx]

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

trend_color = st.sidebar.selectbox(
    "Above / below target highlight",
    options=["Yes", "No"],
    index=0,
    help="Show green color if metric is at or above target and red color if metric is below target"
)
trend_color_bool = (trend_color == "Yes")

table_rows = st.sidebar.selectbox(
    "Show all monthly KPIs",
    options=["Yes", "No"],
    index=1,
    help="Show all available KPIs"
)
table_rows_bool = (table_rows == "Yes")

# dashboard table 
st.subheader("Key Metrics") 
st.write(f"Showing KPIs from {selected_start_month} to {selected_end_month}")
st.markdown("---")

financial_data = key_metrics_monthly(invoices, payments, time_reporting, start_date=start_date_str, end_date=end_date_str)

if table_rows_bool:
    # Define all potential rows we want to display
    desired_rows = [
        'Net Amount',
        'Payments',
        'Revenue',
        'Broker % of Net',
        'Direct % of Net', 
        'Partner % of Net',
        'Billable Hours',
        'Non-Billable Hours',
        'Total Hours',
        'Utilization Percentage'
    ]
else:
    desired_rows = [
    'Net Amount',
    'Revenue',
    'Direct % of Net', 
    'Utilization Percentage'
]

# filter to include only the rows that actually exist in the data
filtered_financial_data = financial_data.loc[
    [row for row in desired_rows if row in financial_data.index]
]

styled_financial_data = filtered_financial_data.style.apply(
    lambda row: highlight_revenue_trend(row, pivot_df=filtered_financial_data, color=trend_color_bool),
    axis=1
)

st.dataframe(
    styled_financial_data,
    use_container_width=True,
    hide_index=False
)

# net Amount MoM growth
growth_fig = plot_net_amount_mom_growth(
    invoices,
    payments,
    time_reporting,
    start_date=start_date_str,
    end_date=end_date_str,
    show_required_line=trend_color_bool,
    first_month_as_zero=True 
)

#st.subheader("MoM growth rate") 
if growth_fig:
    st.plotly_chart(growth_fig, use_container_width=True)
