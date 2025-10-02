# setting up environment
import streamlit as st
import pandas as pd
from src.data_processing import (
    get_month_end_date,
    get_month_filter_data,
    load_process_and_store
)
from src.visualization import (
    plot_monthly_hours_line,
    create_kpi_cards
)

import plotly.express as px

# load data
sales_pipeline, invoices, payments, time_reporting, start_date, end_date, monthly_totals = load_process_and_store()

# global date filter 
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

# hourly filters
hours_type = st.sidebar.selectbox(
    "Select Hours Type",
    options=['billable', 'non billable', 'total', 'utilization'],
    index=0,
    help="Select hour type"
)

hourly_trend_line = st.sidebar.selectbox(
    "Show regression line",
    options=["Yes", "No"],
    index=1,
    help="Show regression line for hours trend"
)
hourly_trend_line_bool = (hourly_trend_line == "Yes")

# KPI Cards section
st.subheader("Utilization") 
st.write(f"Showing {hours_type} hours from {selected_start_month} to {selected_end_month}")
st.markdown("---")

# Get KPI cards data
kpi_data = create_kpi_cards(
    invoices, 
    payments, 
    time_reporting, 
    hours_type, 
    start_date=start_date_str, 
    end_date=end_date_str,
    show_broker=False  # Hours don't have broker breakdown
)

if kpi_data:
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if kpi_data['main_avg']:
            hours_label = hours_type.title().replace('Non Billable', 'Non-Billable')
            if hours_type == 'utilization':
                st.metric(
                    label="Average Utilization", 
                    value=kpi_data['main_avg']['formatted_value']
                )
            else:
                st.metric(
                    label=f"Average {hours_label} Hours", 
                    value=kpi_data['main_avg']['formatted_value']
                )
    
    with col2:
        if kpi_data['main_sum']:
            if hours_type == 'utilization':
                # For utilization, sum doesn't make sense, so show empty placeholder
                st.metric(label="", value="")
            else:
                hours_label = hours_type.title().replace('Non Billable', 'Non-Billable')
                st.metric(
                    label=f"Total {hours_label} Hours", 
                    value=kpi_data['main_sum']['formatted_value']
                )
    
    with col3:
        st.metric(label="", value="") 

# hours trend chart
fig_hours = plot_monthly_hours_line(
    time_reporting,
    start_date=start_date_str,
    end_date=end_date_str,
    hours_type=hours_type,
    show_trend=hourly_trend_line_bool  
)
if fig_hours:
    st.plotly_chart(fig_hours, use_container_width=True)
else:
    st.warning("No hours data available for the selected filters.")




st.header("Utilization by Seniority (Proof of Concept)")
st.caption(" Demo only: Employees need to be  manually or evenly assigned to seniority tiers (K1–K4).")


# --- Manual mapping (replace with real IDs if you want) ---
manual_map = {
    # "1234": "K1 Junior",
    # "5678": "K2 Consultant",
    # "9012": "K3 Senior",
    # "3456": "K4 Partner"
}

df = time_reporting.copy()
df['seniority'] = df['employee_id'].map(manual_map)

# Auto-assign if mapping empty
if df['seniority'].isna().all():
    unique_ids = df['employee_id'].dropna().unique()
    tiers = ["K1 Junior", "K2 Consultant", "K3 Senior", "K4 Partner"]
    mapping_auto = {eid: tiers[i % 4] for i, eid in enumerate(unique_ids)}
    df['seniority'] = df['employee_id'].map(mapping_auto)

# --- Apply Date Filter ---
df['date'] = pd.to_datetime(df['date'])
df = df[(df['date'] >= start_date_str) & (df['date'] <= end_date_str)]

# --- Aggregate monthly hours ---
df['month'] = df['date'].dt.to_period('M')
monthly_hours = df.groupby(['month','seniority'])['total_hours'].sum().reset_index()

# --- KPI Cards (last 3 months) ---
if not monthly_hours.empty:
    latest_months = monthly_hours['month'].unique()[-3:]
    avg_hours = (
        monthly_hours[monthly_hours['month'].isin(latest_months)]
        .groupby('seniority')['total_hours'].mean()
        .round(0)
    )
    st.subheader("Average Hours (last 3 months)")
    cols = st.columns(len(avg_hours))
    for col, (tier, value) in zip(cols, avg_hours.items()):
        col.metric(label=tier, value=f"{int(value)} hrs")

# --- Chart ---
if not monthly_hours.empty:
    color_map = {
        "K1 Junior": "#773344",
        "K2 Consultant": "#E3B5A4",
        "K3 Senior": "#D44D5C",
        "K4 Partner": "#0B0014",
        "Unmapped": "#F5E9E2"
    }

    fig = px.bar(
        monthly_hours,
        x=monthly_hours['month'].astype(str),
        y="total_hours",
        color="seniority",
        barmode="stack",
        color_discrete_map=color_map,
        labels={"total_hours": "Total Hours", "month": "Month", "seniority": "Seniority Level"},
        title=f"Hours by Seniority ({selected_start_month} → {selected_end_month})"
    )

    fig.update_layout(
        height=500,
        width=900,
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(size=14)
    )

    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("No data available for selected period.")
