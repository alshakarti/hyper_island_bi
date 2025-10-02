# setting up environment
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
import plotly.graph_objects as go

# --- Load data ---
sales_pipeline, invoices, payments, time_reporting, monthly_hours, start_date, end_date = load_process_and_store()

# --- Sidebar filters ---
st.sidebar.subheader("Filter")

month_data = get_month_filter_data(start_date, end_date)
all_month_labels = month_data['month_labels']
all_month_values = month_data['month_values']
default_start_idx = month_data['default_start_idx']
default_end_idx = month_data['default_end_idx']

if st.sidebar.button("Reset to Default Date Range", help="Reset date filter to default range"):
    st.session_state.start_month = all_month_labels[default_start_idx]
    st.session_state.end_month = all_month_labels[default_end_idx]

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

start_idx = all_month_labels.index(selected_start_month)
end_idx = all_month_labels.index(selected_end_month)
if end_idx < start_idx:
    st.sidebar.error("End month cannot be before start month.")
    end_idx = start_idx
    selected_end_month = selected_start_month

start_date_str = all_month_values[start_idx]
end_date_obj = pd.to_datetime(all_month_values[end_idx])
end_date_obj = get_month_end_date(end_date_obj)
end_date_str = end_date_obj.strftime('%Y-%m-%d')

# --- Hours filters ---
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

# --- KPI Cards section ---
st.markdown("#### Utilization") 
st.caption(f"Showing {hours_type} hours from {selected_start_month} to {selected_end_month}")
st.markdown("---")

kpi_data = create_kpi_cards(
    invoices,
    payments,
    time_reporting,
    hours_type,
    start_date=start_date_str,
    end_date=end_date_str,
    show_broker=False
)

if kpi_data:
    col1, col2, col3 = st.columns(3)
    with col1:
        if kpi_data['main_avg']:
            hours_label = hours_type.title().replace('Non Billable', 'Non-Billable')
            if hours_type == 'utilization':
                st.metric("Average Utilization", kpi_data['main_avg']['formatted_value'])
            else:
                st.metric(f"Average {hours_label} Hours", kpi_data['main_avg']['formatted_value'])
    with col2:
        if kpi_data['main_sum']:
            if hours_type == 'utilization':
                st.metric(label="", value="")
            else:
                hours_label = hours_type.title().replace('Non Billable', 'Non-Billable')
                st.metric(f"Total {hours_label} Hours", kpi_data['main_sum']['formatted_value'])
    with col3:
        st.metric(label="", value="")

# --- New KPI Cards: Consultant Capacity ---
st.subheader("Consultant Capacity")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Active Consultants", monthly_hours['consultant_hours_total'].iloc[0] // 160)
with col2:
    st.metric("Capacity per Month", f"{monthly_hours['consultant_hours_total'].iloc[0]:,} hrs")
with col3:
    avg_utilization = monthly_hours['utilization_pct'].mean().round(1)
    st.metric("Avg Utilization %", f"{avg_utilization}%")

# --- Hours trend chart ---
fig_hours = None  # make sure the variable always exists

try:
    fig_hours = plot_monthly_hours_line(
        time_reporting,
        start_date=start_date_str,
        end_date=end_date_str,
        hours_type=hours_type,
        show_trend=hourly_trend_line_bool
    )
except Exception as e:
    st.error(f"Error creating hours chart: {e}")

if fig_hours is not None:
    st.plotly_chart(fig_hours, use_container_width=True)
else:
    st.warning("No hours data available for the selected filters.")


# --- New chart: Billable vs Capacity with Utilization % ---
fig_capacity = go.Figure()

fig_capacity.add_trace(go.Bar(
    x=monthly_hours['month'].astype(str),
    y=monthly_hours['billable_hours'],
    name="Billable Hours",
    marker_color="green"
))
fig_capacity.add_trace(go.Bar(
    x=monthly_hours['month'].astype(str),
    y=monthly_hours['consultant_hours_total'],
    name="Consultant Capacity",
    marker_color="lightgray"
))
fig_capacity.add_trace(go.Scatter(
    x=monthly_hours['month'].astype(str),
    y=monthly_hours['utilization_pct'],
    name="Utilization %",
    mode="lines+markers",
    yaxis="y2",
    line=dict(color="blue")
))
fig_capacity.update_layout(
    title="Billable vs Capacity (with Utilization %)",
    xaxis_title="Month",
    yaxis=dict(title="Hours"),
    yaxis2=dict(title="Utilization %", overlaying="y", side="right", range=[0,100]),
    barmode="group",
    height=500,
    width=900
)
st.plotly_chart(fig_capacity, use_container_width=True)

# --- Existing seniority section (unchanged) ---
st.header("Utilization by Seniority (Proof of Concept)")
st.caption(" Demo only: Employees need to be  manually or evenly assigned to seniority tiers (K1–K4).")

manual_map = {}
df = time_reporting.copy()
df['seniority'] = df['employee_id'].map(manual_map)

if df['seniority'].isna().all():
    unique_ids = df['employee_id'].dropna().unique()
    tiers = ["K1 Junior", "K2 Consultant", "K3 Senior", "K4 Partner"]
    mapping_auto = {eid: tiers[i % 4] for i, eid in enumerate(unique_ids)}
    df['seniority'] = df['employee_id'].map(mapping_auto)

df['date'] = pd.to_datetime(df['date'])
df = df[(df['date'] >= start_date_str) & (df['date'] <= end_date_str)]

df['month'] = df['date'].dt.to_period('M')
monthly_hours_seniority = df.groupby(['month','seniority'])['total_hours'].sum().reset_index()

if not monthly_hours_seniority.empty:
    latest_months = monthly_hours_seniority['month'].unique()[-3:]
    avg_hours = (
        monthly_hours_seniority[monthly_hours_seniority['month'].isin(latest_months)]
        .groupby('seniority')['total_hours'].mean()
        .round(0)
    )
    st.subheader("Average Hours (last 3 months)")
    cols = st.columns(len(avg_hours))
    for col, (tier, value) in zip(cols, avg_hours.items()):
        col.metric(label=tier, value=f"{int(value)} hrs")

if not monthly_hours_seniority.empty:
    color_map = {
        "K1 Junior": "#773344",
        "K2 Consultant": "#E3B5A4",
        "K3 Senior": "#D44D5C",
        "K4 Partner": "#0B0014",
        "Unmapped": "#F5E9E2"
    }
    fig = px.bar(
        monthly_hours_seniority,
        x=monthly_hours_seniority['month'].astype(str),
        y="total_hours",
        color="seniority",
        barmode="stack",
        color_discrete_map=color_map,
        labels={"total_hours": "Total Hours", "month": "Month", "seniority": "Seniority Level"},
        title=f"Hours by Seniority ({selected_start_month} → {selected_end_month})"
    )
    fig.update_layout(height=500, width=900, plot_bgcolor="white", paper_bgcolor="white", font=dict(size=14))
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("No data available for selected period.")
