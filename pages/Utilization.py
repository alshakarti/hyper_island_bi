import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from src.data_processing import load_process_and_store, get_month_filter_data, get_month_end_date

# Load datasets
_, _, _, time_reporting, start_date, end_date = load_process_and_store()

st.header("Utilization by Seniority (Proof of Concept)")
st.caption("⚠️ Demo only: Employees are manually or evenly assigned to seniority tiers (K1–K4).")

# --- Sidebar Date Filter ---
st.sidebar.header("Date filter")

month_data = get_month_filter_data(start_date, end_date)
all_month_labels = month_data['month_labels']
all_month_values = month_data['month_values']
default_start_idx = month_data['default_start_idx']
default_end_idx = month_data['default_end_idx']

selected_start_month = st.sidebar.selectbox("Start Month", options=all_month_labels, index=default_start_idx, key="util_start_month")
selected_end_month = st.sidebar.selectbox("End Month", options=all_month_labels, index=default_end_idx, key="util_end_month")

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
