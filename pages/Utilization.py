# setting up environment
import streamlit as st
import pandas as pd
import numpy as np
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

# ------------------ deterministic employee-id normalization ------------------
def _normalize_emp_id(val):
    """Normalize employee id to a stable string or None."""
    if pd.isna(val):
        return None
    s = str(val).strip()
    if s == "" or s.lower() in {"nan", "none"}:
        return None
    try:
        f = float(s)
        if f.is_integer():
            return str(int(f))
    except Exception:
        pass
    return s

# create a normalized column for deterministic counting (used across the page)
time_reporting = time_reporting.copy()
time_reporting['employee_id_norm'] = time_reporting['employee_id'].apply(_normalize_emp_id)
# ------------------------------------------------------------------------------

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
# normalize filter bounds to Timestamps to avoid dtype comparison issues
start_dt = pd.to_datetime(start_date_str)
end_dt = pd.to_datetime(end_date_str)

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

# Toggle: use minimum 10 consultants for capacity calculation
use_min_10 = st.sidebar.selectbox(
    "Capacity calc: use minimum 10 consultants?",
    options=["Yes", "No"],
    index=0,
    help="If Yes, capacity will use at least 10 consultants when calculating capacity bars"
)
use_min_10_bool = (use_min_10 == "Yes")

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
# use normalized timestamps for comparisons
df = df[(df['date'] >= start_dt) & (df['date'] <= end_dt)]

# --- Aggregate monthly hours ---
df['month'] = df['date'].dt.to_period('M')

# Ensure billable/non_billable columns exist
if 'billable_hours' not in df.columns:
    df['billable_hours'] = 0.0
if 'non_billable_hours' not in df.columns:
    df['non_billable_hours'] = 0.0

# Aggregate sums and count unique consultants per month/seniority
monthly_agg = df.groupby(['month', 'seniority']).agg(
    billable_hours=('billable_hours', 'sum'),
    non_billable_hours=('non_billable_hours', 'sum'),
    total_hours=('total_hours', 'sum'),
    consultant_count=('employee_id_norm', lambda x: x.dropna().nunique())
).reset_index()

# Utilization percentage (handle divide-by-zero)
monthly_agg['utilization_pct'] = (
    (monthly_agg['billable_hours'] / monthly_agg['total_hours'])
    .replace([np.inf, -np.inf], 0)
    .fillna(0) * 100
)

# Capacity per consultant: assume 160 working hours per month
monthly_agg['consultant_hours_total'] = monthly_agg['consultant_count'] * 160

monthly_hours = monthly_agg

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

# --- Simplified KPI Cards for Utilization ---
st.subheader("Key Utilization KPIs")

# Build company-level monthly summary if not already present
company_monthly = None
if 'monthly_summary' in globals() or 'monthly_summary' in locals():
    company_monthly = globals().get('monthly_summary', locals().get('monthly_summary'))
    if company_monthly is not None:
        company_monthly = company_monthly.copy()
else:
    # aggregate monthly_hours (seniority-level) into company-level monthly summary
    if 'monthly_hours' in locals() and not monthly_hours.empty:
        tmp = monthly_hours.groupby('month').agg(
            billable_hours=('billable_hours', 'sum') if 'billable_hours' in monthly_hours.columns else ('total_hours', 'sum'),
            non_billable_hours=('non_billable_hours', 'sum') if 'non_billable_hours' in monthly_hours.columns else ('total_hours', 'sum'),
            total_hours=('total_hours', 'sum')
        ).reset_index()
        # Try to recover consultant counts if present
        if 'consultant_count' in monthly_hours.columns:
            tmp['consultants_active'] = monthly_hours.groupby('month')['consultant_count'].sum().values
        company_monthly = tmp

if company_monthly is None or company_monthly.empty:
    st.warning("No utilization data for selected period.")
else:
    # prepare month timestamps and filter by selected range
    company_monthly['month_ts'] = company_monthly['month'].dt.to_timestamp()
    filtered = company_monthly[(company_monthly['month_ts'] >= start_dt) & (company_monthly['month_ts'] <= end_dt)]

    if filtered.empty:
        st.warning("No utilization data for selected period.")
    else:
        total_hours = filtered['total_hours'].sum()
        billable_hours = filtered.get('billable_hours', filtered['total_hours']).sum()

        # utilization percent: prefer precomputed, else compute
        if 'utilization_pct' in filtered.columns:
            avg_utilization = filtered['utilization_pct'].mean().round(1)
        else:
            avg_utilization = round((filtered['billable_hours'] / filtered['total_hours']).replace([np.inf, -np.inf], 0).fillna(0).mean() * 100, 1)

        # capacity: look for capacity fields, fallback to consultants_active * 160
        if 'consultant_hours_total' in filtered.columns:
            capacity = filtered['consultant_hours_total'].mean().round(0)
        elif 'capacity_with_min10' in filtered.columns:
            capacity = filtered['capacity_with_min10'].mean().round(0)
        elif 'capacity' in filtered.columns:
            capacity = filtered['capacity'].mean().round(0)
        elif 'consultants_active' in filtered.columns:
            capacity = (filtered['consultants_active'].mean() * 160).round(0)
        else:
            capacity = 0

        # Prefer using the actual consultants_active mean (rounded) for the "Total Consultants" KPI
        if 'consultants_active' in filtered.columns and filtered['consultants_active'].notna().any():
            total_consultants = int(round(filtered['consultants_active'].mean()))
        else:
            # fallback: derive from capacity; use rounding to avoid floor undercount
            total_consultants = int(round(capacity / 160)) if capacity else 0

        # compute previous-period sums to show percentage deltas (if available)
        try:
            months_all = list(company_monthly['month'])
            start_month = filtered['month'].iloc[0]
            n_months = len(filtered)
            start_idx = months_all.index(start_month)
            prev_slice = company_monthly.iloc[max(0, start_idx - n_months): start_idx]
        except Exception:
            prev_slice = company_monthly.iloc[0:0]

        def _pct_change(curr, prev):
            try:
                prev = float(prev)
                if prev == 0:
                    return None
                return round((curr - prev) / prev * 100, 1)
            except Exception:
                return None

        prev_total = prev_slice['total_hours'].sum() if not prev_slice.empty else 0
        prev_billable = prev_slice.get('billable_hours', prev_slice.get('total_hours', pd.Series(dtype=float))).sum() if not prev_slice.empty else 0
        if 'utilization_pct' in prev_slice.columns and not prev_slice.empty:
            prev_util = prev_slice['utilization_pct'].mean()
        else:
            # compute previous utilization from billable/total if possible
            prev_util = (prev_slice.get('billable_hours', pd.Series(dtype=float)).sum() / prev_slice['total_hours'].sum() * 100) if (not prev_slice.empty and prev_slice['total_hours'].sum() > 0) else None
        prev_capacity = prev_slice['consultant_hours_total'].mean() if ('consultant_hours_total' in prev_slice.columns and not prev_slice.empty) else (prev_slice['capacity_with_min10'].mean() if ('capacity_with_min10' in prev_slice.columns and not prev_slice.empty) else (prev_slice['capacity'].mean() if ('capacity' in prev_slice.columns and not prev_slice.empty) else 0))
        prev_consultants = int(prev_capacity // 160) if prev_capacity else (int(prev_slice['consultants_active'].mean()) if ('consultants_active' in prev_slice.columns and not prev_slice.empty) else 0)

        total_delta = _pct_change(total_hours, prev_total)
        billable_delta = _pct_change(billable_hours, prev_billable)
        util_delta = _pct_change(avg_utilization, prev_util) if prev_util is not None else None
        capacity_delta = _pct_change(capacity, prev_capacity) if prev_capacity else None
        consultants_delta = _pct_change(total_consultants, prev_consultants) if prev_consultants else None

        # render KPI metrics with numeric deltas so Streamlit displays the colored delta badge
        col1, col2, col3, col4, col5 = st.columns(5)
        period_label = f"vs previous {n_months} month(s)"

        with col1:
            # total_hours: shown as integer, delta as percent (numeric) for coloring
            col1.metric("Total Hours", f"{int(total_hours):,}", total_delta if total_delta is not None else None, help=f"Sum of hours in selected range ({period_label})")
        with col2:
            col2.metric("Billable Hours", f"{int(billable_hours):,}", billable_delta if billable_delta is not None else None, help=f"Sum of billable hours in selected range ({period_label})")
        with col3:
            # utilization: value as string with % sign, delta as percent points (numeric)
            col3.metric("Utilization %", f"{avg_utilization}%", util_delta if util_delta is not None else None, help=f"Average billable% across months ({period_label})")
        with col4:
            col4.metric("Capacity (hrs)", f"{int(capacity):,}", capacity_delta if capacity_delta is not None else None, help=f"Average capacity in hrs ({period_label})")
        with col5:
            col5.metric("Total Consultants", f"{total_consultants}", consultants_delta if consultants_delta is not None else None, help=f"Estimated total consultants (capacity/160) ({period_label})")


# --- Company-level monthly summaries & charts (Total hours, Work hours structure, Utilization) ---
# Build a monthly summary across all seniorities for the selected date range
if not df.empty:
    # monthly_summary grouped by month (Period)
    # compute active consultants via employees dim when available
    employees_dim = getattr(st.session_state, 'employees', None)

    # helper to normalize ids: convert floats like 2.0 -> '2', ints -> '2', else string
    def _normalize_id(v):
        if pd.isna(v):
            return None
        try:
            f = float(v)
            if f.is_integer():
                return str(int(f))
            return str(v)
        except Exception:
            return str(v)

    # produce a normalized id column for robust comparisons
    df['employee_id_norm'] = df['employee_id'].apply(_normalize_id)

    if employees_dim is not None and 'employee_id' in employees_dim.columns:
        # normalize employee dim ids to string ints as well
        emp_norm_ids = set(employees_dim['employee_id'].dropna().apply(lambda x: str(int(x)) if float(x).is_integer() else str(x)).astype(str))

        # group and capture the set of unique normalized ids per month
        monthly_summary = df.groupby('month').agg(
            billable_hours=('billable_hours', 'sum'),
            non_billable_hours=('non_billable_hours', 'sum'),
            total_hours=('total_hours', 'sum'),
            consultants_set=('employee_id_norm', lambda x: set([v for v in x.dropna().unique()]))
        ).reset_index()

        # count only those consultants that exist in the employee dim
        monthly_summary['consultants_active'] = monthly_summary['consultants_set'].apply(lambda s: len([e for e in s if e in emp_norm_ids]))
        monthly_summary.drop(columns=['consultants_set'], inplace=True)
    else:
        # fallback: count unique normalized employee ids from time entries
        monthly_summary = df.groupby('month').agg(
            billable_hours=('billable_hours', 'sum'),
            non_billable_hours=('non_billable_hours', 'sum'),
            total_hours=('total_hours', 'sum'),
            consultants_active=('employee_id_norm', lambda x: len([v for v in x.dropna().unique()]))
        ).reset_index()

    # capacity = consultants_active * 160 (approx working hours per month)
    monthly_summary['capacity'] = monthly_summary['consultants_active'] * 160
    # ensure capacity at least reflects 10 consultants if user expects more (but keep actual count separately)
    if use_min_10_bool:
        monthly_summary['capacity_with_min10'] = monthly_summary['capacity'].where(monthly_summary['consultants_active'] >= 10, 10 * 160)
    else:
        monthly_summary['capacity_with_min10'] = monthly_summary['capacity']

    # utilization percentage
    monthly_summary['utilization_pct'] = (
        (monthly_summary['billable_hours'] / monthly_summary['total_hours'])
        .replace([np.inf, -np.inf], 0)
        .fillna(0) * 100
    )

    # format month labels
    monthly_summary['month_str'] = monthly_summary['month'].dt.to_timestamp().dt.strftime('%b')

    # Chart layout: three charts in a row
    c1, c2, c3 = st.columns(3)

    # Chart 1: Total hours vs Capacity (show capacity as light grey)
    import plotly.graph_objects as go

    with c1:
        fig1 = go.Figure()
        # actual total hours
        fig1.add_trace(go.Bar(
            x=monthly_summary['month_str'],
            y=monthly_summary['total_hours'],
            name='Work hours',
            marker_color='#28C19E'
        ))
        # capacity (stacked above as remaining capacity)
        remaining = (monthly_summary['capacity_with_min10'] - monthly_summary['total_hours']).clip(lower=0)
        fig1.add_trace(go.Bar(
            x=monthly_summary['month_str'],
            y=remaining,
            name='Total hours (capacity)',
            marker_color='#EAEAEA'
        ))
        fig1.update_layout(barmode='stack', title='Total hours structure', height=420)
        st.plotly_chart(fig1, use_container_width=True)

    # Chart 2: Work hours structure (Billable vs Non-billable)
    with c2:
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=monthly_summary['month_str'],
            y=monthly_summary['billable_hours'],
            name='Billable hours',
            marker_color='#6B46FF'
        ))
        fig2.add_trace(go.Bar(
            x=monthly_summary['month_str'],
            y=monthly_summary['non_billable_hours'],
            name='Non-Billable hours',
            marker_color='#ECDCFB'
        ))
        fig2.update_layout(barmode='stack', title='Work hours structure', height=420)
        st.plotly_chart(fig2, use_container_width=True)

    # Chart 3: Billable utilization (line + filled area)
    with c3:
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=monthly_summary['month_str'],
            y=monthly_summary['utilization_pct'],
            mode='lines+markers',
            name='Billable utilization rate',
            line=dict(color='#4DA6FF'),
            fill='tozeroy',
            fillcolor='rgba(77,166,255,0.2)'
        ))
        fig3.update_layout(yaxis=dict(title='Utilization %'), title='Billable utilization', height=420)
        st.plotly_chart(fig3, use_container_width=True)

    # Show summary KPI cards (Total hours, Work hours, Billable hours, Utilization)
    tot_hours = int(monthly_summary['total_hours'].sum())
    work_hours = tot_hours
    bill_hours = int(monthly_summary['billable_hours'].sum())
    util_avg = round(monthly_summary['utilization_pct'].mean(), 1)

    st.subheader('Company performance (selected range)')
    k1, k2, k3, k4 = st.columns(4)
    k1.metric('Total hours', f"{tot_hours:,}")
    k2.metric('Work hours', f"{work_hours:,}")
    k3.metric('Billable hours', f"{bill_hours:,}")
    k4.metric('Billable utilization', f"{util_avg}%")

    # warn if detected active consultants fewer than expected
    # Count using normalized employee ids to avoid float/int mismatch (e.g. 2.0 vs 2)
    if 'employee_id_norm' not in df.columns:
        df['employee_id_norm'] = df['employee_id'].apply(_normalize_id)
    total_active_consultants = int(df['employee_id_norm'].dropna().nunique())
    if total_active_consultants < 10:
        st.warning(f"Detected {total_active_consultants} active consultants in the filtered data. You mentioned there should be at least 10 — please check employee_id mapping or data sources.")
