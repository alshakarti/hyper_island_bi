# setting up environment
import streamlit as st
import pandas as pd
from src.data_processing import (
    get_month_end_date,
    get_month_filter_data,
    load_process_and_store
)
from src.visualization import (
    plot_invoice_amounts_line,
    create_kpi_cards
)

# Load data
sales_pipeline, invoices, payments, time_reporting, monthly_hours, start_date, end_date = load_process_and_store()


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

amount_type = st.sidebar.selectbox(
    "Select Amount Type",
    options=['net', 'payments', 'revenue'],
    index=0,
    help="Select amount type"
)
if amount_type == 'net':
    show_broker = st.sidebar.selectbox(
        "Amount breakdown by broker",
        options=['No', 'Yes'],
        index=0,
        help="Show separate bars for each broker type"
    )
    show_broker_bool = (show_broker == 'Yes')
else: 
    show_broker_bool = False

financial_trend_line = st.sidebar.selectbox(
    "Show regression line",
    options=["Yes", "No"],
    index=1,
    help="Show regression line for financial trend"
)
financial_trend_line_bool = (financial_trend_line == "Yes")

# KPI Cards section
#st.markdown("#### Financial trend") 
st.subheader("Financial trend") 
st.write(f"Showing {amount_type} amount from {selected_start_month} to {selected_end_month}")
st.markdown("---")

# Get KPI cards data
kpi_data = create_kpi_cards(
    invoices, 
    payments, 
    time_reporting, 
    amount_type, 
    start_date=start_date_str, 
    end_date=end_date_str,
    show_broker=show_broker_bool
)

if kpi_data:
    if not show_broker_bool:
        # Show simple 2-column layout for main metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            if kpi_data['main_avg']:
                st.metric(
                    label=f"Average {amount_type.title()}", 
                    value=kpi_data['main_avg']['formatted_value']
                )
        with col2:
            if kpi_data['main_sum']:
                st.metric(
                    label=f"Total {amount_type.title()}", 
                    value=kpi_data['main_sum']['formatted_value']
                )
        with col3:
            # placeholder to maintain layout
            st.metric(label="", value="")  
    else:
        # Show broker breakdown - use consistent 3-column layout
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if kpi_data['main_avg']:
                st.metric(
                    label="Average Net Amount", 
                    value=kpi_data['main_avg']['formatted_value']
                )
        with col2:
            if kpi_data['main_sum']:
                st.metric(
                    label="Total Net Amount", 
                    value=kpi_data['main_sum']['formatted_value']
                )
        with col3:
            # placeholder to maintain layout
            st.metric(label="", value="")  
                
        # Create tabs for each broker type
        tab1, tab2, tab3 = st.tabs(["Broker", "Direct", "Partner"])
        
        broker_data = kpi_data['broker_breakdown']
        
        with tab1:
            col1, col2, col3 = st.columns(3)
            if 'broker' in broker_data:
                with col1:
                    if broker_data['broker']['avg_amount']:
                        st.metric("Avg Amount", broker_data['broker']['avg_amount']['formatted_value'])
                with col2:
                    if broker_data['broker']['sum_amount']:
                        st.metric("Total Amount", broker_data['broker']['sum_amount']['formatted_value'])
                with col3:
                    if broker_data['broker']['avg_percentage']:
                        st.metric("% of Net", broker_data['broker']['avg_percentage']['formatted_value'])
        
        with tab2:
            col1, col2, col3 = st.columns(3)
            if 'direct' in broker_data:
                with col1:
                    if broker_data['direct']['avg_amount']:
                        st.metric("Avg Amount", broker_data['direct']['avg_amount']['formatted_value'])
                with col2:
                    if broker_data['direct']['sum_amount']:
                        st.metric("Total Amount", broker_data['direct']['sum_amount']['formatted_value'])
                with col3:
                    if broker_data['direct']['avg_percentage']:
                        st.metric("% of Net", broker_data['direct']['avg_percentage']['formatted_value'])
        
        with tab3:
            col1, col2, col3 = st.columns(3)
            if 'partner' in broker_data:
                with col1:
                    if broker_data['partner']['avg_amount']:
                        st.metric("Avg Amount", broker_data['partner']['avg_amount']['formatted_value'])
                with col2:
                    if broker_data['partner']['sum_amount']:
                        st.metric("Total Amount", broker_data['partner']['sum_amount']['formatted_value'])
                with col3:
                    if broker_data['partner']['avg_percentage']:
                        st.metric("% of Net", broker_data['partner']['avg_percentage']['formatted_value'])

fig = plot_invoice_amounts_line(
    invoices,
    payments,
    start_date=start_date_str,  
    end_date=end_date_str,     
    amount_type=amount_type,
    hue=show_broker_bool,
    show_trend=financial_trend_line_bool
)
if fig:
    st.plotly_chart(fig, use_container_width=False)
else:
    st.warning("No data available for the selected filters.")