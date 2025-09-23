import streamlit as st
from app import sales_funnel_viz
from app import load_process_and_cache

# Get the cached data
sales_pipeline, invoices, payments = load_process_and_cache()

# Create columns for side-by-side charts
col1, col2 = st.columns(2)

with col1:
    st.subheader("Weighted Deal Amounts")
    # For weighted deal amounts
    fig1 = sales_funnel_viz(sales_pipeline, weighted_amount=True)
    st.plotly_chart(fig1, use_container_width=True)