import streamlit as st
from app import load_process_and_cache
import plotly.express as px
import pandas as pd

# Get the cached data
sales_pipeline, invoices = load_process_and_cache()

# Add a header
st.header("Key Metric")
st.markdown("---")

# test to see if everything works below here

st.subheader("Sales Pipeline Overview")
# Display sales pipeline data
st.dataframe(sales_pipeline, use_container_width=True)

# Show some metrics
if not sales_pipeline.empty:
    total_pipeline = invoices['invoice_amount_net'].sum()
    avg_deal_size = invoices['invoice_amount_net'].mean()
    
    metric_col1, metric_col2 = st.columns(2)
    metric_col1.metric("Net invoiced amount", f"${total_pipeline:,.2f}")
    metric_col2.metric("Average Deal Size", f"${avg_deal_size:,.2f}")


st.subheader("Invoice Analysis")
# Display invoice data
st.dataframe(invoices, use_container_width=True)