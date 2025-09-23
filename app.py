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
    get_monthly_invoice_pivot,
    get_invoice_pivot
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
    sales_pipeline, invoices = process_data(df1, df9, df7)
    
    return sales_pipeline, invoices

sales_pipeline, invoices = load_process_and_cache()

# landing page for dashboard
st.header('Key Metrics Dashboard')
st.markdown("---")
st.write("")

st.markdown("""
This dashboard provides insights into sales performance and invoicing trends blah blah blah add some more text here once done.              
""")
st.write("")    
st.write("")
st.write("")
st.header('Created By')
st.markdown("---")

# name and contact info for each team member
team = {
    "Muhammad": "https://www.linkedin.com/in/alshakarti",
    "Aron": "https://www.linkedin.com",
    "Lara": "https://www.linkedin.com",
    "William": "https://www.linkedin.com",
}
cols = st.columns(5)
for (name, contact_info), col in zip(team.items(), cols):
    with col:
        st.markdown(f"**{name}**")
        st.link_button('Go to linkedin profile', contact_info)
        st.write("")
