import streamlit as st

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
