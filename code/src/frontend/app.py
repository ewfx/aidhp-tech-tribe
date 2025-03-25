import streamlit as st
import requests

st.title("🔍 AI-Powered Financial Product Recommender")

query = st.text_input("Enter your financial preferences (e.g., 'I have a high income and want safe investments')")

if st.button("Search"):
    res = requests.post("http://127.0.0.1:8000/query", json={"query": query})
    if res.status_code == 200:
        st.success(f"💡 Recommended Financial Product: **{res.json()['recommended_product']}**")
    else:
        st.error("❌ Error retrieving recommendation. Please try again.")
