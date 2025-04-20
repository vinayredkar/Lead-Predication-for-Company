import streamlit as st
import pandas as pd
import joblib
from datetime import datetime
import joblib

# Replace with your exact path
model_path = r"D:\New folder\project\predication model\lead_scoring_model.pkl"

# Load the model
model = joblib.load(model_path)

# Verify model features (if available)
try:
    print("Model features:", model.feature_names_in_)
except AttributeError:
    print("Model doesn't store feature names. Ensure input data matches training.")
# Title
st.title("XYZ Lead Conversion Predictor")   
st.markdown("""
Predict the likelihood of a lead becoming a buying customer.
""")

# --- Input Form ---
with st.form("lead_form"):
    st.header("Lead Details")
    
    # Website Engagement
    st.subheader("Website Activity")
    col1, col2 = st.columns(2)
    with col1:
        banner_event = st.number_input("Banner clicks", min_value=0, value=0)
        download_link_click = st.number_input("Download link clicks", min_value=0, value=0)
        ecomtry = st.number_input("E-commerce link clicks", min_value=0, value=0)
        newsletter_subscribe = st.number_input("Newsletter subscriptions", min_value=0, value=0)
    with col2:
        page_navigation = st.number_input("Page navigations", min_value=0, value=0)
        part_number_search = st.number_input("Part number searches", min_value=0, value=0)
        search = st.number_input("Search bar uses", min_value=0, value=0)
        search_result_click = st.number_input("Search result clicks", min_value=0, value=0)
    
    # Company Attributes
    st.subheader("Company Information")
    col3, col4 = st.columns(2)
    with col3:
        has_rental_fleet = st.checkbox("Has a rental fleet")
        num_admin = st.number_input("Administrative staff", min_value=0, value=0)
        num_machines_internal = st.number_input("Machines for internal use", min_value=0, value=0)
        num_machines_rental = st.number_input("Machines in rental fleet", min_value=0, value=0)
    with col4:
        num_machines_serviced = st.number_input("Machines serviced", min_value=0, value=0)
        num_service_vans = st.number_input("Service vans", min_value=0, value=0)
        num_technicians = st.number_input("Technicians", min_value=0, value=0)
    
    # Categorical Features
    st.subheader("Additional Details")
    company_type = st.selectbox("Company Type", ["", "cleaning", "ltd", "bv", "llc", "mechanical"])
    macro_region = st.selectbox("Region", ["EMEA", "AMERICAS", "APAC"])
    sells_parts = st.checkbox("Sells parts of equipment")
    owns_equipment = st.checkbox("Owns industrial equipment")
    services_own_equipment = st.checkbox("Services own equipment")
    services_rental_fleet = st.checkbox("Services rental fleet")
    services_customer_equipment = st.checkbox("Services customer equipment")
    
    # Dates for conversion time
    created_date = st.date_input("Form submission date")
    converted_date = st.date_input("Conversion date")
    
    submit_button = st.form_submit_button("Predict")

# --- Prediction Logic ---
if submit_button:
    # Calculate days to convert
    days_to_convert = (converted_date - created_date).days
    
    # One-hot encode categoricals
    company2_encoded = {f"company2_{company_type}": 1} if company_type else {}
    region_encoded = {
        "macro_region_EMEA": 1 if macro_region == "EMEA" else 0,
        "macro_region_AMERICAS": 1 if macro_region == "AMERICAS" else 0,
        "macro_region_APAC": 1 if macro_region == "APAC" else 0
    }
    
    # Prepare feature dictionary
    features = {
        'banner_event': banner_event,
        'download_link_click': download_link_click,
        'ecomtry': ecomtry,
        'newsletter_subscribe': newsletter_subscribe,
        'page_navigation': page_navigation,
        'part_number_search': part_number_search,
        'search': search,
        'search_result_click': search_result_click,
        'distinct_days_active': 1,  # Default (can add input)
        'time_active_minutes': 30.0,  # Default (can add input)
        'Has_a_Rental_Fleet': int(has_rental_fleet),
        'number_of_administrative_people': num_admin,
        'number_of_machines_for_internal_use': num_machines_internal,
        'number_of_machines_in_rental': num_machines_rental,
        'number_of_machines_serviced': num_machines_serviced,
        'number_of_service_vans': num_service_vans,
        'number_of_technicians': num_technicians,
        'sell_parts_of_equipment': int(sells_parts),
        'owns_industrial_equip_for_internal_use': int(owns_equipment),
        'Service_owned_equipment': int(services_own_equipment),
        'Service_rental_fleet_Yes': int(services_rental_fleet),
        'services_customers_equipment': int(services_customer_equipment),
        'days_to_convert': days_to_convert,
        **company2_encoded,
        **region_encoded
    }
    
    # Convert to DataFrame
    input_df = pd.DataFrame([features])
    
    # Ensure all model features are present (fill missing with 0)
    input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)
    
    # Predict
    probability = model.predict_proba(input_df)[0][1]
    
    # Display result
    st.success(f"Conversion Probability: **{probability:.1%}**")
    if probability > 0.7:
        st.markdown("🎯 **High Priority Lead**")
    elif probability > 0.3:
        st.markdown("⚠️ **Medium Priority Lead**")
    else:
        st.markdown("🔍 **Low Priority Lead**")

    # Show raw features (optional)
    with st.expander("View submitted data"):
        st.json(features)