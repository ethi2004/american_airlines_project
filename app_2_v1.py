import streamlit as st
import pandas as pd
from catboost import CatBoostClassifier

# Load your saved model
model = CatBoostClassifier()
model.load_model("flight_delay_model.cbm")

st.title("Flight Delay Predictor")

# User inputs
carrier = st.text_input("Carrier Code (e.g., AA)", "AA")
flight_num = st.text_input("Flight Number (e.g., 80)", "80")
dep_airport = st.text_input("Departure Airport IATA", "DFW")
arr_airport = st.text_input("Arrival Airport IATA", "LHR")
dep_hour = st.number_input("Scheduled Departure Hour (0-23)", min_value=0, max_value=23, value=18)
arr_hour = st.number_input("Scheduled Arrival Hour (0-23)", min_value=0, max_value=23, value=6)
dep_date = st.date_input("Scheduled Departure Date", pd.to_datetime("2025-12-03"))

day_of_week = dep_date.weekday()
month = dep_date.month

# Prepare input
test_flight = pd.DataFrame({
    "OPERAT_CARRIER_CD": [carrier],
    "OPERAT_FLIGHT_NBR": [flight_num],
    "SCHD_DEP_AIRPRT_IATA_CD": [dep_airport],
    "ARVL_AIRPRT_IATA_CD": [arr_airport],
    "SCHD_LEG_DEP_TMS": [dep_hour],
    "dep_hour": [dep_hour],
    "arr_hour": [arr_hour],
    "day_of_week": [day_of_week],
    "month": [month]
})

if st.button("Predict Delay"):
    y_pred = model.predict(test_flight)[0]
    y_proba = model.predict_proba(test_flight)[:, 1][0]

    if y_pred == 1:
        st.error("✈️ Yes, the flight will likely be delayed 15+ minutes.")
    else:
        st.success("🟢 No, the flight is unlikely to be delayed 15+ minutes.")

    st.write(f"Probability of delay: {y_proba:.2f}")
