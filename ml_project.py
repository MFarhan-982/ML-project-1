import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load the trained Random Forest model
try:
    model = joblib.load('random_forest_model.joblib')
except FileNotFoundError:
    st.error("Model file not found! Ensure 'random_forest_model.joblib' exists.")

st.title("💧 Water Potability Prediction App")

st.write("🔍 Enter water quality parameters below to check if the water is safe to drink.")

# Input fields with recommended ranges
ph = st.number_input("pH Level (Ideal: 6.5 - 8.5)", min_value=0.0, max_value=14.0, step=0.1)
hardness = st.number_input("Hardness (Higher levels may indicate contaminants)", min_value=0.0)
solids = st.number_input("Total Dissolved Solids (Ideal: Below 500)", min_value=0.0)
chloramines = st.number_input("Chloramines (Used in disinfection)", min_value=0.0)
sulfate = st.number_input("Sulfate (High levels may cause water hardness)", min_value=0.0)
conductivity = st.number_input("Conductivity (Measures ion concentration)", min_value=0.0)
organic_carbon = st.number_input("Organic Carbon (Higher values may indicate pollutants)", min_value=0.0)
trihalomethanes = st.number_input("Trihalomethanes (Byproducts of chlorination, ideal <80)", min_value=0.0)
turbidity = st.number_input("Turbidity (Higher values indicate cloudy water)", min_value=0.0)

# Predict button
if st.button("🔮 Predict Potability"):
    input_data = np.array([[ph, hardness, solids, chloramines, sulfate,
                            conductivity, organic_carbon, trihalomethanes, turbidity]])

    prediction = model.predict(input_data)[0]
    confidence = model.predict_proba(input_data)[0]  # Get probability scores

    result = "✅ Safe to Drink" if prediction == 1 else "❌ Not Safe to Drink"

    # Display Prediction
    st.subheader(f"Prediction: {result}")
    st.write(f"🔹 Confidence Score: {confidence[prediction]:.2%}")

    # Visualization: Confidence Distribution
    fig, ax = plt.subplots()
    ax.bar(["Safe", "Not Safe"], confidence, color=['green', 'red'])
    ax.set_ylabel("Confidence Level")
    ax.set_title("Model's Prediction Confidence")
    st.pyplot(fig)

    # Save Prediction Report
    report_data = {
        "pH": ph, "Hardness": hardness, "Solids": solids, "Chloramines": chloramines,
        "Sulfate": sulfate, "Conductivity": conductivity, "Organic Carbon": organic_carbon,
        "Trihalomethanes": trihalomethanes, "Turbidity": turbidity, "Prediction": result,
        "Confidence": f"{confidence[prediction]:.2%}"
    }
    df_report = pd.DataFrame([report_data])
    st.download_button(label="📄 Download Report", data=df_report.to_csv(index=False),
                       file_name="water_potability_report.csv", mime="text/csv")

# Footer
st.write("🌍 Ensuring safe drinking water is important. Always test multiple samples for accurate results!")
