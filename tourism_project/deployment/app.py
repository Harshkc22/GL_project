"""
Streamlit front end for the Wellness Tourism Package predictor.
Loads the trained model from the Hugging Face model hub and lets a user
enter a customer's details to get a purchase-likelihood prediction.
"""
import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download

HF_USERNAME = "harshkc"
MODEL_REPO_ID = f"{HF_USERNAME}/tourism-wellness-model"


@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id=MODEL_REPO_ID,
        filename="best_model.joblib",
        repo_type="model",
    )
    return joblib.load(model_path)


model = load_model()

st.title("Visit with Us — Wellness Tourism Package Predictor")
st.write(
    "Enter a customer's details below to predict whether they are likely "
    "to purchase the new Wellness Tourism Package."
)

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=35)
    typeof_contact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
    city_tier = st.selectbox("City Tier", [1, 2, 3])
    duration_of_pitch = st.number_input("Duration Of Pitch (minutes)", min_value=0, max_value=120, value=10)
    occupation = st.selectbox("Occupation", ["Salaried", "Free Lancer", "Small Business", "Large Business"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    num_person_visiting = st.number_input("Number Of Persons Visiting", min_value=1, max_value=10, value=2)
    num_followups = st.number_input("Number Of Followups", min_value=0, max_value=10, value=3)
    product_pitched = st.selectbox("Product Pitched", ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"])

with col2:
    preferred_star = st.selectbox("Preferred Property Star", [3.0, 4.0, 5.0])
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    num_trips = st.number_input("Number Of Trips (per year)", min_value=0, max_value=25, value=2)
    passport = st.selectbox("Holds a Passport", [0, 1])
    pitch_score = st.slider("Pitch Satisfaction Score", 1, 5, 3)
    own_car = st.selectbox("Owns a Car", [0, 1])
    num_children = st.number_input("Number Of Children Visiting", min_value=0, max_value=5, value=0)
    designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
    monthly_income = st.number_input("Monthly Income", min_value=1000, max_value=100000, value=20000)

if st.button("Predict"):
    input_df = pd.DataFrame([{
        "Age": age,
        "TypeofContact": typeof_contact,
        "CityTier": city_tier,
        "DurationOfPitch": duration_of_pitch,
        "Occupation": occupation,
        "Gender": gender,
        "NumberOfPersonVisiting": num_person_visiting,
        "NumberOfFollowups": num_followups,
        "ProductPitched": product_pitched,
        "PreferredPropertyStar": preferred_star,
        "MaritalStatus": marital_status,
        "NumberOfTrips": num_trips,
        "Passport": passport,
        "PitchSatisfactionScore": pitch_score,
        "OwnCar": own_car,
        "NumberOfChildrenVisiting": num_children,
        "Designation": designation,
        "MonthlyIncome": monthly_income,
    }])

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.success(f"✅ Likely to purchase the Wellness Package (probability: {probability:.1%})")
    else:
        st.warning(f"❌ Unlikely to purchase the Wellness Package (probability: {probability:.1%})")
