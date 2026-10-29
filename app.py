
import joblib
import pandas as pd
import streamlit as st
from pathlib import Path

# ==========================================
# 1. CONFIGURATION
# ==========================================
BASE_DIR = Path(__file__).resolve().parent
REFERENCE_YEAR = 2018

st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="wide"
)


# ==========================================
# 2. LOAD MODEL AND DATASET
# ==========================================
@st.cache_resource
def load_model():
    return joblib.load(BASE_DIR / "car_price_model.pkl")


@st.cache_data
def load_dataset():
    return pd.read_csv(BASE_DIR / "car data.csv")


try:
    model = load_model()
    df = load_dataset()

except Exception as exc:
    st.error(f"Unable to load project files: {exc}")
    st.info(
        "Check that car_price_model.pkl and car data.csv "
        "are present in the project folder."
    )
    st.stop()


# ==========================================
# 3. PAGE HEADER
# ==========================================
st.title("🚗 Car Price Prediction")
st.caption("CodeAlpha Data Science Internship Project")

st.markdown(
    """
    Estimate the selling price of a used car using a
    machine learning model trained on historical vehicle data.
    """
)

st.warning(
    "This model uses historical data ending in 2018. "
    "Its predictions are educational estimates, not current market valuations."
)

st.divider()


# ==========================================
# 4. PREPARE DROPDOWN OPTIONS
# ==========================================
car_names = sorted(
    df["Car_Name"].dropna().astype(str).unique().tolist()
)

if not car_names:
    st.error("No car models were found in the dataset.")
    st.stop()


# ==========================================
# 5. INPUT FORM
# ==========================================
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Enter Vehicle Details")

    with st.form("prediction_form"):

        car_name = st.selectbox(
            "Car Model",
            car_names
        )

        left, right = st.columns(2)

        with left:
            present_price = st.number_input(
                "Present Price (₹ lakh)",
                min_value=0.01,
                value=5.0,
                step=0.5
            )

            driven_kms = st.number_input(
                "Kilometers Driven",
                min_value=0,
                max_value=1_000_000,
                value=30000,
                step=1000
            )

            car_year = st.selectbox(
                "Manufacturing Year",
                list(range(2003, REFERENCE_YEAR + 1)),
                index=10
            )

        with right:
            fuel_type = st.selectbox(
                "Fuel Type",
                ["Petrol", "Diesel", "CNG"]
            )

            selling_type = st.selectbox(
                "Seller Type",
                ["Dealer", "Individual"]
            )

            transmission = st.selectbox(
                "Transmission",
                ["Manual", "Automatic"]
            )

            owner = st.selectbox(
                "Previous Owners",
                [0, 1, 2, 3]
            )

        submitted = st.form_submit_button(
            "Predict Selling Price",
            type="primary",
            use_container_width=True
        )


# ==========================================
# 6. MODEL INFORMATION
# ==========================================
with col2:
    st.subheader("About This Model")

    st.metric(
        "Unique Dataset Records",
        len(df.drop_duplicates())
    )

    st.metric(
        "Input Features",
        8
    )

    st.write(
        "Model: Gradient Boosting Regressor"
    )

    st.write(
        "The saved pipeline includes preprocessing "
        "and the trained regression model."
    )


# ==========================================
# 7. PREDICTION
# ==========================================
if submitted:

    car_age = REFERENCE_YEAR - car_year

    input_data = pd.DataFrame([{
        "Car_Name": car_name,
        "Present_Price": present_price,
        "Driven_kms": driven_kms,
        "Fuel_Type": fuel_type,
        "Selling_type": selling_type,
        "Transmission": transmission,
        "Owner": owner,
        "Car_Age": car_age
    }])

    try:
        prediction = float(model.predict(input_data)[0])

        st.divider()
        st.subheader("Prediction Result")

        if not pd.notna(prediction):
            st.error("The model returned an invalid prediction.")

        elif prediction < 0:
            st.error(
                f"The model predicted ₹{prediction:.2f} lakh. "
                "A negative selling price is not valid. "
                "Try another set of inputs."
            )

        else:
            # Display price in lakhs and approximate rupees.
            st.success(
                f"Estimated Selling Price: ₹{prediction:.2f} lakh"
            )

            st.metric(
                "Approximate Amount in Rupees",
                f"₹{prediction * 100000:,.0f}"
            )

            st.caption(
                "This is a model estimate based on historical data. "
                "Actual selling prices may differ substantially."
            )

            # Show the submitted vehicle details.
            with st.expander("View submitted vehicle details"):
                display_data = input_data.copy()
                display_data["Car_Age"] = car_age
                display_data = display_data.rename(
                    columns={
                        "Car_Name": "Car Model",
                        "Present_Price": "Present Price (₹ lakh)",
                        "Driven_kms": "Kilometers Driven",
                        "Fuel_Type": "Fuel Type",
                        "Selling_type": "Seller Type",
                        "Transmission": "Transmission",
                        "Owner": "Previous Owners",
                        "Car_Age": "Age relative to 2018"
                    }
                )

                st.dataframe(
                    display_data,
                    use_container_width=True,
                    hide_index=True
                )

    except Exception as exc:
        st.error(f"Prediction failed: {exc}")


# ==========================================
# 8. FOOTER
# ==========================================
st.divider()

st.caption(
    "Built with Python, Pandas, Scikit-learn and Streamlit."
)
