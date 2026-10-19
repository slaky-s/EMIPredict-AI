import streamlit as st
import pandas as pd
import mlflow
from sqlalchemy import create_engine, text
from urllib.parse import quote_plus

# =========================================================
# MYSQL DATABASE CONNECTION
# =========================================================

username = "root"
password = "Laky%40123"
host = "localhost"
database = "emipredict_ai"

engine = create_engine(
    f"mysql+mysqlconnector://{username}:{password}@{host}/{database}"
)

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="EMIPredict AI",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("💰 EMIPredict AI")

st.sidebar.markdown(
    "### Intelligent Financial Risk Assessment Platform"
)

st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "🔮 EMI Risk Prediction",
        "💰 EMI Amount Prediction",
        "📊 Data Exploration",
        "🤖 Model Performance",
        "⚙️ MLflow Models",
        "🗄️ Admin / CRUD"
    ]
)


# =========================================================
# HOME PAGE
# =========================================================

if page == "🏠 Home":

    st.title("💰 EMIPredict AI")

    st.subheader(
        "Intelligent Financial Risk Assessment Platform"
    )

    st.markdown(
        """
        ### 📌 Project Overview

        EMIPredict AI is a machine learning based financial risk
        assessment platform designed to help evaluate EMI eligibility
        and estimate the maximum monthly EMI amount.

        The platform uses financial, demographic, employment and
        credit-related information to generate data-driven predictions.
        """
    )

    st.divider()

    # -----------------------------------------------------
    # Key Features
    # -----------------------------------------------------

    st.subheader("🚀 Key Features")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Dataset Records",
            "404,800"
        )

    with col2:
        st.metric(
            "ML Models",
            "6"
        )

    with col3:
        st.metric(
            "MLflow Registered Models",
            "2"
        )

    st.divider()

    # -----------------------------------------------------
    # ML Capabilities
    # -----------------------------------------------------

    st.subheader("🧠 Machine Learning Capabilities")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("### 🔮 Classification")

        st.write(
            """
            Predicts EMI eligibility using three categories:
            
            - **Eligible**
            - **High Risk**
            - **Not Eligible**
            """
        )

        st.info(
            "Production model: XGBoost Classifier"
        )

    with col2:

        st.markdown("### 💰 Regression")

        st.write(
            """
            Estimates the maximum monthly EMI amount that
            can be supported based on the applicant's financial profile.
            """
        )

        st.info(
            "Production model: XGBoost Regressor"
        )

    st.divider()

    # -----------------------------------------------------
    # Technology Stack
    # -----------------------------------------------------

    st.subheader("🛠️ Technology Stack")

    st.markdown(
        """
        **Python** • **Pandas** • **Scikit-learn** •
        **XGBoost** • **MLflow** • **Streamlit**
        """
    )

    st.success(
        "EMIPredict AI is ready for financial risk assessment."
    )


# =========================================================
# PLACEHOLDER PAGES
# =========================================================

elif page == "🔮 EMI Risk Prediction":

    st.title("🔮 EMI Risk Prediction")

    st.markdown(
        """
        Enter the applicant's financial and demographic information
        to assess their EMI eligibility.
        """
    )

    st.divider()

    # =====================================================
    # LOAD REGISTERED MODEL
    # =====================================================

    import mlflow
    import pandas as pd

    classification_model = mlflow.pyfunc.load_model(
    "models/classification_model"
)

    # =====================================================
    # PERSONAL INFORMATION
    # =====================================================

    st.subheader("👤 Personal Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=30
        )

    with col2:
        gender = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )

    with col3:
        marital_status = st.selectbox(
            "Marital Status",
            ["Married", "Single"]
        )

    col1, col2 = st.columns(2)

    with col1:
        education = st.selectbox(
            "Education",
            [
                "High School",
                "Graduate",
                "Post Graduate",
                "Professional"
            ]
        )

    with col2:
        family_size = st.number_input(
            "Family Size",
            min_value=1,
            max_value=20,
            value=3
        )

    dependents = st.number_input(
        "Number of Dependents",
        min_value=0,
        max_value=20,
        value=1
    )

    # =====================================================
    # EMPLOYMENT INFORMATION
    # =====================================================

    st.subheader("💼 Employment Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        monthly_salary = st.number_input(
            "Monthly Salary (₹)",
            min_value=0.0,
            value=50000.0,
            step=1000.0
        )

    with col2:
        employment_type = st.selectbox(
            "Employment Type",
            [
                "Private",
                "Government",
                "Self-employed"
            ]
        )

    with col3:
        years_of_employment = st.number_input(
            "Years of Employment",
            min_value=0.0,
            max_value=50.0,
            value=5.0,
            step=0.5
        )

    company_type = st.selectbox(
        "Company Type",
        [
            "Large Indian",
            "MNC",
            "Mid-size",
            "Startup",
            "Small"
        ]
    )

    # =====================================================
    # HOUSING & EXPENSES
    # =====================================================

    st.subheader("🏠 Housing & Monthly Expenses")

    col1, col2 = st.columns(2)

    with col1:
        house_type = st.selectbox(
            "House Type",
            [
                "Rented",
                "Own",
                "Family"
            ]
        )

    with col2:
        monthly_rent = st.number_input(
            "Monthly Rent (₹)",
            min_value=0.0,
            value=5000.0,
            step=500.0
        )

    col1, col2, col3 = st.columns(3)

    with col1:
        school_fees = st.number_input(
            "School Fees (₹)",
            min_value=0.0,
            value=0.0,
            step=500.0
        )

    with col2:
        college_fees = st.number_input(
            "College Fees (₹)",
            min_value=0.0,
            value=0.0,
            step=500.0
        )

    with col3:
        travel_expenses = st.number_input(
            "Travel Expenses (₹)",
            min_value=0.0,
            value=5000.0,
            step=500.0
        )

    col1, col2 = st.columns(2)

    with col1:
        groceries_utilities = st.number_input(
            "Groceries & Utilities (₹)",
            min_value=0.0,
            value=10000.0,
            step=500.0
        )

    with col2:
        other_monthly_expenses = st.number_input(
            "Other Monthly Expenses (₹)",
            min_value=0.0,
            value=5000.0,
            step=500.0
        )

    # =====================================================
    # EXISTING FINANCIAL OBLIGATIONS
    # =====================================================

    st.subheader("💳 Existing Financial Obligations")

    col1, col2 = st.columns(2)

    with col1:
        existing_loans = st.selectbox(
            "Existing Loans",
            ["No", "Yes"]
        )

    with col2:
        current_emi_amount = st.number_input(
            "Current EMI Amount (₹)",
            min_value=0.0,
            value=0.0,
            step=500.0
        )

    # =====================================================
    # CREDIT & FINANCIAL RESERVES
    # =====================================================

    st.subheader("📊 Credit & Financial Reserves")

    col1, col2, col3 = st.columns(3)

    with col1:
        credit_score = st.number_input(
            "Credit Score",
            min_value=0.0,
            max_value=1200.0,
            value=700.0,
            step=1.0
        )

    with col2:
        bank_balance = st.number_input(
            "Bank Balance (₹)",
            min_value=0.0,
            value=100000.0,
            step=1000.0
        )

    with col3:
        emergency_fund = st.number_input(
            "Emergency Fund (₹)",
            min_value=0.0,
            value=50000.0,
            step=1000.0
        )

    # =====================================================
    # EMI REQUEST
    # =====================================================

    st.subheader("💰 EMI Request")

    col1, col2, col3 = st.columns(3)

    with col1:
        emi_scenario = st.selectbox(
            "EMI Scenario",
            [
                "Home Appliances EMI",
                "Personal Loan EMI",
                "E-commerce Shopping EMI",
                "Education EMI",
                "Vehicle EMI"
            ]
        )

    with col2:
        requested_amount = st.number_input(
            "Requested Amount (₹)",
            min_value=0.0,
            value=100000.0,
            step=5000.0
        )

    with col3:
        requested_tenure = st.number_input(
            "Requested Tenure (Months)",
            min_value=1,
            max_value=120,
            value=24
        )

    st.divider()

    # =====================================================
    # PREDICTION BUTTON
    # =====================================================

    if st.button(
        "🔮 Assess EMI Eligibility",
        type="primary",
        use_container_width=True
    ):

        # -------------------------------------------------
        # FEATURE ENGINEERING
        # -------------------------------------------------

        total_education_expenses = (
            school_fees + college_fees
        )

        total_living_expenses = (
            monthly_rent
            + travel_expenses
            + groceries_utilities
            + other_monthly_expenses
        )

        total_monthly_expenses = (
            total_living_expenses
            + total_education_expenses
        )

        total_financial_reserves = (
            bank_balance
            + emergency_fund
        )

        income = max(monthly_salary, 1)

        debt_to_income_ratio = (
            current_emi_amount / income
        )

        expense_to_income_ratio = (
            total_monthly_expenses / income
        )

        affordability_ratio = (
            (
                monthly_salary
                - total_monthly_expenses
                - current_emi_amount
            ) / income
        )

        disposable_income = (
            monthly_salary
            - total_monthly_expenses
            - current_emi_amount
        )

        # -------------------------------------------------
        # CREDIT RISK
        # -------------------------------------------------

        if credit_score <= 579:
            credit_risk_category = "Very_High_Risk"
            credit_risk_score = 1

        elif credit_score <= 669:
            credit_risk_category = "High_Risk"
            credit_risk_score = 2

        elif credit_score <= 739:
            credit_risk_category = "Moderate_Risk"
            credit_risk_score = 3

        elif credit_score <= 799:
            credit_risk_category = "Low_Risk"
            credit_risk_score = 4

        else:
            credit_risk_category = "Very_Low_Risk"
            credit_risk_score = 5

        # -------------------------------------------------
        # EMPLOYMENT STABILITY
        # -------------------------------------------------

        if years_of_employment <= 1:
            employment_stability_category = "Very_Low_Stability"
            employment_stability_score = 1

        elif years_of_employment <= 3:
            employment_stability_category = "Low_Stability"
            employment_stability_score = 2

        elif years_of_employment <= 5:
            employment_stability_category = "Moderate_Stability"
            employment_stability_score = 3

        elif years_of_employment <= 10:
            employment_stability_category = "High_Stability"
            employment_stability_score = 4

        else:
            employment_stability_category = "Very_High_Stability"
            employment_stability_score = 5

        # -------------------------------------------------
        # INTERACTION FEATURES
        # -------------------------------------------------

        income_credit_risk_interaction = (
            monthly_salary * credit_risk_score
        )

        income_employment_stability_interaction = (
            monthly_salary * employment_stability_score
        )

        reserve_credit_risk_interaction = (
            total_financial_reserves * credit_risk_score
        )

        financial_stability_interaction = (
            credit_risk_score
            * employment_stability_score
        )

        requested_to_reserve_ratio = (
            requested_amount
            / max(total_financial_reserves, 1)
        )

        requested_to_income_ratio = (
            requested_amount
            / max(monthly_salary, 1)
        )

        # -------------------------------------------------
        # CREATE INPUT DATAFRAME
        # -------------------------------------------------

        input_data = pd.DataFrame([{
            "age": age,
            "gender": gender,
            "marital_status": marital_status,
            "education": education,
            "monthly_salary": monthly_salary,
            "employment_type": employment_type,
            "years_of_employment": years_of_employment,
            "company_type": company_type,
            "house_type": house_type,
            "monthly_rent": monthly_rent,
            "family_size": family_size,
            "dependents": dependents,
            "school_fees": school_fees,
            "college_fees": college_fees,
            "travel_expenses": travel_expenses,
            "groceries_utilities": groceries_utilities,
            "other_monthly_expenses": other_monthly_expenses,
            "existing_loans": existing_loans,
            "current_emi_amount": current_emi_amount,
            "credit_score": credit_score,
            "bank_balance": bank_balance,
            "emergency_fund": emergency_fund,
            "emi_scenario": emi_scenario,
            "requested_amount": requested_amount,
            "requested_tenure": requested_tenure,
            "total_education_expenses": total_education_expenses,
            "total_living_expenses": total_living_expenses,
            "total_monthly_expenses": total_monthly_expenses,
            "total_financial_reserves": total_financial_reserves,
            "debt_to_income_ratio": debt_to_income_ratio,
            "expense_to_income_ratio": expense_to_income_ratio,
            "affordability_ratio": affordability_ratio,
            "disposable_income": disposable_income,
            "credit_risk_category": credit_risk_category,
            "credit_risk_score": credit_risk_score,
            "employment_stability_category": employment_stability_category,
            "employment_stability_score": employment_stability_score,
            "income_credit_risk_interaction": income_credit_risk_interaction,
            "income_employment_stability_interaction":
                income_employment_stability_interaction,
            "reserve_credit_risk_interaction":
                reserve_credit_risk_interaction,
            "financial_stability_interaction":
                financial_stability_interaction,
            "requested_to_reserve_ratio":
                requested_to_reserve_ratio,
            "requested_to_income_ratio":
                requested_to_income_ratio
        }])

        # -------------------------------------------------
        # MODEL PREDICTION
        # -------------------------------------------------

        prediction = classification_model.predict(
            input_data
        )

        prediction_value = int(prediction[0])

        prediction_mapping = {
            0: "Not_Eligible",
            1: "Eligible",
            2: "High_Risk"
        }

        result = prediction_mapping[prediction_value]

        # -------------------------------------------------
        # DISPLAY RESULT
        # -------------------------------------------------

        st.divider()

        st.subheader("📋 EMI Eligibility Assessment")

        if result == "Eligible":

            st.success(
                "✅ EMI Status: Eligible"
            )

        elif result == "High_Risk":

            st.warning(
                "⚠️ EMI Status: High Risk"
            )

        else:

            st.error(
                "❌ EMI Status: Not Eligible"
            )

        st.metric(
            "Predicted Eligibility",
            result
        )


elif page == "💰 EMI Amount Prediction":

    st.title("💰 EMI Amount Prediction")

    st.markdown(
        """
        Enter the applicant's financial and demographic information
        to estimate the maximum monthly EMI amount.
        """
    )

    st.divider()

    # =====================================================
    # LOAD REGISTERED REGRESSION MODEL
    # =====================================================

    import mlflow
    import pandas as pd

    regression_model = mlflow.pyfunc.load_model("models/regression_model")

    # =====================================================
    # PERSONAL INFORMATION
    # =====================================================

    st.subheader("👤 Personal Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=30
        )

    with col2:
        gender = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )

    with col3:
        marital_status = st.selectbox(
            "Marital Status",
            ["Married", "Single"]
        )

    col1, col2 = st.columns(2)

    with col1:
        education = st.selectbox(
            "Education",
            [
                "High School",
                "Graduate",
                "Post Graduate",
                "Professional"
            ]
        )

    with col2:
        family_size = st.number_input(
            "Family Size",
            min_value=1,
            max_value=20,
            value=3
        )

    dependents = st.number_input(
        "Number of Dependents",
        min_value=0,
        max_value=20,
        value=1
    )

    # =====================================================
    # EMPLOYMENT INFORMATION
    # =====================================================

    st.subheader("💼 Employment Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        monthly_salary = st.number_input(
            "Monthly Salary (₹)",
            min_value=0.0,
            value=50000.0,
            step=1000.0
        )

    with col2:
        employment_type = st.selectbox(
            "Employment Type",
            [
                "Private",
                "Government",
                "Self-employed"
            ]
        )

    with col3:
        years_of_employment = st.number_input(
            "Years of Employment",
            min_value=0.0,
            max_value=50.0,
            value=5.0,
            step=0.5
        )

    company_type = st.selectbox(
        "Company Type",
        [
            "Large Indian",
            "MNC",
            "Mid-size",
            "Startup",
            "Small"
        ]
    )

    # =====================================================
    # HOUSING & EXPENSES
    # =====================================================

    st.subheader("🏠 Housing & Monthly Expenses")

    col1, col2 = st.columns(2)

    with col1:
        house_type = st.selectbox(
            "House Type",
            [
                "Rented",
                "Own",
                "Family"
            ]
        )

    with col2:
        monthly_rent = st.number_input(
            "Monthly Rent (₹)",
            min_value=0.0,
            value=5000.0,
            step=500.0
        )

    col1, col2, col3 = st.columns(3)

    with col1:
        school_fees = st.number_input(
            "School Fees (₹)",
            min_value=0.0,
            value=0.0,
            step=500.0
        )

    with col2:
        college_fees = st.number_input(
            "College Fees (₹)",
            min_value=0.0,
            value=0.0,
            step=500.0
        )

    with col3:
        travel_expenses = st.number_input(
            "Travel Expenses (₹)",
            min_value=0.0,
            value=5000.0,
            step=500.0
        )

    col1, col2 = st.columns(2)

    with col1:
        groceries_utilities = st.number_input(
            "Groceries & Utilities (₹)",
            min_value=0.0,
            value=10000.0,
            step=500.0
        )

    with col2:
        other_monthly_expenses = st.number_input(
            "Other Monthly Expenses (₹)",
            min_value=0.0,
            value=5000.0,
            step=500.0
        )

    # =====================================================
    # EXISTING FINANCIAL OBLIGATIONS
    # =====================================================

    st.subheader("💳 Existing Financial Obligations")

    col1, col2 = st.columns(2)

    with col1:
        existing_loans = st.selectbox(
            "Existing Loans",
            ["No", "Yes"]
        )

    with col2:
        current_emi_amount = st.number_input(
            "Current EMI Amount (₹)",
            min_value=0.0,
            value=0.0,
            step=500.0
        )

    # =====================================================
    # CREDIT & FINANCIAL RESERVES
    # =====================================================

    st.subheader("📊 Credit & Financial Reserves")

    col1, col2, col3 = st.columns(3)

    with col1:
        credit_score = st.number_input(
            "Credit Score",
            min_value=0.0,
            max_value=1200.0,
            value=700.0,
            step=1.0
        )

    with col2:
        bank_balance = st.number_input(
            "Bank Balance (₹)",
            min_value=0.0,
            value=100000.0,
            step=1000.0
        )

    with col3:
        emergency_fund = st.number_input(
            "Emergency Fund (₹)",
            min_value=0.0,
            value=50000.0,
            step=1000.0
        )

    # =====================================================
    # EMI REQUEST
    # =====================================================

    st.subheader("💰 EMI Request")

    col1, col2, col3 = st.columns(3)

    with col1:
        emi_scenario = st.selectbox(
            "EMI Scenario",
            [
                "Home Appliances EMI",
                "Personal Loan EMI",
                "E-commerce Shopping EMI",
                "Education EMI",
                "Vehicle EMI"
            ]
        )

    with col2:
        requested_amount = st.number_input(
            "Requested Amount (₹)",
            min_value=0.0,
            value=100000.0,
            step=5000.0
        )

    with col3:
        requested_tenure = st.number_input(
            "Requested Tenure (Months)",
            min_value=1,
            max_value=120,
            value=24
        )

    st.divider()

    # =====================================================
    # PREDICTION BUTTON
    # =====================================================

    if st.button(
        "💰 Predict Maximum EMI",
        type="primary",
        use_container_width=True
    ):

        # -------------------------------------------------
        # FEATURE ENGINEERING
        # -------------------------------------------------

        total_education_expenses = (
            school_fees + college_fees
        )

        total_living_expenses = (
            monthly_rent
            + travel_expenses
            + groceries_utilities
            + other_monthly_expenses
        )

        total_monthly_expenses = (
            total_living_expenses
            + total_education_expenses
        )

        total_financial_reserves = (
            bank_balance + emergency_fund
        )

        income = max(monthly_salary, 1)

        debt_to_income_ratio = (
            current_emi_amount / income
        )

        expense_to_income_ratio = (
            total_monthly_expenses / income
        )

        affordability_ratio = (
            (
                monthly_salary
                - total_monthly_expenses
                - current_emi_amount
            ) / income
        )

        disposable_income = (
            monthly_salary
            - total_monthly_expenses
            - current_emi_amount
        )

        # -------------------------------------------------
        # CREDIT RISK
        # -------------------------------------------------

        if credit_score <= 579:
            credit_risk_category = "Very_High_Risk"
            credit_risk_score = 1

        elif credit_score <= 669:
            credit_risk_category = "High_Risk"
            credit_risk_score = 2

        elif credit_score <= 739:
            credit_risk_category = "Moderate_Risk"
            credit_risk_score = 3

        elif credit_score <= 799:
            credit_risk_category = "Low_Risk"
            credit_risk_score = 4

        else:
            credit_risk_category = "Very_Low_Risk"
            credit_risk_score = 5

        # -------------------------------------------------
        # EMPLOYMENT STABILITY
        # -------------------------------------------------

        if years_of_employment <= 1:
            employment_stability_category = "Very_Low_Stability"
            employment_stability_score = 1

        elif years_of_employment <= 3:
            employment_stability_category = "Low_Stability"
            employment_stability_score = 2

        elif years_of_employment <= 5:
            employment_stability_category = "Moderate_Stability"
            employment_stability_score = 3

        elif years_of_employment <= 10:
            employment_stability_category = "High_Stability"
            employment_stability_score = 4

        else:
            employment_stability_category = "Very_High_Stability"
            employment_stability_score = 5

        # -------------------------------------------------
        # INTERACTION FEATURES
        # -------------------------------------------------

        income_credit_risk_interaction = (
            monthly_salary * credit_risk_score
        )

        income_employment_stability_interaction = (
            monthly_salary * employment_stability_score
        )

        reserve_credit_risk_interaction = (
            total_financial_reserves * credit_risk_score
        )

        financial_stability_interaction = (
            credit_risk_score
            * employment_stability_score
        )

        requested_to_reserve_ratio = (
            requested_amount
            / max(total_financial_reserves, 1)
        )

        requested_to_income_ratio = (
            requested_amount
            / max(monthly_salary, 1)
        )

        # -------------------------------------------------
        # CREATE INPUT DATAFRAME
        # -------------------------------------------------

        input_data = pd.DataFrame([{
            "age": age,
            "gender": gender,
            "marital_status": marital_status,
            "education": education,
            "monthly_salary": monthly_salary,
            "employment_type": employment_type,
            "years_of_employment": years_of_employment,
            "company_type": company_type,
            "house_type": house_type,
            "monthly_rent": monthly_rent,
            "family_size": family_size,
            "dependents": dependents,
            "school_fees": school_fees,
            "college_fees": college_fees,
            "travel_expenses": travel_expenses,
            "groceries_utilities": groceries_utilities,
            "other_monthly_expenses": other_monthly_expenses,
            "existing_loans": existing_loans,
            "current_emi_amount": current_emi_amount,
            "credit_score": credit_score,
            "bank_balance": bank_balance,
            "emergency_fund": emergency_fund,
            "emi_scenario": emi_scenario,
            "requested_amount": requested_amount,
            "requested_tenure": requested_tenure,
            "total_education_expenses": total_education_expenses,
            "total_living_expenses": total_living_expenses,
            "total_monthly_expenses": total_monthly_expenses,
            "total_financial_reserves": total_financial_reserves,
            "debt_to_income_ratio": debt_to_income_ratio,
            "expense_to_income_ratio": expense_to_income_ratio,
            "affordability_ratio": affordability_ratio,
            "disposable_income": disposable_income,
            "credit_risk_category": credit_risk_category,
            "credit_risk_score": credit_risk_score,
            "employment_stability_category":
                employment_stability_category,
            "employment_stability_score":
                employment_stability_score,
            "income_credit_risk_interaction":
                income_credit_risk_interaction,
            "income_employment_stability_interaction":
                income_employment_stability_interaction,
            "reserve_credit_risk_interaction":
                reserve_credit_risk_interaction,
            "financial_stability_interaction":
                financial_stability_interaction,
            "requested_to_reserve_ratio":
                requested_to_reserve_ratio,
            "requested_to_income_ratio":
                requested_to_income_ratio
        }])

        # -------------------------------------------------
        # MODEL PREDICTION
        # -------------------------------------------------

        prediction = regression_model.predict(
            input_data
        )

        predicted_emi = float(prediction[0])

        # Prevent displaying a negative EMI if the model
        # ever produces a small negative prediction.
        predicted_emi = max(predicted_emi, 0)

        # -------------------------------------------------
        # DISPLAY RESULT
        # -------------------------------------------------

        st.divider()

        st.subheader("📋 EMI Amount Assessment")

        st.metric(
            "Predicted Maximum Monthly EMI",
            f"₹{predicted_emi:,.2f}"
        )

        st.success(
            f"The model estimates a maximum monthly EMI "
            f"of ₹{predicted_emi:,.2f}."
        )


elif page == "📊 Data Exploration":

    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    st.title("📊 Data Exploration")

    st.markdown(
        """
        Explore the financial dataset and understand the patterns
        associated with EMI eligibility.
        """
    )

    # =====================================================
    # LOAD DATASET
    # =====================================================

    @st.cache_data
    def load_data():

        data = pd.read_csv(
            "/Users/lakshmi/Desktop/python_projects/Project_3/emi_prediction_dataset.csv",
            low_memory=False
        )

        # Fix malformed numeric values
        data["age"] = (
            data["age"]
            .astype(str)
            .str.replace(".0.0", ".0", regex=False)
        )
        data["age"] = pd.to_numeric(
            data["age"],
            errors="coerce"
        )

        data["monthly_salary"] = (
            data["monthly_salary"]
            .astype(str)
            .str.replace(".0.0", ".0", regex=False)
        )
        data["monthly_salary"] = pd.to_numeric(
            data["monthly_salary"],
            errors="coerce"
        )

        data["bank_balance"] = (
            data["bank_balance"]
            .astype(str)
            .str.replace(".0.0", ".0", regex=False)
            .replace("nan.0", pd.NA)
        )

        data["bank_balance"] = pd.to_numeric(
            data["bank_balance"],
            errors="coerce"
        )

        # Standardize gender
        data["gender"] = (
            data["gender"]
            .astype(str)
            .str.strip()
            .str.upper()
            .replace({
                "M": "Male",
                "MALE": "Male",
                "F": "Female",
                "FEMALE": "Female"
            })
        )

        # Missing-value treatment
        data["education"] = data["education"].fillna(
            data["education"].mode()[0]
        )

        for column in [
            "monthly_rent",
            "credit_score",
            "bank_balance",
            "emergency_fund"
        ]:
            data[column] = data[column].fillna(
                data[column].median()
            )

        data["monthly_salary"] = data["monthly_salary"].fillna(
            data["monthly_salary"].median()
        )

        return data


    df_explore = load_data()

    # =====================================================
    # DATASET SUMMARY
    # =====================================================

    st.subheader("📌 Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Records",
            f"{len(df_explore):,}"
        )

    with col2:
        st.metric(
            "Total Columns",
            len(df_explore.columns)
        )

    with col3:
        st.metric(
            "Eligible",
            f"{(df_explore['emi_eligibility'] == 'Eligible').sum():,}"
        )

    with col4:
        st.metric(
            "Not Eligible",
            f"{(df_explore['emi_eligibility'] == 'Not_Eligible').sum():,}"
        )

    st.divider()

    # =====================================================
    # FILTER
    # =====================================================

    st.subheader("🔎 Interactive Filters")

    selected_scenario = st.multiselect(
        "Select EMI Scenario",
        sorted(df_explore["emi_scenario"].dropna().unique()),
        default=sorted(
            df_explore["emi_scenario"].dropna().unique()
        )
    )

    filtered_df = df_explore[
        df_explore["emi_scenario"].isin(selected_scenario)
    ]

    st.write(
        f"Showing **{len(filtered_df):,}** records"
    )

    st.divider()

    # =====================================================
    # ELIGIBILITY DISTRIBUTION
    # =====================================================

    st.subheader("1️⃣ EMI Eligibility Distribution")

    fig, ax = plt.subplots(figsize=(8, 5))

    sns.countplot(
        data=filtered_df,
        x="emi_eligibility",
        order=[
            "Eligible",
            "High_Risk",
            "Not_Eligible"
        ],
        ax=ax
    )

    ax.set_title("EMI Eligibility Distribution")
    ax.set_xlabel("EMI Eligibility")
    ax.set_ylabel("Number of Customers")

    plt.xticks(rotation=0)

    st.pyplot(fig)

    plt.close(fig)

    # =====================================================
    # SCENARIO VS ELIGIBILITY
    # =====================================================

    st.subheader("2️⃣ EMI Scenario vs Eligibility")

    scenario_table = pd.crosstab(
        filtered_df["emi_scenario"],
        filtered_df["emi_eligibility"],
        normalize="index"
    ) * 100

    scenario_table = scenario_table[
        [
            column
            for column in [
                "Eligible",
                "High_Risk",
                "Not_Eligible"
            ]
            if column in scenario_table.columns
        ]
    ]

    fig, ax = plt.subplots(figsize=(10, 6))

    scenario_table.plot(
        kind="bar",
        stacked=True,
        ax=ax
    )

    ax.set_title(
        "Eligibility Distribution by EMI Scenario"
    )
    ax.set_xlabel("EMI Scenario")
    ax.set_ylabel("Percentage (%)")

    plt.xticks(rotation=30, ha="right")

    st.pyplot(fig)

    plt.close(fig)

    st.dataframe(
        scenario_table.round(2),
        use_container_width=True
    )

    # =====================================================
    # SALARY VS ELIGIBILITY
    # =====================================================

    st.subheader("3️⃣ Monthly Salary vs Eligibility")

    fig, ax = plt.subplots(figsize=(9, 5))

    sns.boxplot(
        data=filtered_df,
        x="emi_eligibility",
        y="monthly_salary",
        order=[
            "Eligible",
            "High_Risk",
            "Not_Eligible"
        ],
        ax=ax
    )

    ax.set_title(
        "Monthly Salary Distribution by EMI Eligibility"
    )
    ax.set_xlabel("EMI Eligibility")
    ax.set_ylabel("Monthly Salary (₹)")

    st.pyplot(fig)

    plt.close(fig)

    # =====================================================
    # CREDIT SCORE VS ELIGIBILITY
    # =====================================================

    st.subheader("4️⃣ Credit Score vs Eligibility")

    fig, ax = plt.subplots(figsize=(9, 5))

    sns.boxplot(
        data=filtered_df,
        x="emi_eligibility",
        y="credit_score",
        order=[
            "Eligible",
            "High_Risk",
            "Not_Eligible"
        ],
        ax=ax
    )

    ax.set_title(
        "Credit Score Distribution by EMI Eligibility"
    )
    ax.set_xlabel("EMI Eligibility")
    ax.set_ylabel("Credit Score")

    st.pyplot(fig)

    plt.close(fig)

    # =====================================================
    # EXISTING LOANS VS ELIGIBILITY
    # =====================================================

    st.subheader("5️⃣ Existing Loans vs Eligibility")

    loan_table = pd.crosstab(
        filtered_df["existing_loans"],
        filtered_df["emi_eligibility"],
        normalize="index"
    ) * 100

    fig, ax = plt.subplots(figsize=(8, 5))

    loan_table[
        [
            column
            for column in [
                "Eligible",
                "High_Risk",
                "Not_Eligible"
            ]
            if column in loan_table.columns
        ]
    ].plot(
        kind="bar",
        stacked=True,
        ax=ax
    )

    ax.set_title(
        "EMI Eligibility by Existing Loan Status"
    )
    ax.set_xlabel("Existing Loans")
    ax.set_ylabel("Percentage (%)")

    plt.xticks(rotation=0)

    st.pyplot(fig)

    plt.close(fig)

    st.dataframe(
        loan_table.round(2),
        use_container_width=True
    )

    # =====================================================
    # HOUSE TYPE VS ELIGIBILITY
    # =====================================================

    st.subheader("6️⃣ House Type vs Eligibility")

    house_table = pd.crosstab(
        filtered_df["house_type"],
        filtered_df["emi_eligibility"],
        normalize="index"
    ) * 100

    fig, ax = plt.subplots(figsize=(8, 5))

    house_table[
        [
            column
            for column in [
                "Eligible",
                "High_Risk",
                "Not_Eligible"
            ]
            if column in house_table.columns
        ]
    ].plot(
        kind="bar",
        stacked=True,
        ax=ax
    )

    ax.set_title(
        "EMI Eligibility by House Type"
    )
    ax.set_xlabel("House Type")
    ax.set_ylabel("Percentage (%)")

    plt.xticks(rotation=0)

    st.pyplot(fig)

    plt.close(fig)

    st.dataframe(
        house_table.round(2),
        use_container_width=True
    )

    # =====================================================
    # FINANCIAL VARIABLES
    # =====================================================

    st.subheader("7️⃣ Financial Metrics by Eligibility")

    financial_summary = (
        filtered_df
        .groupby("emi_eligibility")[
            [
                "monthly_salary",
                "current_emi_amount",
                "credit_score",
                "bank_balance",
                "emergency_fund",
                "requested_amount",
                "max_monthly_emi"
            ]
        ]
        .mean()
        .reindex(
            [
                "Eligible",
                "High_Risk",
                "Not_Eligible"
            ]
        )
    )

    st.dataframe(
        financial_summary.round(2),
        use_container_width=True
    )

    # =====================================================
    # CORRELATION HEATMAP
    # =====================================================

    st.subheader("8️⃣ Financial Correlation Heatmap")

    correlation_columns = [
        "monthly_salary",
        "monthly_rent",
        "school_fees",
        "college_fees",
        "travel_expenses",
        "groceries_utilities",
        "other_monthly_expenses",
        "current_emi_amount",
        "credit_score",
        "bank_balance",
        "emergency_fund",
        "requested_amount",
        "requested_tenure",
        "max_monthly_emi"
    ]

    correlation_matrix = filtered_df[
        correlation_columns
    ].corr()

    fig, ax = plt.subplots(
        figsize=(12, 9)
    )

    sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        ax=ax
    )

    ax.set_title(
        "Correlation Between Financial Variables"
    )

    st.pyplot(fig)

    plt.close(fig)

elif page == "🤖 Model Performance":

    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    st.title("🤖 Model Performance")

    st.markdown(
        """
        Compare the machine learning models trained for
        EMI eligibility classification and maximum EMI regression.
        """
    )

    # =====================================================
    # CLASSIFICATION PERFORMANCE
    # =====================================================

    st.subheader("🔮 Classification Models")

    classification_results = pd.DataFrame({
        "Model": [
            "Logistic Regression",
            "Random Forest",
            "XGBoost"
        ],
        "Accuracy": [
            0.7970,
            0.9165,
            0.9713
        ],
        "Precision": [
            0.6577,
            0.7274,
            0.9103
        ],
        "Recall": [
            0.7491,
            0.8062,
            0.8295
        ],
        "F1 Score": [
            0.6440,
            0.7561,
            0.8597
        ],
        "ROC-AUC": [
            0.9312,
            0.9735,
            0.9936
        ]
    })

    st.dataframe(
        classification_results.style.format({
            "Accuracy": "{:.4f}",
            "Precision": "{:.4f}",
            "Recall": "{:.4f}",
            "F1 Score": "{:.4f}",
            "ROC-AUC": "{:.4f}"
        }),
        use_container_width=True,
        hide_index=True
    )

    # -----------------------------------------------------
    # Classification comparison chart
    # -----------------------------------------------------

    classification_chart = classification_results.melt(
        id_vars="Model",
        var_name="Metric",
        value_name="Score"
    )

    fig, ax = plt.subplots(
        figsize=(11, 6)
    )

    sns.barplot(
        data=classification_chart,
        x="Model",
        y="Score",
        hue="Metric",
        ax=ax
    )

    ax.set_title(
        "Classification Model Performance"
    )
    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)

    plt.xticks(rotation=0)

    st.pyplot(fig)

    plt.close(fig)

    # -----------------------------------------------------
    # Selected classification model
    # -----------------------------------------------------

    st.success(
        "🏆 Selected Classification Model: XGBoost"
    )

    st.info(
        "XGBoost achieved the highest validation Accuracy "
        "and ROC-AUC among the three classification models."
    )

    st.divider()

    # =====================================================
    # REGRESSION PERFORMANCE
    # =====================================================

    st.subheader("💰 Regression Models")

    regression_results = pd.DataFrame({
        "Model": [
            "Linear Regression",
            "Random Forest",
            "XGBoost"
        ],
        "RMSE": [
            4018.56,
            897.69,
            667.77
        ],
        "MAE": [
            2875.24,
            292.22,
            240.88
        ],
        "R²": [
            0.7306,
            0.9866,
            0.9926
        ],
        "MAPE": [
            189.43,
            5.47,
            6.64
        ]
    })

    st.dataframe(
        regression_results.style.format({
            "RMSE": "{:.2f}",
            "MAE": "{:.2f}",
            "R²": "{:.4f}",
            "MAPE": "{:.2f}%"
        }),
        use_container_width=True,
        hide_index=True
    )

    # -----------------------------------------------------
    # Regression error metrics
    # -----------------------------------------------------

    st.subheader("📉 Regression Error Metrics")

    regression_error_chart = regression_results.melt(
        id_vars="Model",
        value_vars=["RMSE", "MAE"],
        var_name="Metric",
        value_name="Error"
    )

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    sns.barplot(
        data=regression_error_chart,
        x="Model",
        y="Error",
        hue="Metric",
        ax=ax
    )

    ax.set_title(
        "Regression Error Comparison"
    )
    ax.set_xlabel("Model")
    ax.set_ylabel("Error")

    st.pyplot(fig)

    plt.close(fig)

    # -----------------------------------------------------
    # Regression R2
    # -----------------------------------------------------

    st.subheader("📈 R² Comparison")

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    sns.barplot(
        data=regression_results,
        x="Model",
        y="R²",
        ax=ax
    )

    ax.set_title(
        "Regression R² Comparison"
    )
    ax.set_xlabel("Model")
    ax.set_ylabel("R²")
    ax.set_ylim(0, 1.05)

    st.pyplot(fig)

    plt.close(fig)

    # -----------------------------------------------------
    # Selected regression model
    # -----------------------------------------------------

    st.success(
        "🏆 Selected Regression Model: XGBoost"
    )

    st.info(
        "XGBoost achieved the lowest RMSE and MAE and "
        "the highest R² among the three regression models."
    )

    st.divider()

    # =====================================================
    # FINAL PRODUCTION MODELS
    # =====================================================

    st.subheader("🚀 Production Models")

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Classification Model",
            "XGBoost"
        )

        st.write(
            "MLflow Registry: "
            "`EMIPredict_Classification_Model`"
        )

        st.write(
            "Version: **1**"
        )

    with col2:

        st.metric(
            "Regression Model",
            "XGBoost"
        )

        st.write(
            "MLflow Registry: "
            "`EMIPredict_Regression_Model`"
        )

        st.write(
            "Version: **1**"
        )

elif page == "⚙️ MLflow Models":

    import mlflow
    import pandas as pd

    st.title("⚙️ MLflow Model Management")

    st.markdown(
        """
        View the production models registered in the
        EMIPredict AI MLflow Model Registry.
        """
    )

    st.divider()

    # =====================================================
    # MLflow CONFIGURATION
    # =====================================================

    classification_model_name = (
        "EMIPredict_Classification_Model"
    )

    regression_model_name = (
        "EMIPredict_Regression_Model"
    )

    classification_version = "1"
    regression_version = "1"

    classification_run_id = (
        "262d3532f7b14e11ae5c55c20b1c14bb"
    )

    regression_run_id = (
        "f8293897b7114df5a88c0e511ef274db"
    )

    # =====================================================
    # MODEL SUMMARY
    # =====================================================

    st.subheader("🚀 Registered Production Models")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("### 🔮 Classification")

        st.metric(
            "Model",
            "XGBoost"
        )

        st.write(
            "**Registry Name:** "
            "`EMIPredict_Classification_Model`"
        )

        st.write(
            "**Version:** 1"
        )

        st.write(
            "**Task:** EMI Eligibility Classification"
        )

        st.success(
            "Production Model"
        )

    with col2:

        st.markdown("### 💰 Regression")

        st.metric(
            "Model",
            "XGBoost"
        )

        st.write(
            "**Registry Name:** "
            "`EMIPredict_Regression_Model`"
        )

        st.write(
            "**Version:** 1"
        )

        st.write(
            "**Task:** Maximum Monthly EMI Prediction"
        )

        st.success(
            "Production Model"
        )

    st.divider()

    # =====================================================
    # MODEL REGISTRY TABLE
    # =====================================================

    st.subheader("📋 Model Registry")

    registry_data = pd.DataFrame({
        "Model Name": [
            classification_model_name,
            regression_model_name
        ],
        "Model Type": [
            "Classification",
            "Regression"
        ],
        "Algorithm": [
            "XGBoost",
            "XGBoost"
        ],
        "Version": [
            classification_version,
            regression_version
        ],
        "Status": [
            "Production",
            "Production"
        ],
        "Run ID": [
            classification_run_id,
            regression_run_id
        ]
    })

    st.dataframe(
        registry_data,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # =====================================================
    # CLASSIFICATION RUN INFORMATION
    # =====================================================

    st.subheader("🔮 Classification Model Details")

    classification_metrics = pd.DataFrame({
        "Metric": [
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score",
            "ROC-AUC"
        ],
        "Validation Value": [
            0.9713,
            0.9103,
            0.8295,
            0.8597,
            0.9936
        ]
    })

    st.dataframe(
        classification_metrics.style.format({
            "Validation Value": "{:.4f}"
        }),
        use_container_width=True,
        hide_index=True
    )

    st.write(
        "**MLflow Run ID:**"
    )

    st.code(
        classification_run_id
    )

    st.divider()

    # =====================================================
    # REGRESSION RUN INFORMATION
    # =====================================================

    st.subheader("💰 Regression Model Details")

    regression_metrics = pd.DataFrame({
        "Metric": [
            "RMSE",
            "MAE",
            "R²",
            "MAPE"
        ],
        "Validation Value": [
            667.77,
            240.88,
            0.9926,
            6.64
        ]
    })

    st.dataframe(
        regression_metrics.style.format({
            "Validation Value": "{:.2f}"
        }),
        use_container_width=True,
        hide_index=True
    )

    st.write(
        "**MLflow Run ID:**"
    )

    st.code(
        regression_run_id
    )

    st.divider()

    # =====================================================
    # MODEL LOADING TEST
    # =====================================================

    st.subheader("🔍 Model Availability")

    if st.button(
        "🔄 Verify Registered Models",
        use_container_width=True
    ):

        try:

            classification_model = mlflow.pyfunc.load_model("models/classification_model")

            regression_model = mlflow.pyfunc.load_model("models/regression_model")

            st.success(
                "✅ Both registered MLflow models are "
                "available and load successfully."
            )

            col1, col2 = st.columns(2)

            with col1:
                st.write(
                    "🔮 Classification Model: **Available**"
                )

            with col2:
                st.write(
                    "💰 Regression Model: **Available**"
                )

        except Exception as e:

            st.error(
                "Unable to load one or more registered models."
            )

            st.exception(e)


# =========================================================
# ADMIN / CRUD
# =========================================================

elif page == "🗄️ Admin / CRUD":

    st.title("🗄️ Admin / CRUD")

    st.markdown(
        """
        Manage financial records stored in the
        **emipredict_ai → financial_records** MySQL table.
        """
    )

    st.divider()

    # =====================================================
    # READ - VIEW ALL RECORDS
    # =====================================================

    st.subheader("📋 View Financial Records")

    try:

        records_query = """
        SELECT *
        FROM financial_records
        ORDER BY record_id DESC
        """

        records = pd.read_sql(
            records_query,
            engine
        )

        st.write(
            f"Total records: **{len(records)}**"
        )

        if len(records) > 0:

            st.dataframe(
                records,
                use_container_width=True,
                hide_index=True
            )

        else:

            st.info(
                "No financial records found."
            )

    except Exception as e:

        st.error(
            "Unable to retrieve financial records."
        )

        st.exception(e)

    st.divider()

    # =====================================================
    # CREATE - ADD NEW RECORD
    # =====================================================

    st.subheader("➕ Add Financial Record")

    with st.form("create_record_form"):

        # -------------------------------------------------
        # Personal Information
        # -------------------------------------------------

        st.markdown("### 👤 Personal Information")

        col1, col2, col3 = st.columns(3)

        with col1:

            create_age = st.number_input(
                "Age",
                min_value=18.0,
                max_value=100.0,
                value=30.0,
                step=1.0,
                key="create_age"
            )

            create_gender = st.selectbox(
                "Gender",
                ["Male", "Female"],
                key="create_gender"
            )

            create_marital_status = st.selectbox(
                "Marital Status",
                ["Single", "Married"],
                key="create_marital_status"
            )

        with col2:

            create_education = st.selectbox(
                "Education",
                [
                    "High School",
                    "Graduate",
                    "Post Graduate",
                    "Professional"
                ],
                key="create_education"
            )

            create_family_size = st.number_input(
                "Family Size",
                min_value=1,
                max_value=20,
                value=3,
                key="create_family_size"
            )

            create_dependents = st.number_input(
                "Dependents",
                min_value=0,
                max_value=20,
                value=1,
                key="create_dependents"
            )

        with col3:

            create_house_type = st.selectbox(
                "House Type",
                ["Rented", "Own", "Family"],
                key="create_house_type"
            )

            create_monthly_rent = st.number_input(
                "Monthly Rent (₹)",
                min_value=0.0,
                value=5000.0,
                step=500.0,
                key="create_monthly_rent"
            )

        # -------------------------------------------------
        # Employment Information
        # -------------------------------------------------

        st.markdown("### 💼 Employment Information")

        col1, col2, col3, col4 = st.columns(4)

        with col1:

            create_monthly_salary = st.number_input(
                "Monthly Salary (₹)",
                min_value=0.0,
                value=50000.0,
                step=1000.0,
                key="create_monthly_salary"
            )

        with col2:

            create_employment_type = st.selectbox(
                "Employment Type",
                [
                    "Private",
                    "Government",
                    "Self-employed"
                ],
                key="create_employment_type"
            )

        with col3:

            create_years_of_employment = st.number_input(
                "Years of Employment",
                min_value=0.0,
                max_value=50.0,
                value=5.0,
                step=0.5,
                key="create_years_of_employment"
            )

        with col4:

            create_company_type = st.selectbox(
                "Company Type",
                [
                    "Large Indian",
                    "MNC",
                    "Mid-size",
                    "Startup",
                    "Small"
                ],
                key="create_company_type"
            )

        # -------------------------------------------------
        # Monthly Expenses
        # -------------------------------------------------

        st.markdown("### 💰 Monthly Expenses")

        col1, col2, col3, col4 = st.columns(4)

        with col1:

            create_school_fees = st.number_input(
                "School Fees (₹)",
                min_value=0.0,
                value=0.0,
                step=500.0,
                key="create_school_fees"
            )

        with col2:

            create_college_fees = st.number_input(
                "College Fees (₹)",
                min_value=0.0,
                value=0.0,
                step=500.0,
                key="create_college_fees"
            )

        with col3:

            create_travel_expenses = st.number_input(
                "Travel Expenses (₹)",
                min_value=0.0,
                value=5000.0,
                step=500.0,
                key="create_travel_expenses"
            )

        with col4:

            create_groceries_utilities = st.number_input(
                "Groceries & Utilities (₹)",
                min_value=0.0,
                value=10000.0,
                step=500.0,
                key="create_groceries_utilities"
            )

        create_other_monthly_expenses = st.number_input(
            "Other Monthly Expenses (₹)",
            min_value=0.0,
            value=5000.0,
            step=500.0,
            key="create_other_monthly_expenses"
        )

        # -------------------------------------------------
        # Existing Financial Obligations
        # -------------------------------------------------

        st.markdown(
            "### 💳 Existing Financial Obligations"
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            create_existing_loans = st.selectbox(
                "Existing Loans",
                ["No", "Yes"],
                key="create_existing_loans"
            )

        with col2:

            create_current_emi_amount = st.number_input(
                "Current EMI Amount (₹)",
                min_value=0.0,
                value=0.0,
                step=500.0,
                key="create_current_emi_amount"
            )

        with col3:

            create_credit_score = st.number_input(
                "Credit Score",
                min_value=0.0,
                max_value=1200.0,
                value=700.0,
                step=1.0,
                key="create_credit_score"
            )

        # -------------------------------------------------
        # Financial Reserves
        # -------------------------------------------------

        st.markdown("### 🏦 Financial Reserves")

        col1, col2 = st.columns(2)

        with col1:

            create_bank_balance = st.number_input(
                "Bank Balance (₹)",
                min_value=0.0,
                value=100000.0,
                step=1000.0,
                key="create_bank_balance"
            )

        with col2:

            create_emergency_fund = st.number_input(
                "Emergency Fund (₹)",
                min_value=0.0,
                value=50000.0,
                step=1000.0,
                key="create_emergency_fund"
            )

        # -------------------------------------------------
        # EMI Request
        # -------------------------------------------------

        st.markdown("### 🛒 EMI Request")

        col1, col2, col3 = st.columns(3)

        with col1:

            create_emi_scenario = st.selectbox(
                "EMI Scenario",
                [
                    "Home Appliances EMI",
                    "Personal Loan EMI",
                    "E-commerce Shopping EMI",
                    "Education EMI",
                    "Vehicle EMI"
                ],
                key="create_emi_scenario"
            )

        with col2:

            create_requested_amount = st.number_input(
                "Requested Amount (₹)",
                min_value=0.0,
                value=100000.0,
                step=5000.0,
                key="create_requested_amount"
            )

        with col3:

            create_requested_tenure = st.number_input(
                "Requested Tenure (Months)",
                min_value=1,
                max_value=120,
                value=24,
                key="create_requested_tenure"
            )

        # -------------------------------------------------
        # Model Output
        # -------------------------------------------------

        st.markdown("### 🤖 Model Output")

        col1, col2 = st.columns(2)

        with col1:

            create_emi_eligibility = st.selectbox(
                "EMI Eligibility",
                [
                    "Eligible",
                    "High_Risk",
                    "Not_Eligible"
                ],
                key="create_emi_eligibility"
            )

        with col2:

            create_max_monthly_emi = st.number_input(
                "Maximum Monthly EMI (₹)",
                min_value=0.0,
                value=5000.0,
                step=500.0,
                key="create_max_monthly_emi"
            )

        create_button = st.form_submit_button(
            "➕ Add Record",
            use_container_width=True
        )

    # =====================================================
    # INSERT RECORD
    # =====================================================

    if create_button:

        insert_query = text("""
            INSERT INTO financial_records (
                age,
                gender,
                marital_status,
                education,
                monthly_salary,
                employment_type,
                years_of_employment,
                company_type,
                house_type,
                monthly_rent,
                family_size,
                dependents,
                school_fees,
                college_fees,
                travel_expenses,
                groceries_utilities,
                other_monthly_expenses,
                existing_loans,
                current_emi_amount,
                credit_score,
                bank_balance,
                emergency_fund,
                emi_scenario,
                requested_amount,
                requested_tenure,
                emi_eligibility,
                max_monthly_emi
            )
            VALUES (
                :age,
                :gender,
                :marital_status,
                :education,
                :monthly_salary,
                :employment_type,
                :years_of_employment,
                :company_type,
                :house_type,
                :monthly_rent,
                :family_size,
                :dependents,
                :school_fees,
                :college_fees,
                :travel_expenses,
                :groceries_utilities,
                :other_monthly_expenses,
                :existing_loans,
                :current_emi_amount,
                :credit_score,
                :bank_balance,
                :emergency_fund,
                :emi_scenario,
                :requested_amount,
                :requested_tenure,
                :emi_eligibility,
                :max_monthly_emi
            )
        """)

        insert_values = {
            "age": create_age,
            "gender": create_gender,
            "marital_status": create_marital_status,
            "education": create_education,
            "monthly_salary": create_monthly_salary,
            "employment_type": create_employment_type,
            "years_of_employment": create_years_of_employment,
            "company_type": create_company_type,
            "house_type": create_house_type,
            "monthly_rent": create_monthly_rent,
            "family_size": create_family_size,
            "dependents": create_dependents,
            "school_fees": create_school_fees,
            "college_fees": create_college_fees,
            "travel_expenses": create_travel_expenses,
            "groceries_utilities":
                create_groceries_utilities,
            "other_monthly_expenses":
                create_other_monthly_expenses,
            "existing_loans": create_existing_loans,
            "current_emi_amount":
                create_current_emi_amount,
            "credit_score": create_credit_score,
            "bank_balance": create_bank_balance,
            "emergency_fund": create_emergency_fund,
            "emi_scenario": create_emi_scenario,
            "requested_amount": create_requested_amount,
            "requested_tenure": create_requested_tenure,
            "emi_eligibility":
                create_emi_eligibility,
            "max_monthly_emi":
                create_max_monthly_emi
        }

        try:

            with engine.begin() as connection:

                connection.execute(
                    insert_query,
                    insert_values
                )

            st.success(
                "✅ Financial record added successfully."
            )

            st.rerun()

        except Exception as e:

            st.error(
                "Unable to add financial record."
            )

            st.exception(e)

    st.divider()

    # =====================================================
    # UPDATE - EDIT RECORD
    # =====================================================

    st.subheader("✏️ Update Financial Record")

    try:

        id_query = """
        SELECT record_id
        FROM financial_records
        ORDER BY record_id
        """

        id_df = pd.read_sql(
            id_query,
            engine
        )

        if len(id_df) == 0:

            st.info(
                "No records available for updating."
            )

        else:

            selected_update_id = st.selectbox(
                "Select Record ID",
                id_df["record_id"].tolist(),
                key="selected_update_id"
            )

            selected_query = text("""
                SELECT *
                FROM financial_records
                WHERE record_id = :record_id
            """)

            selected_df = pd.read_sql(
                selected_query,
                engine,
                params={
                    "record_id": selected_update_id
                }
            )

            record = selected_df.iloc[0]

            with st.form("update_record_form"):

                st.markdown(
                    f"### Editing Record ID: "
                    f"{selected_update_id}"
                )

                # -------------------------------------------------
                # Personal Information
                # -------------------------------------------------

                st.markdown(
                    "### 👤 Personal Information"
                )

                col1, col2, col3 = st.columns(3)

                with col1:

                    update_age = st.number_input(
                        "Age",
                        min_value=18.0,
                        max_value=100.0,
                        value=float(record["age"]),
                        key="update_age"
                    )

                    update_gender = st.selectbox(
                        "Gender",
                        ["Male", "Female"],
                        index=(
                            0
                            if record["gender"] == "Male"
                            else 1
                        ),
                        key="update_gender"
                    )

                    update_marital_status = st.selectbox(
                        "Marital Status",
                        ["Single", "Married"],
                        index=(
                            0
                            if record["marital_status"] == "Single"
                            else 1
                        ),
                        key="update_marital_status"
                    )

                with col2:

                    education_options = [
                        "High School",
                        "Graduate",
                        "Post Graduate",
                        "Professional"
                    ]

                    update_education = st.selectbox(
                        "Education",
                        education_options,
                        index=(
                            education_options.index(
                                record["education"]
                            )
                            if record["education"]
                            in education_options
                            else 0
                        ),
                        key="update_education"
                    )

                    update_family_size = st.number_input(
                        "Family Size",
                        min_value=1,
                        max_value=20,
                        value=int(record["family_size"]),
                        key="update_family_size"
                    )

                    update_dependents = st.number_input(
                        "Dependents",
                        min_value=0,
                        max_value=20,
                        value=int(record["dependents"]),
                        key="update_dependents"
                    )

                with col3:

                    house_options = [
                        "Rented",
                        "Own",
                        "Family"
                    ]

                    update_house_type = st.selectbox(
                        "House Type",
                        house_options,
                        index=(
                            house_options.index(
                                record["house_type"]
                            )
                            if record["house_type"]
                            in house_options
                            else 0
                        ),
                        key="update_house_type"
                    )

                    update_monthly_rent = st.number_input(
                        "Monthly Rent (₹)",
                        min_value=0.0,
                        value=float(
                            record["monthly_rent"]
                        ),
                        step=500.0,
                        key="update_monthly_rent"
                    )

                # -------------------------------------------------
                # Employment
                # -------------------------------------------------

                st.markdown(
                    "### 💼 Employment Information"
                )

                col1, col2, col3, col4 = st.columns(4)

                with col1:

                    update_salary = st.number_input(
                        "Monthly Salary (₹)",
                        min_value=0.0,
                        value=float(
                            record["monthly_salary"]
                        ),
                        step=1000.0,
                        key="update_salary"
                    )

                with col2:

                    employment_options = [
                        "Private",
                        "Government",
                        "Self-employed"
                    ]

                    update_employment_type = st.selectbox(
                        "Employment Type",
                        employment_options,
                        index=(
                            employment_options.index(
                                record["employment_type"]
                            )
                            if record["employment_type"]
                            in employment_options
                            else 0
                        ),
                        key="update_employment_type"
                    )

                with col3:

                    update_years_employment = st.number_input(
                        "Years of Employment",
                        min_value=0.0,
                        max_value=50.0,
                        value=float(
                            record["years_of_employment"]
                        ),
                        step=0.5,
                        key="update_years_employment"
                    )

                with col4:

                    company_options = [
                        "Large Indian",
                        "MNC",
                        "Mid-size",
                        "Startup",
                        "Small"
                    ]

                    update_company_type = st.selectbox(
                        "Company Type",
                        company_options,
                        index=(
                            company_options.index(
                                record["company_type"]
                            )
                            if record["company_type"]
                            in company_options
                            else 0
                        ),
                        key="update_company_type"
                    )

                # -------------------------------------------------
                # Expenses
                # -------------------------------------------------

                st.markdown(
                    "### 💰 Monthly Expenses"
                )

                col1, col2, col3, col4 = st.columns(4)

                with col1:

                    update_school_fees = st.number_input(
                        "School Fees (₹)",
                        min_value=0.0,
                        value=float(
                            record["school_fees"]
                        ),
                        step=500.0,
                        key="update_school_fees"
                    )

                with col2:

                    update_college_fees = st.number_input(
                        "College Fees (₹)",
                        min_value=0.0,
                        value=float(
                            record["college_fees"]
                        ),
                        step=500.0,
                        key="update_college_fees"
                    )

                with col3:

                    update_travel = st.number_input(
                        "Travel Expenses (₹)",
                        min_value=0.0,
                        value=float(
                            record["travel_expenses"]
                        ),
                        step=500.0,
                        key="update_travel"
                    )

                with col4:

                    update_groceries = st.number_input(
                        "Groceries & Utilities (₹)",
                        min_value=0.0,
                        value=float(
                            record["groceries_utilities"]
                        ),
                        step=500.0,
                        key="update_groceries"
                    )

                update_other_expenses = st.number_input(
                    "Other Monthly Expenses (₹)",
                    min_value=0.0,
                    value=float(
                        record["other_monthly_expenses"]
                    ),
                    step=500.0,
                    key="update_other_expenses"
                )

                # -------------------------------------------------
                # Existing Obligations
                # -------------------------------------------------

                st.markdown(
                    "### 💳 Existing Financial Obligations"
                )

                col1, col2, col3 = st.columns(3)

                with col1:

                    loan_options = ["No", "Yes"]

                    update_existing_loans = st.selectbox(
                        "Existing Loans",
                        loan_options,
                        index=(
                            loan_options.index(
                                record["existing_loans"]
                            )
                            if record["existing_loans"]
                            in loan_options
                            else 0
                        ),
                        key="update_existing_loans"
                    )

                with col2:

                    update_current_emi = st.number_input(
                        "Current EMI Amount (₹)",
                        min_value=0.0,
                        value=float(
                            record["current_emi_amount"]
                        ),
                        step=500.0,
                        key="update_current_emi"
                    )

                with col3:

                    update_credit_score = st.number_input(
                        "Credit Score",
                        min_value=0.0,
                        max_value=1200.0,
                        value=float(
                            record["credit_score"]
                        ),
                        step=1.0,
                        key="update_credit_score"
                    )

                # -------------------------------------------------
                # Reserves
                # -------------------------------------------------

                st.markdown(
                    "### 🏦 Financial Reserves"
                )

                col1, col2 = st.columns(2)

                with col1:

                    update_bank_balance = st.number_input(
                        "Bank Balance (₹)",
                        min_value=0.0,
                        value=float(
                            record["bank_balance"]
                        ),
                        step=1000.0,
                        key="update_bank_balance"
                    )

                with col2:

                    update_emergency_fund = st.number_input(
                        "Emergency Fund (₹)",
                        min_value=0.0,
                        value=float(
                            record["emergency_fund"]
                        ),
                        step=1000.0,
                        key="update_emergency_fund"
                    )

                # -------------------------------------------------
                # EMI Request
                # -------------------------------------------------

                st.markdown(
                    "### 🛒 EMI Request"
                )

                col1, col2, col3 = st.columns(3)

                with col1:

                    scenario_options = [
                        "Home Appliances EMI",
                        "Personal Loan EMI",
                        "E-commerce Shopping EMI",
                        "Education EMI",
                        "Vehicle EMI"
                    ]

                    update_scenario = st.selectbox(
                        "EMI Scenario",
                        scenario_options,
                        index=(
                            scenario_options.index(
                                record["emi_scenario"]
                            )
                            if record["emi_scenario"]
                            in scenario_options
                            else 0
                        ),
                        key="update_scenario"
                    )

                with col2:

                    update_requested_amount = st.number_input(
                        "Requested Amount (₹)",
                        min_value=0.0,
                        value=float(
                            record["requested_amount"]
                        ),
                        step=5000.0,
                        key="update_requested_amount"
                    )

                with col3:

                    update_requested_tenure = st.number_input(
                        "Requested Tenure (Months)",
                        min_value=1,
                        max_value=120,
                        value=int(
                            record["requested_tenure"]
                        ),
                        key="update_requested_tenure"
                    )

                # -------------------------------------------------
                # Model Output
                # -------------------------------------------------

                st.markdown(
                    "### 🤖 Model Output"
                )

                col1, col2 = st.columns(2)

                with col1:

                    eligibility_options = [
                        "Eligible",
                        "High_Risk",
                        "Not_Eligible"
                    ]

                    update_eligibility = st.selectbox(
                        "EMI Eligibility",
                        eligibility_options,
                        index=(
                            eligibility_options.index(
                                record["emi_eligibility"]
                            )
                            if record["emi_eligibility"]
                            in eligibility_options
                            else 0
                        ),
                        key="update_eligibility"
                    )

                with col2:

                    update_max_emi = st.number_input(
                        "Maximum Monthly EMI (₹)",
                        min_value=0.0,
                        value=float(
                            record["max_monthly_emi"]
                        ),
                        step=500.0,
                        key="update_max_emi"
                    )

                update_button = st.form_submit_button(
                    "💾 Update Record",
                    use_container_width=True
                )

            # ---------------------------------------------
            # Execute Update
            # ---------------------------------------------

            if update_button:

                update_query = text("""
                    UPDATE financial_records
                    SET
                        age = :age,
                        gender = :gender,
                        marital_status = :marital_status,
                        education = :education,
                        monthly_salary = :monthly_salary,
                        employment_type = :employment_type,
                        years_of_employment =
                            :years_of_employment,
                        company_type = :company_type,
                        house_type = :house_type,
                        monthly_rent = :monthly_rent,
                        family_size = :family_size,
                        dependents = :dependents,
                        school_fees = :school_fees,
                        college_fees = :college_fees,
                        travel_expenses = :travel_expenses,
                        groceries_utilities =
                            :groceries_utilities,
                        other_monthly_expenses =
                            :other_monthly_expenses,
                        existing_loans = :existing_loans,
                        current_emi_amount =
                            :current_emi_amount,
                        credit_score = :credit_score,
                        bank_balance = :bank_balance,
                        emergency_fund = :emergency_fund,
                        emi_scenario = :emi_scenario,
                        requested_amount =
                            :requested_amount,
                        requested_tenure =
                            :requested_tenure,
                        emi_eligibility =
                            :emi_eligibility,
                        max_monthly_emi =
                            :max_monthly_emi
                    WHERE record_id = :record_id
                """)

                update_values = {
                    "record_id": selected_update_id,
                    "age": update_age,
                    "gender": update_gender,
                    "marital_status":
                        update_marital_status,
                    "education": update_education,
                    "monthly_salary": update_salary,
                    "employment_type":
                        update_employment_type,
                    "years_of_employment":
                        update_years_employment,
                    "company_type":
                        update_company_type,
                    "house_type":
                        update_house_type,
                    "monthly_rent":
                        update_monthly_rent,
                    "family_size":
                        update_family_size,
                    "dependents":
                        update_dependents,
                    "school_fees":
                        update_school_fees,
                    "college_fees":
                        update_college_fees,
                    "travel_expenses":
                        update_travel,
                    "groceries_utilities":
                        update_groceries,
                    "other_monthly_expenses":
                        update_other_expenses,
                    "existing_loans":
                        update_existing_loans,
                    "current_emi_amount":
                        update_current_emi,
                    "credit_score":
                        update_credit_score,
                    "bank_balance":
                        update_bank_balance,
                    "emergency_fund":
                        update_emergency_fund,
                    "emi_scenario":
                        update_scenario,
                    "requested_amount":
                        update_requested_amount,
                    "requested_tenure":
                        update_requested_tenure,
                    "emi_eligibility":
                        update_eligibility,
                    "max_monthly_emi":
                        update_max_emi
                }

                try:

                    with engine.begin() as connection:

                        connection.execute(
                            update_query,
                            update_values
                        )

                    st.success(
                        f"✅ Record {selected_update_id} "
                        "updated successfully."
                    )

                    st.rerun()

                except Exception as e:

                    st.error(
                        "Unable to update the record."
                    )

                    st.exception(e)

    except Exception as e:

        st.error(
            "Unable to load records for updating."
        )

        st.exception(e)

    st.divider()

    # =====================================================
    # DELETE - DELETE RECORD
    # =====================================================

    st.subheader("🗑️ Delete Financial Record")

    try:

        delete_id_query = """
        SELECT record_id
        FROM financial_records
        ORDER BY record_id
        """

        delete_id_df = pd.read_sql(
            delete_id_query,
            engine
        )

        if len(delete_id_df) == 0:

            st.info(
                "No records available for deletion."
            )

        else:

            selected_delete_id = st.selectbox(
                "Select Record ID to Delete",
                delete_id_df["record_id"].tolist(),
                key="selected_delete_id"
            )

            st.warning(
                f"You are about to delete Record "
                f"ID **{selected_delete_id}**."
            )

            confirm_delete = st.checkbox(
                "I confirm that I want to delete this record.",
                key="confirm_delete"
            )

            delete_button = st.button(
                "🗑️ Delete Record",
                disabled=not confirm_delete,
                use_container_width=True
            )

            if delete_button:

                delete_query = text("""
                    DELETE FROM financial_records
                    WHERE record_id = :record_id
                """)

                try:

                    with engine.begin() as connection:

                        connection.execute(
                            delete_query,
                            {
                                "record_id":
                                    selected_delete_id
                            }
                        )

                    st.success(
                        f"✅ Record {selected_delete_id} "
                        "deleted successfully."
                    )

                    st.rerun()

                except Exception as e:

                    st.error(
                        "Unable to delete the record."
                    )

                    st.exception(e)

    except Exception as e:

        st.error(
            "Unable to load records for deletion."
        )

        st.exception(e)