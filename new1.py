import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

# Streamlit App Title
st.title("AI-Powered Sales Lift Prediction & Optimization")

# File Upload
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file, encoding='utf-8')
    df.columns = df.columns.str.strip().str.replace('[^A-Za-z0-9_]+', '_', regex=True)
    st.write("Dataset Preview:", df.head())

    required_columns = ['Promotion_Type', 'Season', 'Sales_Lift']
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        st.error(f"The following columns are missing in the dataset: {', '.join(missing_columns)}")
    else:
        # Handle categorical data
        df = pd.get_dummies(df, columns=["Promotion_Type", "Season"])
        X = df.drop(columns=["Retailer_ID", "Product_ID", "Sales_Lift"])
        y = df["Sales_Lift"]

        # Convert to numeric
        X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
        y = y.apply(pd.to_numeric, errors='coerce').fillna(0)

        feature_names = X.columns.tolist()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        st.write(f"Model Performance - Mean Absolute Error: {mae:.2f}")

        # Feature Importance
        st.subheader("Feature Importances")
        feature_importances = model.feature_importances_
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': feature_importances
        }).sort_values(by='Importance', ascending=False)

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(x='Importance', y='Feature', data=importance_df, ax=ax, color="skyblue")
        ax.set_title("Feature Importances")
        st.pyplot(fig)

        # Updated Correlation Heatmap
        st.subheader("Feature Correlation Heatmap")
        numeric_df = df.select_dtypes(include=[np.number])
        promotion_type_columns = [col for col in df.columns if 'Promotion_Type_' in col]
        numeric_df = pd.concat([numeric_df, df[promotion_type_columns]], axis=1)

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

        # Sidebar - Promotion Optimization
        st.sidebar.header("Optimize Your Promotion")
        base_price = st.sidebar.number_input("Base Price ($)", min_value=0, value=500, key="base_price")
        discount = st.sidebar.slider("Discount (%)", min_value=0, max_value=50, value=10, key="discount")
        ad_spend = st.sidebar.number_input("Ad Spend ($)", min_value=0, value=5000, key="ad_spend")
        previous_sales = st.sidebar.number_input("Previous Sales", min_value=0, value=1000, key="previous_sales")

        promo_types = ["BOGO", "Coupon", "Flash Sale", "Discount"]
        selected_promo = st.sidebar.selectbox("Promotion Type", promo_types, key="promo_type")

        seasons = ["Holiday", "Off-Season", "Peak"]
        selected_season = st.sidebar.selectbox("Season", seasons, key="season")

        input_data = {
            "Base_Price": [base_price],
            "Discount": [discount],
            "Ad_Spend": [ad_spend],
            "Previous_Sales": [previous_sales],
        }

        # One-hot encode user inputs
        for col in feature_names:
            if "Promotion_Type_" in col:
                input_data[col] = [1 if selected_promo in col else 0]
            elif "Season_" in col:
                input_data[col] = [1 if selected_season in col else 0]
            elif col not in input_data:
                input_data[col] = [0]

        test_promo = pd.DataFrame(input_data)[feature_names]
        test_promo = test_promo.apply(pd.to_numeric, errors='coerce')

        predicted_sales_lift = model.predict(test_promo)[0]
        st.sidebar.subheader(f"Predicted Sales Lift: {predicted_sales_lift:.2f}")

        if predicted_sales_lift < 10:
            st.warning("Consider increasing your discount or ad spend for better results!")
        elif predicted_sales_lift > 20:
            st.success("Great! Your promotional strategy is likely to yield high sales lift!")

        # Promotion ROI Calculator
        st.sidebar.header("Promotion ROI Calculator")
        cogs = st.sidebar.number_input("Cost of Goods Sold (COGS) per unit ($)", min_value=0, value=300, key="cogs")
        promotion_cost = st.sidebar.number_input("Promotion Cost ($)", min_value=0, value=1000, key="promo_cost")

        profit_margin = (base_price * (1 - discount / 100)) - cogs
        total_profit = profit_margin * predicted_sales_lift
        total_investment = ad_spend + promotion_cost
        roi = ((total_profit - total_investment) / total_investment) * 100

        st.sidebar.write(f"Profit Margin per Unit: ${profit_margin:.2f}")
        st.sidebar.write(f"Total Profit: ${total_profit:.2f}")
        st.sidebar.write(f"Return on Investment (ROI): {roi:.2f}%")

        # Sensitivity Analysis (User Inputs Instead of Ranges)
        st.subheader("Sensitivity Analysis")
        sens_discount = st.number_input("Enter Discount (%)", min_value=0, max_value=50, value=10, key="sens_discount")
        sens_ad_spend = st.number_input("Enter Ad Spend ($)", min_value=0, value=5000, key="sens_ad_spend")

        input_data["Discount"] = [sens_discount]
        input_data["Ad_Spend"] = [sens_ad_spend]
        test_promo = pd.DataFrame(input_data)[feature_names]
        sensitivity_sales_lift = model.predict(test_promo)[0]

        st.write(f"Predicted Sales Lift for chosen values: {sensitivity_sales_lift:.2f}")

        # A/B Testing Simulation (No Graph)
        st.subheader("A/B Testing Simulation")
        promo_type_a = st.selectbox("Promotion Type A", promo_types, key="promo_a")
        promo_type_b = st.selectbox("Promotion Type B", promo_types, key="promo_b")

        ab_results = []
        for promo in [promo_type_a, promo_type_b]:
            scenario_data = input_data.copy()
            for col in feature_names:
                if "Promotion_Type_" in col:
                    scenario_data[col] = [1 if promo in col else 0]
            scenario_df = pd.DataFrame(scenario_data)[feature_names]
            scenario_lift = model.predict(scenario_df)[0]
            ab_results.append((promo, scenario_lift))

        ab_df = pd.DataFrame(ab_results, columns=["Promotion Type", "Predicted Sales Lift"])
        st.write(ab_df)

        # **Adding Constant Feature Visualization**
        constant_data = {
            "Base_Price": base_price,
            "Discount": discount,
            "Ad_Spend": ad_spend,
            "Previous_Sales": previous_sales
        }

        # Show constant feature values in a table
        st.write("Constant Features (Same for both Promotion Types):")
        constant_df = pd.DataFrame(list(constant_data.items()), columns=["Feature", "Value"])
        st.write(constant_df)

        # # Creating a plot to show constant values (bar plot)
        # fig, ax = plt.subplots(figsize=(6, 4))
        # ax.barh(constant_data.keys(), constant_data.values(), color="lightblue")
        # ax.set_xlabel("Feature Values")
        # ax.set_title("Constant Features (Same for both Promotion Types)")
        # st.pyplot(fig)

        # Downloadable Report
        st.subheader("Download Report")
        report = f"""
        Sales Lift Prediction Report:
        - Predicted Sales Lift: {predicted_sales_lift:.2f}
        - Profit Margin per Unit: ${profit_margin:.2f}
        - Total Profit: ${total_profit:.2f}
        - Return on Investment (ROI): {roi:.2f}%
        """
        st.download_button("Download Report", report, file_name="sales_lift_report.txt")
