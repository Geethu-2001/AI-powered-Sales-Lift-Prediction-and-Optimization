# AI-powered-Sales-Lift-Prediction-and-Optimization

Project: AI-powered Sales Lift Prediction and Optimization
This project focuses on predicting and optimizing sales lift for a company’s promotional campaigns using machine learning techniques. The goal is to build a model that helps businesses understand how different promotional strategies (like discounts, ad spending, or promotion types) impact sales growth, thereby helping them optimize their promotional efforts to achieve the maximum return on investment (ROI).

Here’s a detailed explanation of each component of the project:

1. Dataset Overview:
The dataset used in this project includes various details about past promotions, including:

Promotion_Type: Different types of promotions (e.g., BOGO, Coupon, Flash Sale, Discount).
Season: The time period when the promotion took place (e.g., Holiday, Off-Season, Peak).
Sales_Lift: The change in sales due to the promotion.
Base_Price, Discount, Ad_Spend, Previous_Sales: Additional features that contribute to the prediction model, such as the base price of the product, the discount percentage, the ad spend for the campaign, and the sales performance of the product prior to the promotion.
2. Data Preprocessing:
Handling Categorical Data: The categorical variables like Promotion_Type and Season are converted into numerical format using one-hot encoding. This step creates binary columns representing each possible category (e.g., a column for each type of promotion like "Promotion_Type_BOGO", "Promotion_Type_Coupon", etc.).

Feature Selection: After processing the data, the relevant features (i.e., those directly related to the sales lift) are selected for the model. The target variable is Sales_Lift, while other columns like Base_Price, Discount, Ad_Spend, and one-hot encoded columns for Promotion_Type and Season are used as input features.

Handling Missing Data: Any missing or non-numeric data points are replaced with zeros or a sensible default value.

3. Model Development:
Gradient Boosting Regressor (GBR): The machine learning model used to predict sales lift is the Gradient Boosting Regressor. This is a powerful ensemble learning method that builds multiple decision trees and combines them to improve prediction accuracy.

Training and Testing Split: The dataset is divided into a training set (80%) and a test set (20%) using train_test_split. The training set is used to train the model, while the test set is used to evaluate its performance.

Model Training: The Gradient Boosting model is trained on the training data, and the performance is evaluated using the Mean Absolute Error (MAE), which measures the average magnitude of errors between predicted and actual sales lift values.

4. Key Features and Optimization Tools:
4.1. Sales Lift Prediction:
The model predicts the expected sales lift based on a variety of promotional strategies. The user can input different values, such as:

Base Price: The price at which the product is sold.
Discount: The discount applied during the promotion.
Ad Spend: The amount spent on advertising the promotion.
Previous Sales: Sales performance before the promotion.
The model then predicts the Sales Lift (i.e., the change in sales) based on these inputs. Users can see how their promotional strategy would affect sales lift.

4.2. Sensitivity Analysis:
This feature allows users to perform sensitivity analysis by adjusting specific variables (like Discount and Ad Spend) to understand how sensitive the predicted sales lift is to changes in those values. For instance, if a small increase in ad spend leads to a significant increase in sales, the business might choose to invest more in advertising.

4.3. A/B Testing Simulation:
This simulation allows users to compare two different promotional strategies (e.g., BOGO vs Discount) and predict which one would likely yield higher sales lift. The user can select two types of promotions, and the model predicts the potential sales lift for each, allowing businesses to choose the most effective strategy.

4.4. Feature Importance and Correlation Heatmap:
The Feature Importance plot helps identify which features are contributing the most to the model's predictions. For example, it could show that Ad Spend or Discount has the highest importance in predicting sales lift. Additionally, a correlation heatmap is generated to visualize the relationships between various numerical features in the dataset (like Base Price, Discount, Ad Spend, etc.). This helps to understand how features are related to each other, which can assist in refining the model.

4.5. Promotion ROI Calculator:
This tool allows users to calculate the Return on Investment (ROI) of a promotional campaign by considering:

Profit Margin per Unit (calculated from the Base Price and COGS).
Total Profit (based on predicted sales lift).
Total Investment (which includes Ad Spend and Promotion Cost).
The ROI is calculated to help users assess the financial effectiveness of their promotional strategies.

5. User Interface:
The project has an interactive Streamlit app that provides a clean and user-friendly interface. Key components include:

A file uploader to input the dataset.
A sidebar for users to input values for promotion optimization and ROI calculation.
Dynamic graphs and tables that show results like predicted sales lift, feature importance, correlation heatmap, and A/B testing simulation.
6. Key Outputs:
Predicted Sales Lift: The model’s prediction of how much sales will increase with the given promotional strategy.
Feature Importance Visualization: A bar plot showing which features have the most impact on sales lift predictions.
Correlation Heatmap: A heatmap showing the correlation between numerical features in the dataset.
Promotion Optimization: Insights into the most effective promotional strategies based on user inputs.
ROI Calculation: A financial breakdown of the profit margins, total profit, and return on investment for the selected promotion.
Conclusion:
The project provides a comprehensive tool for businesses to optimize their promotional strategies, predict sales lift, and assess the financial return on investment of different promotions. The combination of machine learning for sales prediction, sensitivity analysis, A/B testing, and ROI calculations allows businesses to make informed, data-driven decisions about their marketing campaigns.









You’ve hit the Free plan limit for GPT-4o.
Responses will use another model until your limit resets 
