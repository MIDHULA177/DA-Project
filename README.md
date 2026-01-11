HR Analytics – Employee Attrition Prediction

This project is an end-to-end HR Analytics solution developed to analyze employee data, identify the main causes of employee resignation, and predict future attrition using machine learning techniques. The project also provides actionable business insights to help organizations improve employee retention.

Project Objective
The main objective of this project is to use data analytics and machine learning to understand employee attrition patterns and predict whether an employee is likely to leave the organization. The project focuses on identifying key factors influencing attrition and suggesting effective retention strategies.

Tools and Technologies
Python
Pandas and NumPy for data processing
Matplotlib and Seaborn for exploratory data analysis
Scikit-learn for machine learning models
SHAP for model explainability
Plotly Dash for interactive dashboard (Power BI alternative)

Exploratory Data Analysis (EDA)
Exploratory data analysis was performed to understand attrition trends and employee behavior. The analysis includes department-wise attrition, salary impact on attrition, job satisfaction trends, work-life balance, overtime impact, promotion history, and experience-based analysis.

Machine Learning Models
The following classification models were built and evaluated:
Logistic Regression
Decision Tree
Random Forest

The Random Forest model performed best with an accuracy of approximately 88%.

Model Explainability
SHAP (SHapley Additive exPlanations) was used to interpret model predictions. SHAP analysis helped identify the most important features influencing employee attrition and provided explanations for individual predictions.

Dashboard
An interactive dashboard was developed using Plotly Dash to visualize attrition trends and key risk factors. The dashboard serves as an alternative to Power BI and allows dynamic exploration of data.

Key Results
Overall attrition rate: 16.1%
Top risk factors: low income, overtime, poor job satisfaction, work-life imbalance
Best model: Random Forest (88% accuracy)
Expected impact: 15–25% reduction in attrition with an estimated ROI of 200–300%
