"""
HR Analytics - Employee Attrition Prediction
Complete implementation with EDA, ML models, visualizations, and SHAP analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder
import shap
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class HRAnalytics:
    def __init__(self, data_path):
        """Initialize the HR Analytics class"""
        self.data_path = data_path
        self.df = None
        self.df_processed = None
        self.models = {}
        self.results = {}
        
    def load_data(self):
        """Load and display basic information about the dataset"""
        print("=" * 60)
        print("HR ANALYTICS - EMPLOYEE ATTRITION PREDICTION")
        print("=" * 60)
        
        self.df = pd.read_csv(self.data_path)
        print(f"\nDataset loaded successfully!")
        print(f"Shape: {self.df.shape}")
        print(f"Columns: {len(self.df.columns)}")
        
        # Basic info
        print("\n" + "="*50)
        print("DATASET OVERVIEW")
        print("="*50)
        print(f"Total Employees: {len(self.df)}")
        print(f"Attrition Rate: {(self.df['Attrition'] == 'Yes').mean():.2%}")
        print(f"Missing Values: {self.df.isnull().sum().sum()}")
        
        return self.df
    
    def exploratory_data_analysis(self):
        """Perform comprehensive EDA"""
        print("\n" + "="*50)
        print("EXPLORATORY DATA ANALYSIS")
        print("="*50)
        
        # Create figure for EDA plots
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Attrition Distribution
        plt.subplot(3, 4, 1)
        attrition_counts = self.df['Attrition'].value_counts()
        plt.pie(attrition_counts.values, labels=attrition_counts.index, autopct='%1.1f%%', startangle=90)
        plt.title('Overall Attrition Distribution', fontsize=12, fontweight='bold')
        
        # 2. Department-wise Attrition
        plt.subplot(3, 4, 2)
        dept_attrition = pd.crosstab(self.df['Department'], self.df['Attrition'], normalize='index') * 100
        dept_attrition['Yes'].plot(kind='bar', color='coral')
        plt.title('Attrition Rate by Department', fontsize=12, fontweight='bold')
        plt.ylabel('Attrition Rate (%)')
        plt.xticks(rotation=45)
        
        # 3. Monthly Income Distribution
        plt.subplot(3, 4, 3)
        sns.boxplot(data=self.df, x='Attrition', y='MonthlyIncome')
        plt.title('Monthly Income vs Attrition', fontsize=12, fontweight='bold')
        
        # 4. Age Distribution
        plt.subplot(3, 4, 4)
        sns.histplot(data=self.df, x='Age', hue='Attrition', bins=20, alpha=0.7)
        plt.title('Age Distribution by Attrition', fontsize=12, fontweight='bold')
        
        # 5. Years at Company
        plt.subplot(3, 4, 5)
        sns.boxplot(data=self.df, x='Attrition', y='YearsAtCompany')
        plt.title('Years at Company vs Attrition', fontsize=12, fontweight='bold')
        
        # 6. Job Role Attrition
        plt.subplot(3, 4, 6)
        job_attrition = pd.crosstab(self.df['JobRole'], self.df['Attrition'], normalize='index') * 100
        job_attrition['Yes'].sort_values(ascending=False).head(8).plot(kind='bar', color='lightblue')
        plt.title('Top Job Roles by Attrition Rate', fontsize=12, fontweight='bold')
        plt.ylabel('Attrition Rate (%)')
        plt.xticks(rotation=45)
        
        # 7. Overtime vs Attrition
        plt.subplot(3, 4, 7)
        overtime_attrition = pd.crosstab(self.df['OverTime'], self.df['Attrition'], normalize='index') * 100
        overtime_attrition['Yes'].plot(kind='bar', color='orange')
        plt.title('Overtime vs Attrition Rate', fontsize=12, fontweight='bold')
        plt.ylabel('Attrition Rate (%)')
        
        # 8. Distance from Home
        plt.subplot(3, 4, 8)
        sns.boxplot(data=self.df, x='Attrition', y='DistanceFromHome')
        plt.title('Distance from Home vs Attrition', fontsize=12, fontweight='bold')
        
        # 9. Job Satisfaction
        plt.subplot(3, 4, 9)
        satisfaction_attrition = pd.crosstab(self.df['JobSatisfaction'], self.df['Attrition'], normalize='index') * 100
        satisfaction_attrition['Yes'].plot(kind='bar', color='green')
        plt.title('Job Satisfaction vs Attrition', fontsize=12, fontweight='bold')
        plt.ylabel('Attrition Rate (%)')
        
        # 10. Work Life Balance
        plt.subplot(3, 4, 10)
        balance_attrition = pd.crosstab(self.df['WorkLifeBalance'], self.df['Attrition'], normalize='index') * 100
        balance_attrition['Yes'].plot(kind='bar', color='purple')
        plt.title('Work Life Balance vs Attrition', fontsize=12, fontweight='bold')
        plt.ylabel('Attrition Rate (%)')
        
        # 11. Years Since Last Promotion
        plt.subplot(3, 4, 11)
        sns.boxplot(data=self.df, x='Attrition', y='YearsSinceLastPromotion')
        plt.title('Years Since Last Promotion vs Attrition', fontsize=12, fontweight='bold')
        
        # 12. Correlation Heatmap (top features)
        plt.subplot(3, 4, 12)
        # Select numeric columns for correlation
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        corr_data = self.df[numeric_cols].copy()
        corr_data['Attrition_Binary'] = (self.df['Attrition'] == 'Yes').astype(int)
        
        # Get top correlated features with attrition
        correlations = corr_data.corr()['Attrition_Binary'].abs().sort_values(ascending=False)
        top_features = correlations.head(10).index
        
        sns.heatmap(corr_data[top_features].corr(), annot=True, cmap='coolwarm', center=0, fmt='.2f')
        plt.title('Correlation Heatmap (Top Features)', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('hr_analytics_eda.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print key insights
        print("\nKEY INSIGHTS FROM EDA:")
        print("-" * 30)
        
        # Department insights
        dept_rates = pd.crosstab(self.df['Department'], self.df['Attrition'], normalize='index')['Yes'] * 100
        print(f"• Highest attrition department: {dept_rates.idxmax()} ({dept_rates.max():.1f}%)")
        
        # Income insights
        avg_income_stay = self.df[self.df['Attrition'] == 'No']['MonthlyIncome'].mean()
        avg_income_leave = self.df[self.df['Attrition'] == 'Yes']['MonthlyIncome'].mean()
        print(f"• Average income (Stay): ${avg_income_stay:,.0f}")
        print(f"• Average income (Leave): ${avg_income_leave:,.0f}")
        
        # Overtime insights
        overtime_rate = pd.crosstab(self.df['OverTime'], self.df['Attrition'], normalize='index')['Yes'] * 100
        print(f"• Overtime attrition rate: {overtime_rate['Yes']:.1f}%")
        print(f"• No overtime attrition rate: {overtime_rate['No']:.1f}%")
        
    def preprocess_data(self):
        """Preprocess data for machine learning"""
        print("\n" + "="*50)
        print("DATA PREPROCESSING")
        print("="*50)
        
        # Create a copy for processing
        self.df_processed = self.df.copy()
        
        # Remove unnecessary columns
        columns_to_drop = ['EmployeeNumber', 'Over18', 'StandardHours', 'EmployeeCount']
        columns_to_drop = [col for col in columns_to_drop if col in self.df_processed.columns]
        self.df_processed = self.df_processed.drop(columns_to_drop, axis=1)
        
        # Convert target variable
        self.df_processed['Attrition'] = (self.df_processed['Attrition'] == 'Yes').astype(int)
        
        # Encode categorical variables
        categorical_columns = self.df_processed.select_dtypes(include=['object']).columns
        
        for col in categorical_columns:
            if col != 'Attrition':
                self.df_processed = pd.get_dummies(self.df_processed, columns=[col], prefix=col, drop_first=True)
        
        print(f"Original features: {self.df.shape[1]}")
        print(f"Processed features: {self.df_processed.shape[1]}")
        print(f"Target distribution: {self.df_processed['Attrition'].value_counts().to_dict()}")
        
        return self.df_processed
    
    def build_models(self):
        """Build and train multiple ML models"""
        print("\n" + "="*50)
        print("MODEL BUILDING & TRAINING")
        print("="*50)
        
        # Prepare features and target
        X = self.df_processed.drop('Attrition', axis=1)
        y = self.df_processed['Attrition']
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features for logistic regression
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Store data for later use
        self.X_train, self.X_test = X_train, X_test
        self.y_train, self.y_test = y_train, y_test
        self.X_train_scaled, self.X_test_scaled = X_train_scaled, X_test_scaled
        self.scaler = scaler
        self.feature_names = X.columns.tolist()
        
        # 1. Logistic Regression
        print("Training Logistic Regression...")
        lr_model = LogisticRegression(random_state=42, max_iter=1000)
        lr_model.fit(X_train_scaled, y_train)
        lr_pred = lr_model.predict(X_test_scaled)
        lr_prob = lr_model.predict_proba(X_test_scaled)[:, 1]
        
        self.models['Logistic Regression'] = lr_model
        self.results['Logistic Regression'] = {
            'predictions': lr_pred,
            'probabilities': lr_prob,
            'accuracy': accuracy_score(y_test, lr_pred),
            'auc': roc_auc_score(y_test, lr_prob),
            'confusion_matrix': confusion_matrix(y_test, lr_pred),
            'classification_report': classification_report(y_test, lr_pred)
        }
        
        # 2. Decision Tree
        print("Training Decision Tree...")
        dt_model = DecisionTreeClassifier(random_state=42, max_depth=10, min_samples_split=20)
        dt_model.fit(X_train, y_train)
        dt_pred = dt_model.predict(X_test)
        dt_prob = dt_model.predict_proba(X_test)[:, 1]
        
        self.models['Decision Tree'] = dt_model
        self.results['Decision Tree'] = {
            'predictions': dt_pred,
            'probabilities': dt_prob,
            'accuracy': accuracy_score(y_test, dt_pred),
            'auc': roc_auc_score(y_test, dt_prob),
            'confusion_matrix': confusion_matrix(y_test, dt_pred),
            'classification_report': classification_report(y_test, dt_pred)
        }
        
        # 3. Random Forest (bonus model)
        print("Training Random Forest...")
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        rf_model.fit(X_train, y_train)
        rf_pred = rf_model.predict(X_test)
        rf_prob = rf_model.predict_proba(X_test)[:, 1]
        
        self.models['Random Forest'] = rf_model
        self.results['Random Forest'] = {
            'predictions': rf_pred,
            'probabilities': rf_prob,
            'accuracy': accuracy_score(y_test, rf_pred),
            'auc': roc_auc_score(y_test, rf_prob),
            'confusion_matrix': confusion_matrix(y_test, rf_pred),
            'classification_report': classification_report(y_test, rf_pred)
        }
        
        print("All models trained successfully!")
        
    def evaluate_models(self):
        """Evaluate and compare all models"""
        print("\n" + "="*50)
        print("MODEL EVALUATION & COMPARISON")
        print("="*50)
        
        # Create evaluation plots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Model comparison metrics
        model_comparison = []
        
        for i, (model_name, results) in enumerate(self.results.items()):
            # Store metrics for comparison
            model_comparison.append({
                'Model': model_name,
                'Accuracy': results['accuracy'],
                'AUC': results['auc']
            })
            
            # Confusion Matrix
            row, col = i // 3, i % 3
            sns.heatmap(results['confusion_matrix'], annot=True, fmt='d', 
                       cmap='Blues', ax=axes[0, col])
            axes[0, col].set_title(f'{model_name}\nConfusion Matrix')
            axes[0, col].set_xlabel('Predicted')
            axes[0, col].set_ylabel('Actual')
            
            # ROC Curve
            fpr, tpr, _ = roc_curve(self.y_test, results['probabilities'])
            axes[1, col].plot(fpr, tpr, label=f'{model_name} (AUC = {results["auc"]:.3f})')
            axes[1, col].plot([0, 1], [0, 1], 'k--')
            axes[1, col].set_xlabel('False Positive Rate')
            axes[1, col].set_ylabel('True Positive Rate')
            axes[1, col].set_title(f'{model_name}\nROC Curve')
            axes[1, col].legend()
            axes[1, col].grid(True)
        
        plt.tight_layout()
        plt.savefig('model_evaluation.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print model comparison
        comparison_df = pd.DataFrame(model_comparison)
        print("\nMODEL PERFORMANCE COMPARISON:")
        print("-" * 40)
        print(comparison_df.to_string(index=False, float_format='%.3f'))
        
        # Best model
        best_model = comparison_df.loc[comparison_df['AUC'].idxmax(), 'Model']
        print(f"\nBest Model: {best_model}")
        
        # Detailed results for best model
        print(f"\nDETAILED RESULTS FOR {best_model.upper()}:")
        print("-" * 50)
        print(self.results[best_model]['classification_report'])
        
        return best_model
    
    def feature_importance_analysis(self):
        """Analyze feature importance using different methods"""
        print("\n" + "="*50)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("="*50)
        
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        
        # 1. Logistic Regression Coefficients
        lr_coef = pd.DataFrame({
            'Feature': self.feature_names,
            'Coefficient': abs(self.models['Logistic Regression'].coef_[0])
        }).sort_values('Coefficient', ascending=False).head(15)
        
        sns.barplot(data=lr_coef, y='Feature', x='Coefficient', ax=axes[0])
        axes[0].set_title('Logistic Regression\nFeature Importance')
        
        # 2. Decision Tree Feature Importance
        dt_importance = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': self.models['Decision Tree'].feature_importances_
        }).sort_values('Importance', ascending=False).head(15)
        
        sns.barplot(data=dt_importance, y='Feature', x='Importance', ax=axes[1])
        axes[1].set_title('Decision Tree\nFeature Importance')
        
        # 3. Random Forest Feature Importance
        rf_importance = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': self.models['Random Forest'].feature_importances_
        }).sort_values('Importance', ascending=False).head(15)
        
        sns.barplot(data=rf_importance, y='Feature', x='Importance', ax=axes[2])
        axes[2].set_title('Random Forest\nFeature Importance')
        
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print top features
        print("TOP 10 IMPORTANT FEATURES:")
        print("-" * 30)
        print("Random Forest:")
        for i, row in rf_importance.head(10).iterrows():
            print(f"  {row['Feature']}: {row['Importance']:.3f}")
    
    def shap_analysis(self):
        """Perform SHAP analysis for model interpretability"""
        print("\n" + "="*50)
        print("SHAP ANALYSIS FOR MODEL INTERPRETABILITY")
        print("="*50)
        
        # Use Random Forest for SHAP analysis
        model = self.models['Random Forest']
        
        # Create SHAP explainer
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(self.X_test)
        
        # If binary classification, take positive class
        if len(shap_values) == 2:
            shap_values = shap_values[1]
        
        # Create SHAP plots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Summary plot
        plt.subplot(2, 2, 1)
        shap.summary_plot(shap_values, self.X_test, feature_names=self.feature_names, 
                         max_display=15, show=False)
        plt.title('SHAP Summary Plot')
        
        # 2. Feature importance
        plt.subplot(2, 2, 2)
        shap.summary_plot(shap_values, self.X_test, feature_names=self.feature_names, 
                         plot_type="bar", max_display=15, show=False)
        plt.title('SHAP Feature Importance')
        
        # 3. Waterfall plot for first prediction
        plt.subplot(2, 2, 3)
        shap.waterfall_plot(explainer.expected_value[1], shap_values[0], 
                           self.X_test.iloc[0], feature_names=self.feature_names, 
                           max_display=10, show=False)
        plt.title('SHAP Waterfall Plot (Sample Prediction)')
        
        # 4. Force plot data (we'll create a custom visualization)
        plt.subplot(2, 2, 4)
        # Get SHAP values for top features
        feature_importance = np.abs(shap_values).mean(0)
        top_indices = np.argsort(feature_importance)[-10:]
        
        plt.barh(range(len(top_indices)), feature_importance[top_indices])
        plt.yticks(range(len(top_indices)), [self.feature_names[i] for i in top_indices])
        plt.xlabel('Mean |SHAP Value|')
        plt.title('Top 10 Features by SHAP Importance')
        
        plt.tight_layout()
        plt.savefig('shap_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print SHAP insights
        print("SHAP ANALYSIS INSIGHTS:")
        print("-" * 25)
        
        # Top features by SHAP importance
        feature_importance = np.abs(shap_values).mean(0)
        top_features = np.argsort(feature_importance)[-10:][::-1]
        
        print("Top 10 features by SHAP importance:")
        for i, idx in enumerate(top_features):
            print(f"  {i+1}. {self.feature_names[idx]}: {feature_importance[idx]:.3f}")
    
    def generate_insights_and_recommendations(self):
        """Generate business insights and recommendations"""
        print("\n" + "="*60)
        print("BUSINESS INSIGHTS & RECOMMENDATIONS")
        print("="*60)
        
        insights = []
        recommendations = []
        
        # Analyze key factors
        # 1. Income analysis
        avg_income_stay = self.df[self.df['Attrition'] == 'No']['MonthlyIncome'].mean()
        avg_income_leave = self.df[self.df['Attrition'] == 'Yes']['MonthlyIncome'].mean()
        income_diff = ((avg_income_stay - avg_income_leave) / avg_income_leave) * 100
        
        insights.append(f"Employees who stay earn {income_diff:.1f}% more on average")
        recommendations.append("Implement competitive salary reviews and market-based compensation")
        
        # 2. Overtime analysis
        overtime_attrition = pd.crosstab(self.df['OverTime'], self.df['Attrition'], normalize='index')['Yes'] * 100
        overtime_impact = overtime_attrition['Yes'] - overtime_attrition['No']
        
        insights.append(f"Overtime increases attrition risk by {overtime_impact:.1f} percentage points")
        recommendations.append("Monitor and limit excessive overtime, improve work-life balance")
        
        # 3. Department analysis
        dept_attrition = pd.crosstab(self.df['Department'], self.df['Attrition'], normalize='index')['Yes'] * 100
        highest_dept = dept_attrition.idxmax()
        
        insights.append(f"{highest_dept} department has highest attrition rate ({dept_attrition.max():.1f}%)")
        recommendations.append(f"Focus retention efforts on {highest_dept} department")
        
        # 4. Job satisfaction analysis
        low_satisfaction = self.df[self.df['JobSatisfaction'] <= 2]['Attrition'].value_counts(normalize=True)['Yes'] * 100
        high_satisfaction = self.df[self.df['JobSatisfaction'] >= 3]['Attrition'].value_counts(normalize=True)['Yes'] * 100
        
        insights.append(f"Low job satisfaction leads to {low_satisfaction:.1f}% attrition vs {high_satisfaction:.1f}% for high satisfaction")
        recommendations.append("Implement regular employee satisfaction surveys and improvement programs")
        
        # 5. Years at company analysis
        avg_years_stay = self.df[self.df['Attrition'] == 'No']['YearsAtCompany'].mean()
        avg_years_leave = self.df[self.df['Attrition'] == 'Yes']['YearsAtCompany'].mean()
        
        insights.append(f"Employees who leave have {avg_years_stay - avg_years_leave:.1f} fewer years at company on average")
        recommendations.append("Develop early career engagement and mentorship programs")
        
        # Print insights and recommendations
        print("\nKEY INSIGHTS:")
        print("-" * 15)
        for i, insight in enumerate(insights, 1):
            print(f"{i}. {insight}")
        
        print("\nRECOMMENDATIONS:")
        print("-" * 18)
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec}")
        
        # Additional strategic recommendations
        print("\nSTRATEGIC RECOMMENDATIONS:")
        print("-" * 28)
        strategic_recs = [
            "Implement predictive analytics dashboard for early attrition warning",
            "Create personalized retention strategies based on employee profiles",
            "Establish regular career development and promotion pathways",
            "Improve manager training on employee engagement and retention",
            "Develop flexible work arrangements to improve work-life balance"
        ]
        
        for i, rec in enumerate(strategic_recs, 1):
            print(f"{i}. {rec}")
        
        return insights, recommendations
    
    def save_results(self):
        """Save all results and generate summary report"""
        print("\n" + "="*50)
        print("SAVING RESULTS & GENERATING REPORT")
        print("="*50)
        
        # Create results summary
        summary = {
            'Dataset Info': {
                'Total Employees': len(self.df),
                'Attrition Rate': f"{(self.df['Attrition'] == 'Yes').mean():.2%}",
                'Features': self.df.shape[1]
            },
            'Model Performance': {}
        }
        
        for model_name, results in self.results.items():
            summary['Model Performance'][model_name] = {
                'Accuracy': f"{results['accuracy']:.3f}",
                'AUC Score': f"{results['auc']:.3f}"
            }
        
        # Save model results to CSV
        results_df = pd.DataFrame([
            {
                'Model': name,
                'Accuracy': results['accuracy'],
                'AUC_Score': results['auc']
            }
            for name, results in self.results.items()
        ])
        
        results_df.to_csv('model_results.csv', index=False)
        
        # Save feature importance
        rf_importance = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': self.models['Random Forest'].feature_importances_
        }).sort_values('Importance', ascending=False)
        
        rf_importance.to_csv('feature_importance.csv', index=False)
        
        print("Results saved successfully!")
        print("Files generated:")
        print("- hr_analytics_eda.png")
        print("- model_evaluation.png") 
        print("- feature_importance.png")
        print("- shap_analysis.png")
        print("- model_results.csv")
        print("- feature_importance.csv")
        
        return summary

def main():
    """Main function to run the complete HR Analytics pipeline"""
    
    # Initialize HR Analytics
    hr_analytics = HRAnalytics("HR_Attrition.csv")
    
    # Run complete pipeline
    try:
        # 1. Load data
        hr_analytics.load_data()
        
        # 2. Exploratory Data Analysis
        hr_analytics.exploratory_data_analysis()
        
        # 3. Preprocess data
        hr_analytics.preprocess_data()
        
        # 4. Build models
        hr_analytics.build_models()
        
        # 5. Evaluate models
        best_model = hr_analytics.evaluate_models()
        
        # 6. Feature importance analysis
        hr_analytics.feature_importance_analysis()
        
        # 7. SHAP analysis
        hr_analytics.shap_analysis()
        
        # 8. Generate insights and recommendations
        hr_analytics.generate_insights_and_recommendations()
        
        # 9. Save results
        summary = hr_analytics.save_results()
        
        print("\n" + "="*60)
        print("HR ANALYTICS PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Best performing model: {best_model}")
        print("Check generated files for detailed analysis and visualizations.")
        
    except Exception as e:
        print(f"Error in pipeline: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()