"""
HR Analytics Interactive Dashboard
Alternative to Power BI using Plotly Dash
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc

class HRDashboard:
    def __init__(self, data_path):
        """Initialize the HR Dashboard"""
        self.df = pd.read_csv(data_path)
        self.app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        self.setup_layout()
        self.setup_callbacks()
    
    def create_summary_cards(self):
        """Create summary cards for key metrics"""
        total_employees = len(self.df)
        attrition_rate = (self.df['Attrition'] == 'Yes').mean() * 100
        avg_age = self.df['Age'].mean()
        avg_income = self.df['MonthlyIncome'].mean()
        
        cards = dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4(f"{total_employees:,}", className="card-title text-primary"),
                        html.P("Total Employees", className="card-text")
                    ])
                ], className="mb-3")
            ], width=3),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4(f"{attrition_rate:.1f}%", className="card-title text-danger"),
                        html.P("Attrition Rate", className="card-text")
                    ])
                ], className="mb-3")
            ], width=3),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4(f"{avg_age:.0f}", className="card-title text-info"),
                        html.P("Average Age", className="card-text")
                    ])
                ], className="mb-3")
            ], width=3),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4(f"${avg_income:,.0f}", className="card-title text-success"),
                        html.P("Average Income", className="card-text")
                    ])
                ], className="mb-3")
            ], width=3)
        ])
        
        return cards
    
    def create_attrition_overview(self):
        """Create attrition overview chart"""
        attrition_counts = self.df['Attrition'].value_counts()
        
        fig = go.Figure(data=[
            go.Pie(
                labels=attrition_counts.index,
                values=attrition_counts.values,
                hole=0.4,
                marker_colors=['#2E86AB', '#A23B72']
            )
        ])
        
        fig.update_layout(
            title="Overall Attrition Distribution",
            title_x=0.5,
            height=400
        )
        
        return fig
    
    def create_department_analysis(self):
        """Create department-wise attrition analysis"""
        dept_data = pd.crosstab(self.df['Department'], self.df['Attrition'], normalize='index') * 100
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='No Attrition',
            x=dept_data.index,
            y=dept_data['No'],
            marker_color='#2E86AB'
        ))
        
        fig.add_trace(go.Bar(
            name='Attrition',
            x=dept_data.index,
            y=dept_data['Yes'],
            marker_color='#A23B72'
        ))
        
        fig.update_layout(
            title="Attrition Rate by Department",
            xaxis_title="Department",
            yaxis_title="Percentage (%)",
            barmode='stack',
            height=400
        )
        
        return fig
    
    def create_income_analysis(self):
        """Create income vs attrition analysis"""
        fig = px.box(
            self.df, 
            x='Attrition', 
            y='MonthlyIncome',
            color='Attrition',
            title="Monthly Income Distribution by Attrition Status"
        )
        
        fig.update_layout(height=400)
        return fig
    
    def create_age_distribution(self):
        """Create age distribution by attrition"""
        fig = px.histogram(
            self.df,
            x='Age',
            color='Attrition',
            nbins=20,
            title="Age Distribution by Attrition Status",
            opacity=0.7
        )
        
        fig.update_layout(height=400)
        return fig
    
    def create_job_role_analysis(self):
        """Create job role attrition analysis"""
        job_attrition = pd.crosstab(self.df['JobRole'], self.df['Attrition'], normalize='index')['Yes'] * 100
        job_attrition = job_attrition.sort_values(ascending=True).tail(10)
        
        fig = go.Figure(go.Bar(
            x=job_attrition.values,
            y=job_attrition.index,
            orientation='h',
            marker_color='#F18F01'
        ))
        
        fig.update_layout(
            title="Top 10 Job Roles by Attrition Rate",
            xaxis_title="Attrition Rate (%)",
            yaxis_title="Job Role",
            height=500
        )
        
        return fig
    
    def create_overtime_analysis(self):
        """Create overtime vs attrition analysis"""
        overtime_data = pd.crosstab(self.df['OverTime'], self.df['Attrition'], normalize='index') * 100
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='No Attrition',
            x=overtime_data.index,
            y=overtime_data['No'],
            marker_color='#2E86AB'
        ))
        
        fig.add_trace(go.Bar(
            name='Attrition',
            x=overtime_data.index,
            y=overtime_data['Yes'],
            marker_color='#A23B72'
        ))
        
        fig.update_layout(
            title="Overtime vs Attrition Rate",
            xaxis_title="Overtime",
            yaxis_title="Percentage (%)",
            barmode='group',
            height=400
        )
        
        return fig
    
    def create_satisfaction_analysis(self):
        """Create job satisfaction analysis"""
        satisfaction_data = pd.crosstab(self.df['JobSatisfaction'], self.df['Attrition'], normalize='index')['Yes'] * 100
        
        fig = go.Figure(go.Scatter(
            x=satisfaction_data.index,
            y=satisfaction_data.values,
            mode='lines+markers',
            marker=dict(size=10, color='#C73E1D'),
            line=dict(width=3)
        ))
        
        fig.update_layout(
            title="Job Satisfaction vs Attrition Rate",
            xaxis_title="Job Satisfaction Level",
            yaxis_title="Attrition Rate (%)",
            height=400
        )
        
        return fig
    
    def create_correlation_heatmap(self):
        """Create correlation heatmap"""
        # Select numeric columns
        numeric_cols = ['Age', 'MonthlyIncome', 'YearsAtCompany', 'YearsSinceLastPromotion', 
                       'YearsWithCurrManager', 'JobSatisfaction', 'WorkLifeBalance']
        
        corr_data = self.df[numeric_cols + ['Attrition']].copy()
        corr_data['Attrition'] = (corr_data['Attrition'] == 'Yes').astype(int)
        
        correlation_matrix = corr_data.corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=correlation_matrix.round(2).values,
            texttemplate="%{text}",
            textfont={"size": 10}
        ))
        
        fig.update_layout(
            title="Feature Correlation Heatmap",
            height=500
        )
        
        return fig
    
    def setup_layout(self):
        """Setup the dashboard layout"""
        self.app.layout = dbc.Container([
            # Header
            dbc.Row([
                dbc.Col([
                    html.H1("HR Analytics Dashboard", className="text-center mb-4"),
                    html.Hr()
                ])
            ]),
            
            # Summary Cards
            self.create_summary_cards(),
            
            html.Br(),
            
            # First row of charts
            dbc.Row([
                dbc.Col([
                    dcc.Graph(
                        id='attrition-overview',
                        figure=self.create_attrition_overview()
                    )
                ], width=6),
                
                dbc.Col([
                    dcc.Graph(
                        id='department-analysis',
                        figure=self.create_department_analysis()
                    )
                ], width=6)
            ]),
            
            # Second row of charts
            dbc.Row([
                dbc.Col([
                    dcc.Graph(
                        id='income-analysis',
                        figure=self.create_income_analysis()
                    )
                ], width=6),
                
                dbc.Col([
                    dcc.Graph(
                        id='age-distribution',
                        figure=self.create_age_distribution()
                    )
                ], width=6)
            ]),
            
            # Third row of charts
            dbc.Row([
                dbc.Col([
                    dcc.Graph(
                        id='job-role-analysis',
                        figure=self.create_job_role_analysis()
                    )
                ], width=6),
                
                dbc.Col([
                    dcc.Graph(
                        id='overtime-analysis',
                        figure=self.create_overtime_analysis()
                    )
                ], width=6)
            ]),
            
            # Fourth row of charts
            dbc.Row([
                dbc.Col([
                    dcc.Graph(
                        id='satisfaction-analysis',
                        figure=self.create_satisfaction_analysis()
                    )
                ], width=6),
                
                dbc.Col([
                    dcc.Graph(
                        id='correlation-heatmap',
                        figure=self.create_correlation_heatmap()
                    )
                ], width=6)
            ]),
            
            # Footer
            html.Hr(),
            dbc.Row([
                dbc.Col([
                    html.P("HR Analytics Dashboard - Employee Attrition Analysis", 
                          className="text-center text-muted")
                ])
            ])
            
        ], fluid=True)
    
    def setup_callbacks(self):
        """Setup interactive callbacks (if needed)"""
        # Add any interactive callbacks here
        pass
    
    def run_dashboard(self, debug=True, port=8050):
        """Run the dashboard"""
        print("Starting HR Analytics Dashboard...")
        print(f"Dashboard will be available at: http://localhost:{port}")
        self.app.run_server(debug=debug, port=port)

def main():
    """Main function to run the dashboard"""
    try:
        # Create and run dashboard
        dashboard = HRDashboard("HR_Attrition.csv")
        dashboard.run_dashboard()
        
    except Exception as e:
        print(f"Error running dashboard: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()