import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from scipy.io import arff
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

# Function to load data
def load_data():
    url = "http://promise.site.uottawa.ca/SERepository/datasets/jm1.arff"
    try:
        response = requests.get(url)
        data = arff.loads(response.text)
        df = pd.DataFrame(data[0])
        # Handle missing values by filling with median
        df.fillna(df.median(), inplace=True)
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# Function to calculate feature importance
def feature_importance(data, target_col):
    if target_col not in data.columns:
        st.warning(f"{target_col} not found in the dataset.")
        return

    # Split the data into features and target
    X = data.drop(columns=[target_col])
    y = data[target_col]

    # Encode the categorical labels to numeric
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Impute missing values
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)

    # Fit a random forest classifier to determine feature importance
    clf = RandomForestClassifier()
    clf.fit(X_imputed, y_encoded)

    # Get feature importances
    feature_importances = clf.feature_importances_

    # Create a DataFrame to store feature names and their importance scores
    importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})

    # Sort features by importance
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    # Plot feature importance using Plotly
    fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h', title='Feature Importance')
    st.plotly_chart(fig)

# Function to plot numeric distribution
def plot_numeric_distribution(data, selected_features, color_type, line_type, line_width):
    if not selected_features:
        st.warning("Please select numeric features for visualization.")
        return

    # Create a figure with subplots
    fig = make_subplots(rows=1, cols=len(selected_features), subplot_titles=selected_features, shared_yaxes=True)

    # Set line style
    line_styles = ['solid', 'dotted', 'dashed']
    line_style = line_styles[0]  # Default to solid line
    if line_type in line_styles:
        line_style = line_type

    # Set line width
    try:
        line_width = float(line_width)
    except ValueError:
        line_width = 1.0

    # Set color
    colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black']
    color = colors[0]  # Default to blue
    if color_type in colors:
        color = color_type

    # Plot histograms for each selected numeric feature
    for i, col in enumerate(selected_features):
        histogram = go.Histogram(x=data[col], marker_color=color)
        histogram.update_xaxes(title_text=col)
        histogram.update_yaxes(title_text='Frequency')

        fig.add_trace(histogram, row=1, col=i+1)

    # Update the layout
    fig.update_layout(
        title="Numeric Feature Distribution",
        xaxis=dict(type='category'),  # Use category type for x-axis labels
        xaxis_showgrid=False,
        yaxis_showgrid=False,
        barmode='overlay',
        bargap=0.1,
        bargroupgap=0.1,
        showlegend=False
    )

    # Update line style and line width
    for i in range(len(selected_features)):
        fig.update_traces(marker=dict(line=dict(width=line_width, dash=line_style))

    # Show the Plotly figure in Streamlit
    st.write(fig)

# Streamlit app
def main():

    # Function to load data
    st.title("Data Visualization App")
    data = load_data()
    
    st.sidebar.header("Data Filters")
    selected_features = st.sidebar.multiselect("Select Features for Visualization", data.columns)

    st.sidebar.header("Visualization Settings")
    color_type = st.sidebar.selectbox("Color Type", ['b', 'g', 'r', 'c', 'm', 'y', 'k'])
    line_type = st.sidebar.selectbox("Line Type", ['solid', 'dotted', 'dashed'])
    line_width = st.sidebar.number_input("Line Width", min_value=0.1, max_value=6.0, value=0.1)

    st.subheader("Feature Importance")
    feature_importance(data, target_col='defects')

    st.subheader("Numeric Feature Distribution")
    plot_numeric_distribution(data, selected_features, color_type, line_type, line_width)

if __name__ == "__main__":
    main()
