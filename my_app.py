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
@st.cache
def load_data():
    data = arff.loadarff('jm1.arff')
    df = pd.DataFrame(data[0])
    return df

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

    numeric_features = data.select_dtypes(include=['float64', 'int64'])

    # Create subplots
    fig, axes = plt.subplots(1, len(selected_features), figsize=(15, 5))

    # Set line style
    line_styles = ['solid', 'dotted', 'dashed']
    line_style = line_styles[0]  # Default to solid line
    if line_type in line_styles:
        line_style = line_type

    # Set line width
    try:
        line_width = int(line_width)
    except ValueError:
        line_width = 1

    # Set color
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    color = colors[0]  # Default to blue
    if color_type in colors:
        color = color_type

    # Plot histograms for each selected numeric feature
    for i, col in enumerate(selected_features):
        ax = axes[i]
        sns.histplot(data[col], kde=True, ax=ax, color=color, linestyle=line_style, linewidth=line_width)
        ax.set_title(f'Distribution of {col}')
        ax.set_xlabel(col)
        ax.set_ylabel('Frequency')

    plt.tight_layout()
    st.pyplot(fig)

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
