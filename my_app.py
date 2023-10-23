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
def plot_numeric_distribution(data):
    numeric_features = data.select_dtypes(include=['float64', 'int64'])

    n_cols = 3  # Adjust the number of columns
    n_rows = int(math.ceil(numeric_features.shape[1] / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 10))
    axes = axes.flatten()

    for i, col in enumerate(numeric_features.columns):
        ax = axes[i]
        sns.histplot(data[col], kde=True, ax=ax)
        ax.set_title(f'Distribution of {col}')
        ax.set_xlabel(col)
        ax.set_ylabel('Frequency')

    for i in range(numeric_features.shape[1], len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()

    return fig

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
    plot_numeric_distribution(data)

if __name__ == "__main__":
    main()
