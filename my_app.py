import streamlit as st

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

st.title("Data Visualization App")
data = load_data()

st.sidebar.header("Data Filters")
min_rows = st.sidebar.number_input("Minimum Number of Rows", min_value=0, value=0)
selected_features = st.sidebar.multiselect("Select Features for Visualization", data.columns)

filtered_data = data.head(min_rows)  # Filter by the number of rows

st.sidebar.header("Visualization Settings")
color_type = st.sidebar.selectbox("Color Type", ['b', 'g', 'r', 'c', 'm', 'y', 'k'])
line_type = st.sidebar.selectbox("Line Type", ['solid', 'dotted', 'dashed'])

st.subheader("Feature Importance")
feature_importance(data, selected_features)

st.subheader("Numeric Feature Distribution")
plot_numeric_distribution(data, selected_features, color_type, line_type)
