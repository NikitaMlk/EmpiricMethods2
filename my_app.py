import streamlit as st

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
