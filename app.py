import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load the pre-trained model from the .pkl file
with open('LinearRegressionModel.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the dataset to get feature names and data
df = pd.read_csv(r"D:\car\carprices.csv")

# Streamlit app title
st.title("Car Price Prediction App")

# Sidebar for user input
st.sidebar.header('User Input Features')

# Dropdown for car company
companies = sorted(df['Car Model'].unique())
selected_car_model = st.sidebar.selectbox('Select Car Model', companies)

# Input for mileage
mileage = st.sidebar.number_input('Mileage (in km)', min_value=0, max_value=500000, step=1000, value=10000)

# Input for age of the car
selected_age = st.sidebar.number_input('Age of Car (in years)', min_value=0, max_value=30, step=1, value=5)

# Button to predict the car price
if st.sidebar.button('Predict Price'):
    # Prepare input data in the correct format
    input_data = pd.DataFrame([[selected_car_model, mileage, selected_age]],
                               columns=['Car Model', 'Mileage', 'Age(yrs)'])

    # One-hot encode the input data
    input_data_encoded = pd.get_dummies(input_data, columns=['Car Model'])

    # Align the columns of the input data with the model
    input_data_encoded = input_data_encoded.reindex(columns=model.feature_names_in_, fill_value=0)

    # Make prediction
    try:
        prediction = model.predict(input_data_encoded)
        st.write(f"### Predicted Car Price: ${np.round(float(prediction[0]), 2):,.2f}")
    except Exception as e:
        st.error(f"Error in prediction: {e}")
        st.write("Input Data:", input_data_encoded)

# Option to display raw data
if st.checkbox('Show Raw Data'):
    st.write(df)
