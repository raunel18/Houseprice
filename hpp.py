'''
import streamlit as st
import pickle
import numpy as np

#Load the model from pickel file
with open ('hpp.pkl', 'rb') as pickleFile:
    model = pickle.load(pickleFile)
    
# Streamlit app layout
st.title ('House Price Prediction App')
st.image ('house.png', width=400, caption='Predict Prices')

# Function to encode ocean proximity
def encode_ocean_proximity(ocean_proximity):
    ocean_mapping = {
        '<1H OCEAN': [1, 0, 0, 0, 0],
        'INLAND': [0, 1, 0, 0, 0],
        'ISLAND':[0, 0, 1, 0, 0],
        'NEAR BAY':[0, 0, 0, 1, 0],
        'NEAR OCEAN': [0, 0, 0, 0, 1]
    }
    return ocean_mapping.get(ocean_proximity, [0, 0, 0, 0, 0])
# Function to make predictions
def predict_price(housing_median_age, median_income,  ocean_proximity):
    #Combine features and encode ocean proximity
    features = [housing_median_age, median_income] + encode_ocean_proximity(ocean_proximity)
    return model.predict([features])[0]

# Siderbar instructions
st.sidebar.header('How to Use!')
instructions = """
1. Enter the details of the house in the form below.
2. Click 'Predict' to see the estimated price.
3. Adjust values to test different scenarios.

**Example** A 2000 sqft house, 3 bedrooms, 2 bathrooms, with garage in Location 1
"""

st.sidebar.markdown(instructions)
st.sidebar.info('Data source and model details can be provided here.')

# Main content 
st.subheader('Enter House Details')
col1, col2 = st.columns(2)

# User inputs in Column 1
with col1:
    housing_median_age = st.number_input('Housing Median Age:', min_value = 0, value=41)
    median_income = st.number_input('Median Income:', min_value=0.0, format = "%.4f", value = 8.3252)
    ocean_proximity = st.selectbox('Ocean Proximity:', ['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN'])

# Predict button and results
if st.button('Predict Price'):
    predicted_price = predict_price(housing_median_age, median_income, ocean_proximity)
    st.success(f'Predicted Median House Value: ${predicted_price:,.2f}')
    st.balloons()
    
# Run the app
if __name__ == '__main__':
    pass
'''

import streamlit as st
import pickle
import numpy as np

#Load the model from pickel file
with open ('hpp.pkl', 'rb') as pickleFile:
    model = pickle.load(pickleFile)
    
# Streamlit app layout
st.title ('House Price Prediction App')
st.image ('house.png', width=400, caption='Predict Prices')

# Function to encode ocean proximity
def encode_ocean_proximity(ocean_proximity):
    ocean_mapping = {
        '<1H OCEAN': [1, 0, 0, 0, 0],
        'INLAND': [0, 1, 0, 0, 0],
        'ISLAND':[0, 0, 1, 0, 0],
        'NEAR BAY':[0, 0, 0, 1, 0],
        'NEAR OCEAN': [0, 0, 0, 0, 1]
    }
    return ocean_mapping.get(ocean_proximity, [0, 0, 0, 0, 0])
# Function to make predictions
def predict_price(housing_median_age, median_income,  ocean_proximity):
    #Combine features and encode ocean proximity
    features = [housing_median_age, median_income] + encode_ocean_proximity(ocean_proximity)
    return model.predict([features])[0]

# Siderbar instructions
st.sidebar.header('How to Use!')
instructions = """
1. Enter the details of the house in the form below.
2. Click 'Predict' to see the estimated price.
3. Adjust values to test different scenarios.

**Example** A 2000 sqft house, 3 bedrooms, 2 bathrooms, with garage in Location 1
"""

st.sidebar.markdown(instructions)
st.sidebar.info('Data source and model details can be provided here.')

# Main content 
st.subheader('Enter House Details')
col1, col2 = st.columns(2)

# User inputs in Column 1
with col1:
    housing_median_age = st.number_input('Housing Median Age:', min_value = 0, value=41)
    median_income = st.number_input('Median Income:', min_value=0.0, format = "%.4f", value = 8.3252)
    ocean_proximity = st.selectbox('Ocean Proximity:', ['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN'])

# Predict button and results
if st.button('Predict Price'):
    predicted_price = predict_price(housing_median_age, median_income, ocean_proximity)
    st.success(f'Predicted Median House Value: ${predicted_price:,.2f}')
    st.balloons()
    
# Run the app
if __name__ == '__main__':
    pass