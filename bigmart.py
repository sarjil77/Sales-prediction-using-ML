# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 15:13:21 2024

@author: asfaq
"""
import streamlit as st
import joblib as jb
import pandas as pd 
from sklearn.preprocessing import LabelEncoder


# loading the saved models
model = jb.load(open("C:/Machine Learning/Deploying ML models/Big mart/big.sav","rb"))

# Function to make predictions
def predict_sales(input_data):
    
    # Preprocess input_features if necessary
    # Label Encoding

    encoder = LabelEncoder()

    # we are encoding all the labels of categorical data
    # Encoding categorical data
    
    input_data['Item_Fat_Content'] = encoder.fit_transform(input_data['Item_Fat_Content'])
    input_data['Item_Type'] = encoder.fit_transform(input_data['Item_Type'])
    input_data['Outlet_Identifier'] = encoder.fit_transform(input_data['Outlet_Identifier'])
    input_data['Outlet_Size'] = encoder.fit_transform(input_data['Outlet_Size'])
    input_data['Outlet_Location_Type'] = encoder.fit_transform(input_data['Outlet_Location_Type'])
    input_data['Outlet_Type'] = encoder.fit_transform(input_data['Outlet_Type'])


    # Make predictions using the loaded model
    prediction = model.predict(input_data)
    return prediction

# Main function to run the Streamlit app
def main():
    st.title('Sales Prediction')

    # User inputs
    st.sidebar.header('Input Features')
    
    #input fields for user to enter data
    Item_Weight = st.sidebar.slider('Item Weight', 4, 30, 15)
    Item_Fat_Content = st.sidebar.selectbox('Select the fat content', ('Low Fat', 'Regular'))
    Item_Visibility = st.sidebar.slider('Item Visibility', 0.0, 1.0, 0.5)
    Item_Type = st.sidebar.selectbox('Item Type',('Dairy',
                                                  'Baking Goods', 
                                                  'Breads', 
                                                  'Breakfast', 
                                                  'Canned',
                                                  'Frozen Foods', 
                                                  'Fruits and Vegetables', 
                                                  'Hard Drinks',
                                                  'Health and Hygiene', 
                                                  'Household', 
                                                  'Meat', 
                                                  'Seafood', 
                                                  'Snack Foods', 
                                                  'Soft Drinks', 
                                                  'Starchy Foods',
                                                  'Others'))
    Item_MRP = st.sidebar.slider('Item MRP', 20, 300, 100)
    Outlet_Identifier = st.sidebar.selectbox('Outlet identifier', ('OUT010',
                                                           'OUT013',
                                                           'OUT017',
                                                           'OUT018',
                                                           'OUT019',
                                                           'OUT027',
                                                           'OUT035',
                                                           'OUT045',
                                                           'OUT046',
                                                           'OUT049'))
    Outlet_Establishment_Year = st.sidebar.selectbox('Outlet establishment year',('1985',
                                                                                  '1987',
                                                                                  '1997', 
                                                                                  '1998',
                                                                                  '1999',
                                                                                  '2002',   
                                                                                  '2004',
                                                                                  '2007',
                                                                                  '2009'))
    Outlet_Size =  st.sidebar.selectbox('Outlet Size',('Small','Medium','Large'))
    Outlet_Location_Type = st.sidebar.selectbox('Location',('Tier 1','Tier 2','Tier 3'))
    Outlet_Type = st.sidebar.selectbox('Type of Outlet',('Grocery Store','Supermarket Type 1','Supermarket Type 2','Supermarket Type 3'))
    
    
   
    # Convert 'Outlet_Establishment_Year' to integer
    Outlet_Establishment_Year = int(Outlet_Establishment_Year)
    
    # Create a dataframe from user inputs
    input_data = pd.DataFrame({
                               'Item_Weight':Item_Weight,
                               'Item_Fat_Content':Item_Fat_Content,
                               'Item_Visibility':Item_Visibility,
                               'Item_Type':Item_Type,
                               'Item_MRP':Item_MRP,
                               'Outlet_Identifier':Outlet_Identifier,
                               'Outlet_Establishment_Year':Outlet_Establishment_Year,
                               'Outlet_Size':Outlet_Size,
                               'Outlet_Location_Type':Outlet_Location_Type,
                               'Outlet_Type':Outlet_Type
                                 }, index = [0])

    # Make predictions when user clicks the predict button
    if st.sidebar.button('Predict'):
        prediction = predict_sales(input_data)
        st.write('Predicted Sales:', prediction)

if __name__ == '__main__':
    main()