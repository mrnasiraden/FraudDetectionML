import numpy as np
import pandas as pd
import pickle
import streamlit as st



# Mapping between original feature names and selected features
feature_mapping = {
    'Time': 0,
    'V1': 1, 'V2': 2, 'V3': 3, 'V4': 4, 'V5': 5,
    'V6': 6, 'V7': 7, 'V8': 8, 'V9': 9, 'V10': 10,
    'V11': 11, 'V12': 12, 'V13': 13, 'V14': 14, 'V15': 15,
    'V16': 16, 'V17': 17, 'V18': 18, 'V19': 19, 'V20': 20,
    'V21': 21, 'V22': 22, 'V23': 23, 'V24': 24, 'V25': 25,
    'V26': 26, 'V27': 27, 'V28': 28, 'Amount': 29
}

selected_features = ['V17', 'V14', 'V10', 'V12', 'V7', 'V4', 'V21', 'V3', 'V28', 'V18',
                     'V26', 'V20', 'V27', 'V1', 'V25', 'V13', 'V15', 'Time', 'V8',
                     'V24', 'V11', 'V9', 'V2', 'Amount', 'V16']

def app():

    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://www.investopedia.com/thmb/XXyQSZtX3thaBvs5nW5NY3lUzdU=/1500x0/filters:no_upscale():max_bytes(150000):strip_icc()/credit-card--concept-credit-card-payment-1156857742-c265746dcaea46e6bcc5f0bcda1ed871.jpg");
             background-attachment: fixed;
             background-size: cover;

         }}
         </style>
         """,
         unsafe_allow_html=True
     )
    
    # Loading our trained model
    pickle_in = open("xgb_tuned.pkl", "rb")
    xgb = pickle.load(pickle_in)

    st.title("Credit Card Fraud Detection System")
    st.write("Enter the following features to check if the transaction is legitimate or fraudulent:")

    input_df = st.text_input('Input All features')
    input_df_lst = input_df.split(',')


    if input_df and len(input_df_lst) == len(feature_mapping):
        filtered_input = [input_df_lst[feature_mapping[col]] for col in selected_features]

    predict = st.button("Predict")

    if predict:
        features = np.array(filtered_input, dtype=np.float64)
        prediction = xgb.predict(features.reshape(1, -1))
        if prediction[0] == 0:
                st.markdown(
                '<div style="background-color: #FFFFFF; padding: 2px; border-radius: 2px; text-align: center; width: 50%;">'
                '<h4 style="color: #15FF0D;">Legitmate transaction</h4>'
                '</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div style="background-color: #FFFFFF; padding: 2px; border-radius: 2px; text-align: center; width: 50%;">'
                '<h4 style="color: #F30000;">Fraudulent transaction!</h4>'
                '</div>',
                unsafe_allow_html=True
            )
    else:
        st.warning("Please enter valid feature values.")


if __name__ == '__main__':
    app()
