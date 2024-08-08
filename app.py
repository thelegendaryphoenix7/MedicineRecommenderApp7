import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model_filename = 'gradient_boosting_model (1).pkl'  # Ensure this is the correct filename and path
with open(model_filename, 'rb') as file:
    trained_model = joblib.load(file)

# Streamlit app
def main():
    st.title("Medicine Recommendation System")

    # Define the input fields
    disease = st.text_input("Disease")
    age = st.text_input("Age")
    dosage_form = st.text_input("Dosage Form")
    price = st.text_input("Price")

    # Predict button
    if st.button("Predict"):
        # Prepare the input data as a DataFrame
        input_data = pd.DataFrame({
            'Disease': [disease],
            'Age': [age],
            'Dosage Form': [dosage_form],
            'Price': [price]
        })
        
        # Ensure the input data goes through the same preprocessing steps as the training data
        try:
            prediction = trained_model.predict(input_data)
            st.write("Predicted Medicine:", prediction[0])
        except Exception as e:
            st.error(f"Error during prediction: {e}")

if __name__ == "__main__":
    main()
