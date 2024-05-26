import streamlit as st
import pickle as p
import numpy as np

# Load the trained model
try:
    with open('food.pkl', 'rb') as file:
        model = p.load(file)
    if model is None:
        st.error("Model file is empty.")
except FileNotFoundError:
    st.error("Model file not found. Please ensure that 'food.pkl' exists in the current directory.")
except Exception as e:
    st.error(f"Error loading model: {e}")

def predict_delivery_time(Delivery_person_Age,Total_distance,Delivery_person_Ratings,Weatherconditions,Road_traffic_density,Vehicle_condition,multiple_deliveries,Festival,City):
    try:
        # Create input array
        features = np.array([Delivery_person_Age, Total_distance,Delivery_person_Ratings,Weatherconditions,Road_traffic_density,Vehicle_condition,multiple_deliveries,Festival,City]).reshape(1, -1)

        # Predict
        prediction = model.predict(features)

        return prediction[0]  # Adjust this according to your model's output shape
    except Exception as e:
        st.error(f"Prediction error: {e}")



def main():
    st.title('Food Delivery Time Prediction')
    st.write('Fill out the form below to predict delivery time.')
    # Input form
    with st.form(key='delivery_time_form'):
        Delivery_person_Age = st.number_input('Delivery Person Age', min_value=18, step=1)
        Delivery_person_Ratings = st.slider('Delivery Person Ratings', min_value=1.0, max_value=5.0, step=0.1)
        Total_distance = st.number_input('Distance', min_value=1, step=1)
        Weatherconditions = st.number_input('Weather conditions', min_value=0,max_value=4, step=1)
        Road_traffic_density = st.number_input('Road traffic density', min_value=0,max_value=3, step=1)
        Vehicle_condition = st.number_input('Vehicle condition', min_value=0.0,max_value=3.0, step=1.0)
        multiple_deliveries = st.number_input('Multiple Deliveries', min_value=0.0,max_value=5.0, step=1.0)
        Festival = st.number_input('Festival', min_value=0,max_value=1, step=1)
        City = st.number_input('City', min_value=1,max_value=2, step=1)



        
        

        if st.form_submit_button(label='Predict'):
            prediction = predict_delivery_time(Delivery_person_Age,Total_distance,Delivery_person_Ratings,Weatherconditions,Road_traffic_density,Vehicle_condition,multiple_deliveries,Festival,City)
            if prediction is not None:
                st.write(f'Predicted Delivery Time: {prediction} minutes')

if __name__ == '__main__':
    main()
