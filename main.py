import streamlit as st
from GradientDescentModel import Cross_Entropy
import time

st.title("Insurance Price Prediction")
# Create a sidebar with navigation options
navigation_menu = st.sidebar.radio("Navigation Menu", ["Predict Insurance Cost"])

def main():
    if navigation_menu == "Predict Insurance Cost":
        Predict_value()

def display_spinner():
    with st.spinner('Loading....'):
        time.sleep(5)
        st.success('Done!')

def Predictionbtn():
    
    # restoring values
    age = st.session_state.age
    sex = st.session_state.sex
    bmi = st.session_state.bmi
    child = st.session_state.child
    smoker = st.session_state.smoker
    region = st.session_state.region
    
    # converting values
    age = int(age)
    bmi = float(bmi)
    child = int(child)
    rearranged_value = {
        'sex': {'male': 0, 'female': 1},
        'smoker': {'yes': 0, 'no': 1},
        'region': {'southeast': 0, 'southwest': 1, 'northeast': 2, 'northwest': 3}
    }
    c_sex = rearranged_value["sex"][sex.lower()]
    c_smoke = rearranged_value["smoker"][smoker.lower()]
    c_region = rearranged_value["region"][region.lower()]
    
    # predicting values
    predicted_value = Cross_Entropy(age, c_sex, bmi, child, c_smoke, c_region)
    display_spinner()
    st.text(f"The insurance cost is USD => {round(predicted_value)}")

def clear():
    st.session_state.age = ""
    st.session_state.sex = "Male"
    st.session_state.bmi = ""
    st.session_state.child = ""
    st.session_state.smoker = "Yes"
    st.session_state.region = "southeast"

def Predict_value():
    age = st.text_input("Enter Your Age")
    sex = st.radio("Choose your gender", ["Male", "Female"])
    bmi = st.text_input("Enter your BMI value")
    child = st.text_input("Enter Your Number of children")
    smoker = st.radio("Choose your smoke", ["Yes", "No"])
    region = st.radio("Choose Your Region", ["southeast", "southwest", "northeast", "northwest"])
    
    if st.button("submit", use_container_width=50, type="primary"):
        st.session_state.age = age
        st.session_state.sex = sex
        st.session_state.bmi = bmi
        st.session_state.child = child
        st.session_state.smoker = smoker
        st.session_state.region = region
        Predictionbtn()
    if st.button("clear", use_container_width=50):
        clear()


if __name__ == "__main__":
    main()