
import pandas as pd
import streamlit as st
import pickle
import sklearn

df = pd.read_csv('./Jamboree_file.csv')

st.title('Addmission prediction model')

st.subheader("Sample Dataframe for references")
st.dataframe(df.iloc[:,:-1]) 

try:
    with open('jamboree_linear_model.pkl','rb') as file:
        model = pickle.load(file)
except Exception as e:
    print(f"Error loading model: {e}")

st.divider()

st.subheader("Please fill in the data. :sunglasses:")

col1,col2, col3 = st.columns(3)
gre_score = col1.number_input("Input numerical value (GRE)")
toefl_score = col2.number_input("Input numerical value (TOEFL)")
university_rating = col3.selectbox("Select the University Rating",[1,2,3,4,5])
sop = col1.number_input("Input numerical value (SOP)")
lor = col2.number_input("Input numerical value (LOR)")
cgpa = col3.number_input("Input numerical value (CGPA)")
research = col1.selectbox("Select the rating (Reserch)",[0,1])

if st.button("Predict the Addmission"):
    input_data = [gre_score,toefl_score,university_rating,sop,lor,cgpa,research]
    chances = model.predict([input_data])[0]*100
    st.header(round(chances,2))
    if chances>=70:
        st.balloons()