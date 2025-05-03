# GEnder -- > 1  Female   0 Male 
# Churn --- > 1 Yes       0 No 
# scaler is exported 
import  streamlit as st 
import joblib 
import numpy as np 

scaler = joblib.load("scaler.pkl")
model = joblib.load("gridkn.pkl")

st.title("📊📊 Churn Prediction App ")
st.divider()
st.write("Please Enter the Value and hit The predict Button for Getting a Prefiction🔍🔍 ")

st.divider()

age =st.number_input("Enter age ",min_value=10,max_value=100,value=30)

tenure = st.number_input("Enter Tenue" ,min_value=0,max_value=130,value=10)

month_charge = st.number_input("Enter montly Charge" , min_value=30 ,max_value=150)


gender = st.selectbox("Enter the Gender ", ["Male","Female"])

st.divider()

prediction = st.button("prediction") 
if prediction :
    gender_selected = 1 if gender == "Female" else 0 
    x= [age,gender_selected,tenure,month_charge]
    X1 = np.array(x)
    X_array =scaler.transform([X1])
    prediction = model.predict(X_array)[0]

    predictied = "Churn" if prediction == 1 else "Not Churn"
    st.balloons()
    st.write(f"predictied : {predictied}")

else : 
    st.write("Please enter the value and use predict button ")    

