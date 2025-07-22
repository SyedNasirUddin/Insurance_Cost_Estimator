import streamlit as st
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import ElasticNet
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
scaler = joblib.load("scalerss.pkl")
def load_and_prepare_data():
    df = pd.read_csv("insurance.csv")
    df['sex'] = df['sex'].map({'male': 1, 'female': 0})
    df['smoker'] = df['smoker'].map({'yes': 1, 'no': 0})
    df = pd.get_dummies(df, columns=['region'], drop_first=True)
    X = df.drop('charges', axis=1)
    y = df['charges']
    return X, y
X, y = load_and_prepare_data()
st.sidebar.header("Insurance Input")
age = st.sidebar.slider("Age", 18, 65, 30)
sex = st.sidebar.radio("Sex", ["male", "female"])
bmi = st.sidebar.slider("BMI", 15.0, 50.0, 25.0)
children = st.sidebar.slider("Children", 0, 5, 1)
smoker = st.sidebar.radio("Smoker", ["yes", "no"])
region = st.sidebar.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

user_input = pd.DataFrame([{
    "age": age,
    "sex": 1 if sex == "male" else 0,
    "bmi": bmi,
    "children": children,
    "smoker": 1 if smoker == "yes" else 0,
    "region_northwest": 1 if region == "northwest" else 0,
    "region_southeast": 1 if region == "southeast" else 0,
    "region_southwest": 1 if region == "southwest" else 0
}])

model_choice = st.sidebar.selectbox("Choose Model", [
    "Random Forest", "Decision Tree", "SVR", "Elastic Net", "KNN", "MLP"
])
def get_model(name):
    if name == "Random Forest":
        return RandomForestRegressor()
    elif name == "Decision Tree":
        return DecisionTreeRegressor()
    elif name == "SVR":
        return SVR()
    elif name == "Elastic Net":
        return ElasticNet()
    elif name == "KNN":
        return KNeighborsRegressor()
    elif name == "MLP":
        return MLPRegressor(max_iter=1000)
    else:
        return RandomForestRegressor()
if st.button("Insurance Charges Predicter"):
    model = get_model(model_choice)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_scaled = scaler.transform(X_train)
    model.fit(X_train_scaled, y_train)
    user_scaled = scaler.transform(user_input)
    prediction = model.predict(user_scaled)[0]
    st.success(f"Predicted Insurance Cost using {model_choice}: ₹ {prediction:,.2f}")
