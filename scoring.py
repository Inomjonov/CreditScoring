import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score
from joblib import dump
import streamlit as st

file_path = 'train.csv'
df = pd.read_csv(file_path)

categorical_cols = ['Occupation', 'Credit_Mix']
numeric_cols = [col for col in df.columns if col not in categorical_cols + ['Name', 'Credit_Score']]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numeric_cols),
        ('cat', OneHotEncoder(), categorical_cols)
    ])

X = df.drop(['Name', 'Credit_Score'], axis=1)
y = df['Credit_Score']

X_encoded = preprocessor.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

tree_model = DecisionTreeClassifier()
tree_model.fit(X_train, y_train)
accuracy = tree_model.score(X_test, y_test)
# print(f"Accuracy: {accuracy}")
# # Step 4: Save the trained model using joblib
dump(tree_model, 'decision_tree_model.joblib')
dump(preprocessor, 'preprocessor.joblib')
st.title("Credit Score Prediction")


name = st.text_input("Name:")
age = st.number_input("Age:")
occupation = st.selectbox("Occupation:", df['Occupation'].unique())
annual_income = st.number_input("Annual Income:")
monthly_inhand_salary = st.number_input("Monthly Inhand Salary:")
num_bank_accounts = st.number_input("Number of Bank Accounts:")
num_credit_card = st.number_input("Number of Credit Cards:")
interest_rate = st.number_input("Interest Rate:")
num_of_loan = st.number_input("Number of Loans:")
delay_from_due_date = st.number_input("Delay from Due Date:")
num_of_delayed_payment = st.number_input("Number of Delayed Payments:")
credit_mix = st.selectbox("Credit Mix:", df['Credit_Mix'].unique())
credit_history_age = st.number_input("Credit History Age:")
monthly_balance = st.number_input("Monthly Balance:")


if st.button("Predict Credit Score"):
    # Prepare input data
    input_data = pd.DataFrame({
        'Name': [name],
        'Age': [age],
        'Occupation': [occupation],
        'Annual_Income': [annual_income],
        'Monthly_Inhand_Salary': [monthly_inhand_salary],
        'Num_Bank_Accounts': [num_bank_accounts],
        'Num_Credit_Card': [num_credit_card],
        'Interest_Rate': [interest_rate],
        'Num_of_Loan': [num_of_loan],
        'Delay_from_due_date': [delay_from_due_date],
        'Num_of_Delayed_Payment': [num_of_delayed_payment],
        'Credit_Mix': [credit_mix],
        'Credit_History_Age': [credit_history_age],
        'Monthly_Balance': [monthly_balance]
    })

   
    input_encoded = preprocessor.transform(input_data)

  
    prediction = tree_model.predict(input_encoded)

   
    st.success(f"The predicted Credit Score for {name} is: {prediction[0]}")
