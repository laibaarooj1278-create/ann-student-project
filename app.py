import streamlit as st
from predict import evaluate_student

st.title("🎓 Student Performance Predictor")

attendance = st.number_input("Attendance")
assignment = st.number_input("Assignment")
quiz = st.number_input("Quiz")
mid = st.number_input("Mid Exam")
study_hours = st.number_input("Study Hours")

if st.button("Predict"):
    result = evaluate_student(attendance, assignment, quiz, mid, study_hours)
    
    st.success(f"Prediction: {result}")

    if result == 1:
        st.write("Good performance 👍")
    else:
        st.write("Needs improvement 📉")