import joblib
import numpy as np

model = joblib.load("model.joblib")
scaler = joblib.load("scaler.joblib")

def evaluate_student(attendance, assignment, quiz, mid, study_hours):
    data = np.array([[attendance, assignment, quiz, mid, study_hours]])
    data_scaled = scaler.transform(data)
    prediction = model.predict(data_scaled)
    return "Pass" if prediction[0] == 1 else "Fail"