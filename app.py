from flask import Flask,request,app,render_template
from flask import Response
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

scaler = pickle.load(open("/Users/mdshalique/Downloads/practicePythonOP/ProjectsOP/Diabetes_Prediction/Model/StandardScaler.pkl","rb"))
model = pickle.load(open("/Users/mdshalique/Downloads/practicePythonOP/ProjectsOP/Diabetes_Prediction/Model/DiabetesModel.pkl","rb"))

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predictData", methods = ["GET","POST"])
def predict_datapoint():
    results = ""
    
    if request.method == "POST" :
        Pregnancies = float(request.form.get("Pregnancies"))
        Glucose = float(request.form.get("Glucose"))
        BloodPressure = float(request.form.get("BloodPressure"))
        SkinThickness = float(request.form.get("SkinThickness"))
        Insulin = float(request.form.get("Insulin"))
        BMI = float(request.form.get("BMI"))
        DiabetesPedigreeFunction = float(request.form.get("DiabetesPedigreeFunction"))
        Age = float(request.form.get("Age"))
        
        Age = float(request.form.get("Age"))
        scaled_Data = scaler.transform([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
        prediction = model.predict(scaled_Data)
        
        if prediction[0] == 1:
            result = "Diabetic"
        else:
            result = "Non-Diabetic"
            
            
        return render_template("Prediction.html",result = result)
    
    else:
        return render_template("home.html")
    
if __name__ == '__main__':
    app.run(host = "0.0.0.0")