# Diabetes Prediction System

This project is a web-based diabetes prediction system using a 
Random Forest machine learning model.

The model is trained in Google Colab and integrated with a Flask backend.
The frontend is developed using HTML, CSS, and JavaScript.

## Technologies Used
- Python
- Flask
- Scikit-learn
- HTML, CSS, JavaScript
- Google Colab (for model training)

## Project Structure
Diabetes_Project/
│
├── app.py
├── diabetes_model.pkl
│
├── templates/
│   └── index.html
│
├── static/
│   ├── style.css
│   └── script.js

## How to Run the Project
1. Install required libraries:
   pip install flask scikit-learn numpy

2. Run the Flask app:
   python app.py

3. Open browser and go to:
   http://127.0.0.1:5000/

## Description
The user enters health parameters such as age, BMI, blood pressure,
glucose levels, and lifestyle factors. The backend processes the data
using a trained Random Forest model and displays the prediction result.

## Note
This system performs real-time predictions and does not use a database.
