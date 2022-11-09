from flask import Flask, request, jsonify
from app import app
from user.models import User, guess_emotion
from models.chat import get_response

@app.route('/user/signup', methods=['POST'])
def signup():
    return User().signup()

@app.route('/user/signout')
def signout():
    return User().signout()

@app.route('/user/login', methods=['POST'])
def login():
    return User().login()

@app.route('/predict-emotion', methods=['POST'])
def predict_emotion():
    user_say = request.form['user_say']
    result_text = guess_emotion(user_say)
    return str(result_text)

@app.route("/predict", methods=['POST'])
def predict():
    text = request.get_json().get("message")
    #TODO:check if text is valid
    response = get_response(text)
    message = {"answer" : response}
    return jsonify(message)