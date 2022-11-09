from flask import Flask, render_template, redirect, session, request, jsonify
#from models.chat import get_response
from functools import wraps
import pymongo

app = Flask(__name__)
app.secret_key = b'\x82\xb9\x04f\x0f\x9e\xed\x03\xc9\xf7\xa4\x1f\xb3.T\xfc'


#Database
client = pymongo.MongoClient('localhost', 27017)
db = client.user_login

#Decorators
def login_required(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        if 'logged_in' in session:
            return f(*args, **kwargs)
        else:
            return redirect('/')
        
    return wrap
#Routes
from user import routes

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html')

#To do : Predict user's emotion

# @app.post("/predict")
# def predict():
#     text = request.get_json().get("message")
#     #TODO:check if text is valid
#     response = get_response(text)
#     message = {"answer" : response}
#     return jsonify(message)


