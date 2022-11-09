import json
from flask import Flask, jsonify, request, session, redirect
from passlib.hash import pbkdf2_sha256
from app import db
import uuid
import re
from user.emo_classification_model import *
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

class User:
    
    def start_session(self, user):
        del user['password']
        session['logged_in'] = True
        session['user'] = user
        return jsonify(user), 200
    
    def signup(self):
        print(request.form)
        
        # Create the user object
        user = {
            "_id": uuid.uuid4().hex,
            "email": request.form.get('email'),
            "password": request.form.get('password'),
            "nickname": request.form.get('nickname'),
            "age": request.form.get('age'),
            "gender": request.form.get('gender')
        }
        
        #Encrypt the password
        user['password'] = pbkdf2_sha256.encrypt(user['password'])
        
        #Check for existing email address
        if db.users.find_one({"email": user['email']}):
            return jsonify({ "error": "Email address already in use" }), 400
        
        if db.users.insert_one(user):
            return self.start_session(user)
        
        return jsonify({ "error": "Signup failed"}), 400
    
    def signout(self):
        session.clear()
        return redirect('/')
    
    def login(self):
        
        user = db.users.find_one({
            "email": request.form.get('email')
        })
        
        if user and pbkdf2_sha256.verify(request.form.get('password'), user['password']):
            return self.start_session(user)
        
        return jsonify({ "error": "Invaild login credentials"}), 401

SENT_MAX_LEN = 13
encoder = LabelEncoder()
encoder.fit(train_df['감정_대분류'])

def clean_text(sent):
    sent_clean = re.sub("[^가-힣ㄱ-ㅎㅏ-ㅣ\\s]", "", sent)
    return sent_clean

def guess_emotion(sentence):
    test_data_sents = []

    test_tokenized_text = vocab[tokenizer(clean_text(sentence))]

    tokens = [vocab[vocab.bos_token]]
    tokens += pad_sequences([test_tokenized_text],
                            SENT_MAX_LEN,
                            value=vocab[vocab.padding_token],
                            padding='post').tolist()[0]
    tokens += [vocab[vocab.eos_token]]

    test_data_sents.append(tokens)
    test_data_sents = np.array(test_data_sents, dtype=np.int64)

    #     load_model = TFGPT2Classifier(dir_path='./gpt_ckpt', num_class=6)
    #     load_model.load_weights('model_weights_output_gpt2/02-0.96427.ckpt')
    #     load_model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

    predicted_emotion = load_model.predict(test_data_sents)
    # print(predicted_emotion)
    emotion = np.argmax(predicted_emotion, axis=1)
    emotion = encoder.inverse_transform(emotion)
    print(emotion)
    return emotion