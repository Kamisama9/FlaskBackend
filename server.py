from flask import Flask, request, jsonify
import os
import random
import json
import pickle
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
import numpy as np

app = Flask(__name__, static_url_path='/static')

# uncomment to Download 
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('omw-1.4')

lemmatizer = WordNetLemmatizer()


words_file_path = os.path.join(app.root_path, 'static', 'words.pkl')
classes_file_path = os.path.join(app.root_path, 'static', 'classes.pkl')
model_file_path = os.path.join(app.root_path, 'static', 'chatbot_model.h5')


words = pickle.load(open(words_file_path, 'rb'))
classes = pickle.load(open(classes_file_path, 'rb'))
model = load_model(model_file_path)


intents_file_path = os.path.join(app.root_path, 'static', 'intents.json')
intents_data = json.loads(open(intents_file_path).read())

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)

    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]

    ERROR_THRESHOLD = 0.25

    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)

    return_list = []

    for r in results:
        return_list.append({'intent': classes[r[0]],
                            'probability': str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    if intents_list:
        intent_tag = intents_list[0]['intent']
        list_of_intents = intents_json['intents']

        result = ''

        for intent in list_of_intents:
            if 'tag' in intent and intent['tag'] == intent_tag:
                result = random.choice(intent['responses'])
                break
        return result
    else:
        return "Sorry, I couldn't understand your symptoms. Please try again."

def display_doctor_info(intent_tag):
    for intent in intents_data['intents']:
        if 'tag' in intent and intent['tag'].lower() == intent_tag.lower() and 'doctor' in intent:
            return f"Doctor: {intent['doctor']}\nContact: {intent['doctor_phone']}"
    return "Doctor information not available."

def calling_the_bot(text_input):
    intents_list = predict_class(text_input)
    response = get_response(intents_list, intents_data)
   
    
    if intents_list and 'intent' in intents_list[0]:
        doctor_info = display_doctor_info(intents_list[0]['intent'])
    
    return response, doctor_info

@app.route('/')
def index():
    return "Hello"

@app.route('/chat', methods=['POST'])
def chat():
    text_input = request.form['text_input']
    response, doctor_info = calling_the_bot(text_input)
    return jsonify({'response': response, 'doctor_info': doctor_info})


# if __name__ == '__main__':
#     app.run(debug=True)