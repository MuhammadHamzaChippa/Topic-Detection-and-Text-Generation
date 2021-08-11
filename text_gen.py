from flask import Flask , render_template , request
from flask_pymongo import PyMongo
from pymongo import MongoClient

import fastai 
from fastai import *
import pathlib
import pickle
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
from fastai.text.all import *

app = Flask(__name__)
app.config['MONGO_URI'] = 'mongodb://localhost:27017/posts'
mongo = PyMongo(app)

# Connection to the MongoDB Server
mongoClient = MongoClient ('mongodb://localhost:27017/posts')
# Connection to the database
db = mongoClient.posts
#Collection
collection = db.record
generated_text = collection.find()

@app.route('/', methods = ['GET', 'POST'])
def home():
    global cat 
    cat = ''
    global clean_text
    clean_text = '' 
    if request.method == 'POST':
        # lodaing classification model 
        classifier = load_learner('reddit_classifier.pkl')
        # classification 
        user_input  = request.form.get('post_text')
        cat,_,probs = classifier.predict(user_input)
        cat = cat.capitalize()
        print(cat)

        # generate text
        clean_text = generate_text(user_input)

        # generate text again if the same text found in the database
        for document in generated_text:
            if (clean_text == document['Generated_Text']):
                clean_text = generate_text(user_input)
             
        # storing the generated text and category in database
        mongo.db.record.insert([dict(Category = cat , Generated_Text = clean_text)])
        
    return render_template('index_main.html' , prediction = cat , comment = clean_text)

def generate_text(text_input):
    # initialize tokenizer and model from pretrained GPT2 model
    tokenizer = pickle.load(open('gpt2_tokenizer.pkl','rb'))
    model = pickle.load(open('gpt2_language.pkl','rb'))
    # encode user_input
    inputs = tokenizer.encode(text_input, return_tensors='pt')
    # we pass a maximum output length of 200 tokens
    outputs = model.generate(inputs, max_length=100, do_sample=True)
    #decode user_input
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    text = text.replace("\n", "")
    return text

if __name__ == '__main__':
    app.run(debug= True)
