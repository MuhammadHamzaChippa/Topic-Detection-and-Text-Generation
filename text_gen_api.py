#importing libraries
from flask import Flask 
from flask_restplus import Api, Resource, fields 
# from flask_pymongo import PyMongo
from pymongo import MongoClient
import fastai 
from fastai import *
import pathlib
import pickle
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
from fastai.text.all import *

# CONFIGURING MONGODB 
# Connection to the MongoDB Server
mongoClient = MongoClient ('mongodb+srv://hamza:mongodb@mongodb-heroku.igizt.mongodb.net/myFirstDatabase?retryWrites=true&w=majority')
# Connection to the database
db = mongoClient.posts
#Collection
collection = db.record
# reading all the generated text 
generated_text = collection.find()

app = Flask(__name__)
api = Api(app,version='1.0', title='Text Generation',
    description='A simple Topic Detection and Text Generation API')

api = api.namespace('TEXTGEN', description='Topic Detection and Text Generation')

a_text = api.model('USER_INPUT', {'Input' : fields.String('Enter the text')})


@api.route('/generated_text')
class Text_Gen(Resource):
    def get(self):
        '''Show all the Generated Text in Database'''
    
        db1 = mongoClient.posts
        collection1 = db1.record
        generated_text1 = collection1.find()
        result = []
        for document in generated_text1:
            text = {"Category" : document['Category'], "Generated_Text" : document['Generated_Text']}
            result.append(text)
        return result

    @api.expect(a_text)
    def post(self):
        '''Detect Topic and Generate Text'''
        user_input = api.payload
        ml_result = ml_model(user_input['Input']) 
        # adding the generated text and category in mongodb database
        collection.insert(ml_result)
        return ({"Category" : ml_result['Category'], "Generated_Text" : ml_result['Generated_Text']}), 201 

def ml_model(input):
    # lodaing classification model 
    classifier = load_learner('reddit_classifier.pkl')
    # classification  
    cat,_,probs = classifier.predict(input)
    cat = cat.capitalize()
    print(cat)

    # generate text
    clean_text = generate_text(input)
    # generate text again if the same text found in the database
    for document in generated_text:
        if (clean_text == document['Generated_Text']):
            clean_text = generate_text(input)
    prediction = {"Category" : cat, "Generated_Text" : clean_text}
    return prediction
        


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
    text = text.replace("\"" , "")
    return text


if __name__ == '__main__':
    app.run(debug=True)