import random
from flask import Flask, request
from pymessenger.bot import Bot
import requests
from io import BytesIO
import flask
import sys
import os
import glob
import re
from pathlib import Path

import fastai

# Import fast.ai Library
from fastai import *
from fastai.vision import *


app = Flask(__name__)       # Initializing our Flask application
ACCESS_TOKEN = 'EAAEmm5v8LIQBAGAgnG1pj740BTp1XPKp2eSdkwEngNZC6WXZC6GiZBaq9rF20bZBlcy2kW5qNKbVBraCemZCdTZCMtYwx9sF51MJ7gs9whYe1uDayWL27kmaouyCCPZCCixbDkUhirk4khUy5tZB9PLorvYjomnSyNaVf3FgRa6CgwZDZD'
VERIFY_TOKEN = 'Yeayyeahyeahs'
bot = Bot(ACCESS_TOKEN)

# Importing standard route and two requst types: GET and POST.
# We will receive messages that Facebook sends our bot at this endpoint
@app.route('/', methods=['GET', 'POST'])
def receive_message():
    if request.method == 'GET':
        # Before allowing people to message your bot Facebook has implemented a verify token
        # that confirms all requests that your bot receives came from Facebook.
        token_sent = request.args.get("hub.verify_token")
        return verify_fb_token(token_sent)
    # If the request was not GET, it  must be POSTand we can just proceed with sending a message
    # back to user
    else:
            # get whatever message a user sent the bot
        output = request.get_json()
        for event in output['entry']:
            messaging = event['messaging']
            for message in messaging:
                if message.get('message'):
                    # Facebook Messenger ID for user so we know where to send response back to
                    recipient_id = message['sender']['id']
                    if message['message'].get('text'):
                        response_sent_text = get_message()
                        send_message(recipient_id, response_sent_text)
                    # if user send us a GIF, photo, video or any other non-text item
                    if message['message'].get('attachments'):
                        if message['message']['attachments'][0]['type'] == "image":
                            image_url = message["message"]["attachments"][0]["payload"]["url"]
                            pred_message = model_predict(image_url)
                            send_message(recipient_id, pred_message)
    
    return "Message Processed"


def verify_fb_token(token_sent):
    # take token sent by Facebook and verify it matches the verify token you sent
    # if they match, allow the request, else return an error
    if token_sent == VERIFY_TOKEN:
        return request.args.get("hub.challenge")
    return 'Invalid verification token'


def get_message():
    sample_responses = ["You are so cool", "We're proud of you",
                        "Keep on being cool!", "We're greatful to know you :)"]
    # return selected item to the user
    return random.choice(sample_responses)


# Uses PyMessenger to send response to the user
def send_message(recipient_id, response):
    # sends user the text message provided via input response parameter
    bot.send_text_message(recipient_id, response)
    return "success"

path = Path("path")
classes = ['Apple___Apple_scab','Apple___Black_rot','Apple___Cedar_apple_rust'
,'Apple___healthy','Blueberry___healthy','Cherry_(including_sour)___Powdery_mildew',
'Cherry_(including_sour)___healthy','Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
'Corn_(maize)___Common_rust_','Corn_(maize)___Northern_Leaf_Blight',
'Corn_(maize)___healthy','Grape___Black_rot','Grape___Esca_(Black_Measles)',
'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
'Grape___healthy','Orange___Haunglongbing_(Citrus_greening)',
'Peach___Bacterial_spot','Peach___healthy','Pepper,_bell___Bacterial_spot',
'Pepper,_bell___healthy','Potato___Early_blight','Potato___Late_blight',
'Potato___healthy','Raspberry___healthy','Soybean___healthy',
'Squash___Powdery_mildew','Strawberry___Leaf_scorch','Strawberry___healthy',
'Tomato___Bacterial_spot','Tomato___Early_blight','Tomato___Late_blight',
'Tomato___Leaf_Mold','Tomato___Septoria_leaf_spot',
'Tomato___Spider_mites Two-spotted_spider_mite','Tomato___Target_Spot',
'Tomato___Tomato_Yellow_Leaf_Curl_Virus','Tomato___Tomato_mosaic_virus',
'Tomato___healthy','background']



data2 = ImageDataBunch.single_from_classes(path, classes, ds_tfms=get_transforms(), size=224).normalize(imagenet_stats)

path1 = Path("./models")
learn = load_learner(path1, 'export_model.pkl')


@app.route('/analyse', methods=['GET', 'POST'])
def model_predict(url):
    """
       model_predict will return the preprocessed image
    """
    #url = flask.request.args.get("url")
    response = requests.get(url)
    img = open_image(BytesIO(response.content))
    pred_class,pred_idx,outputs = learn.predict(img)
    return str(pred_class)


# Add description here about this if statement.
if __name__ == "__main__":
    app.run()