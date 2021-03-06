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
import wikipedia as wk
import fastai
import json

# Import fast.ai Library
from fastai import *
from fastai.vision import *


app = Flask(__name__)       # Initializing our Flask application

ACCESS_TOKEN = os.environ['ACCESS_TOKEN']
VERIFY_TOKEN = os.environ['VERIFY_TOKEN']
bot = Bot(ACCESS_TOKEN)

class NotificationType(Enum):
    regular = "REGULAR"
    silent_push = "SILENT_PUSH"
    no_push = "NO_PUSH"
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
                   # x = message['message']['text'] 
                    if message['message'].get('text'):
                        x = message['message']['text']
                        if x == "Hey": 
                            quick_response(recipient_id,
                            "Hi my name is Agrotorch and I'm here to help, to start you can upload an image of a plant leaf or choose what you want to learn below ",
                                'maize', 'soybean', 'potato', 'tomato', postcard1="maize", postcard2="soybean", postcard3="potato", postcard4="tomato")
                                
                        elif x == "maize":
                            send_message(recipient_id, "https://youtu.be/AwkXRwCPHI0")
                        elif x == "soybean":
                            send_message(recipient_id, "https://youtu.be/O0TOGKSWsMs")
                        elif x == "potato":
                            send_message(recipient_id, "https://youtu.be/yy9B2ctQBt0")
                        elif x == "tomato":
                            send_message(recipient_id, "https://youtu.be/qXdw-hBiu1A")
                    
                    # if user send us a GIF, photo, video or any other non-text item
                    if message['message'].get('attachments'):
                        if message['message']['attachments'][0]['type'] == "image":
                            image_url = message["message"]["attachments"][0]["payload"]["url"]
                            pred_message = model_predict(image_url)
                            send_message(recipient_id, pred_message)
                else:
                    d_response = get_message()
                    send_message(recipient_id, d_response)
    
    return "Message Processed"

def quick_response(sender_id, message_text, title1, title2, title3, title4, postcard1="", postcard2="",postcard3="", postcard4=""):
    r = requests.post("https://graph.facebook.com/v2.6/me/messages",

                      params={"access_token": ACCESS_TOKEN},

                      headers={"Content-Type": "application/json"},

                      data=json.dumps({
                          "recipient": {"id": sender_id},
                          "messaging_type": "RESPONSE",
                          "message": {
                              "text": message_text,
                              "quick_replies": [
                                  {
                                      "content_type": "text",
                                      "title": title1,
                                      "payload": postcard1
                                  }, {
                                      "content_type": "text",
                                      "title": title2,
                                      "payload": postcard2
                                  },
                                     {
                                      "content_type": "text",
                                      "title": title3,
                                      "payload": postcard3
                                  }, {
                                      "content_type": "text",
                                      "title": title4,
                                      "payload": postcard4
                                  }
                              ]
                          }
                      }))


def verify_fb_token(token_sent):
    # take token sent by Facebook and verify it matches the verify token you sent
    # if they match, allow the request, else return an error
    if token_sent == VERIFY_TOKEN:
        return request.args.get("hub.challenge")
    return 'Invalid verification token'


def get_message():
    sample_responses = ["Type 'Hey' case sensitive to see the video menu or upload a plant leaf image to run a diagnosis",
                        "I'm not yet that smart, type 'Hey' or upload a leaf image"]
    # return selected item to the user
    return random.choice(sample_responses)


# Uses PyMessenger to send response to the user
def send_message(recipient_id, response):
    # sends user the text message provided via input response parameter
    bot.send_text_message(recipient_id, response)
    return "success"

def send_video_url(recipient_id, video_url):
    bot.send_video_url(recipient_id, video_url)
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
    # url = flask.request.args.get("url")
    response = requests.get(url)
    img = open_image(BytesIO(response.content))
    pred_class,pred_idx,outputs = learn.predict(img)
    img_message = str(pred_class)
    wiki_msg = img_message[img_message.find('___'):]
    wiki_info = wk.summary(wiki_msg, sentences = 3)
    wiki_result=(f'Diagnosis: {img_message}\n'
            f'\n'
           f'Definition: {wiki_info}')
    return wiki_result



# Add description here about this if statement.
if __name__ == "__main__":
    app.run()