import tensorflow as tf
# # Hide GPU from visible devices
import tensorflow_text as text
import requests
from fastapi import FastAPI
from pydantic import BaseModel
from util import genius, bert_model
from collections import Counter
import numpy as np
import string
import re
import uvicorn
import json

table = str.maketrans("","", string.punctuation)
index = ['Romance','Sorrow','Power','Joy','Calm']
error_msg = "The server is currently busy. Please try again Later."
msg = "no lyrics found"

app = FastAPI()

# loading the saved model
# bert_model = tf.keras.models.load_model("D:\\Git projects\\MLApp\\ML_deployment\\App_test\\models\\newmodel")


def predict_emotion(text):
    prediction = bert_model.predict([text])
    return np.argmax(prediction)


def get_prediction_probability(text):
    probability = bert_model.predict([text])
    return probability[0]


def get_song_lyrics(title, artist):
    try:
        song = genius.search_song(title, artist)
    except:
        return error_msg

    if(song != None):
        return song.lyrics
    else:
        return "no lyrics found"


def text_preprocess(uncln):
    pattern = 'Lyrics([ \w]{1,})'
    uncln = re.sub(r'[^\w\s]', ' ', uncln, flags=re.MULTILINE)
    uncln = re.sub("[ \n]+", " ", uncln)
    uncln = re.sub(r'[0-9]'," ",uncln)
    uncln = re.sub(r' +'," ",uncln)
    lyric_list = re.findall(pattern, uncln)

    if len(lyric_list) != 0:
        return lyric_list[0].strip().lower()

    return uncln.strip().lower()

def text_preprocess_Level1(text):
    text = re.sub(r'(\d{1,}|embed)', '', text)
    text = re.sub(r'you might also likeembed', '', text)

    return text.strip().lower()

def remove_duplicates(text):
    text = text.split('\n')
    sentences = Counter(text)
    non_duplicate_text = "\n".join(sentences.keys())

    return non_duplicate_text

def remove_punct(text):
    return text.translate(table)


def data_process(song, artist):
    uncln_lyrics = get_song_lyrics(song, artist)
    #song_to_display = display_song(uncln_lyrics)

    if uncln_lyrics == msg:
        return msg, msg
    elif uncln_lyrics == error_msg:
        return error_msg, error_msg
    
    lyrics = remove_duplicates(uncln_lyrics)
    lyrics = text_preprocess(lyrics)
    lyrics = text_preprocess_Level1(lyrics)
    lyrics = remove_punct(lyrics)

    print(lyrics)

    # prediction = predict_emotion(lyrics)
    probability = get_prediction_probability(lyrics)
    prediction = np.argmax(probability)

    return index[prediction], probability

class model_input(BaseModel):
    song: str
    artist: str

@app.post('/song_prediction')
def song_pred(input_parameters : model_input):

    input_data = input_parameters.json()
    input_dictionary = json.loads(input_data)
    print(input_dictionary)

    song = input_dictionary['song']
    artist = input_dictionary['artist']

    prediction, probability = data_process(song, artist)

    if (prediction == error_msg or prediction == msg):
        return prediction
    else:
        prediction_dict = {
            'final_emotion': prediction,
            'Romance': str(probability[0]),
            'Sorrow': str(probability[1]),
            'Power': str(probability[2]),
            'Joy': str(probability[3]),
            'Calm': str(probability[4])
        }
        return prediction_dict
    
