import requests
import json

url = "http://127.0.0.1:8000/song_prediction"

song = input("Enter the song")
artist = input("Enter the artist")

input_data_for_model = {
    'song': song,
    'artist': artist
}

input_json = json.dumps(input_data_for_model)

response = requests.post(url, data=input_json)

print(response.text)