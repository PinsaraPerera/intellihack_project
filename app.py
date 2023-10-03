import tensorflow as tf
import tensorflow_text as text

# Import the API
from util import genius, bert_model
# import download 
import streamlit as st
import pandas as pd
import numpy as np
import time
import base64
import string
import streamlit.components.v1 as components
from collections import Counter

# regular expressions
import re
# punctuations
path = "# path to your model"
df = pd.read_csv("Resources/max_emotionnnnnnn.csv")
table = str.maketrans("","", string.punctuation)
index = ['Romance','Sorrow','Power','Joy','Calm']


st.set_page_config(
    page_title="Song analysis",
    layout="wide",
    initial_sidebar_state="collapsed",
)


@st.cache_data
def get_image_as_base64(file):
    with open(file, "rb") as t:
        data = t.read()
    return base64.b64encode(data).decode()

# @st.cache_data
# def get_essentials_load(path):
#     model = tf.keras.models.load_model(path)
#     return model


img = get_image_as_base64("Resources/background_1.jpeg")
side_bar_image = get_image_as_base64("Resources/wall.jpg")
# bert_model = get_essentials_load(path)

with open('style.css') as f:
    style = f.read()
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"]{{
background-image: url("data:image/jpg;base64,{img}");
background-size: cover;
background-attachment: local;
}}
div.e1fqkh3o3{{
background-image: url("data:image/jpg;base64,{side_bar_image}");
background-size: cover;
background-attachment: local;
}}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)


genius.verbose = False
genius.remove_section_headers = True
genius.skip_non_songs = True

# bert_model = tf.keras.models.load_model("# path to model")
print("loading....")
error_msg = "The server is currently busy. Please try again Later."
msg = "no lyrics found"



def predict_emotion(text):
    prediction = bert_model.predict([text])
    return np.argmax(prediction)


def get_prediction_probability(text):
    probability = bert_model.predict([text])
    print(probability)
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

def score_cal(probability):
    score = []
    score_per = []
    MAX = 425
    for i in probability:
        score_per.append(int(i*100))
        score.append(int(MAX - MAX*i) )
        
    return score, score_per

def song_recommend(emotion):
    sample = df[df['emotion'] == emotion.lower()].sample(11)
    sample = sample.sort_values(by=[emotion.lower()],ascending=False)
    sample = sample.reset_index(drop=True)
    return sample[['Artist','Title']]

def main():
    st.title("Song Emotion Classifier Application")
    menu = ["Home", "Recommendations", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.subheader("Emotion of the song")

        with st.form(key='emotion_detect_form'):
            col1, col2 = st.columns(2)

            with col1:
                song = st.text_input(
                    "Track Name", placeholder="Enter the song")

            with col2:
                artist = st.text_input(
                    "Artist", placeholder="Enter the artist name")

            submit_text = st.form_submit_button(label="Submit")

        if submit_text:
            col1, col2 = st.columns(2)

            prediction, probability = data_process(song, artist)
            # audio model join
            # if(song != None and artist != None):
            #     pred_dict = download.main(song, artist)

            with st.spinner('Wait for it...'):
                time.sleep(3)

            with col1:
                st.success("Entered Song Name")
                st.write(song)
                st.success("Artist Name")
                st.write(artist)

            with col2:
                if (prediction == error_msg or prediction == msg):
                    #e = RuntimeError(prediction)
                    st.error(prediction)
                else:
                    st.success("Prediction")
                    st.write(prediction)

                    st.success("Prediction Probability")
                    # st.write(probability)
                    pf = pd.DataFrame(probability, index=['Romance','Sorrow','Power','Joy','Calm'], columns=['Predictions'])
                    st.write(pf.T)
            
            if (prediction == error_msg or prediction == msg):
                    pass
            else:
                score, percentage = score_cal(probability)

                col3, col4, col5, col6, col7 = st.columns(5)
                with col3:
                    st.write("Romance")
                    components.html(f'''
                                        <style>
                                        {style}
                                        @keyframes anim {{
                                            100%{{
                                                stroke-dashoffset: {score[0]};
                                            }}
                                        }}
                                        </style>
                                        <div class="emotion">
                                            <div class="outer">
                                                <div class="inner">
                                                    <div id="number"></div>
                                                </div>
                                            </div>

                                            <svg class="icon"
                                                xmlns="http://www.w3.org/2000/svg"
                                                version="1.1"
                                                width="160px"
                                                height="160px"
                                            >
                                                <defs>
                                                    <linearGradient id="GradientColor">
                                                        <stop offset="0%" stop-color="#e91e63" />
                                                        <stop offset="100%" stop-color="#673ab7" />
                                                    </linearGradient>
                                                </defs>
                                                <circle cx="80" cy="80" r="65" stroke-linecap="round" />
                                            </svg>
                                        </div>
                                        <script>
                                            let number = document.getElementById("number");
                                            let counter = -1;
                                            setInterval(() => {{
                                            if (counter == {percentage[0]}) {{
                                                clearInterval();
                                            }} else {{
                                                counter += 1;
                                                number.innerHTML = counter + "%";
                                            }}
                                            }}, 30);
                                        </script>
                                    ''',height=200, width=200)
                    st.markdown(f'<style>{style}</style>', unsafe_allow_html=True)
                with col4:
                    st.write("Sorrow")
                    components.html(f'''
                                        <style>
                                        {style}
                                        @keyframes anim {{
                                            100%{{
                                                stroke-dashoffset: {score[1]};
                                            }}
                                        }}
                                        </style>
                                        <div class="emotion">
                                            <div class="outer">
                                                <div class="inner">
                                                    <div id="number"></div>
                                                </div>
                                            </div>

                                            <svg class="icon"
                                                xmlns="http://www.w3.org/2000/svg"
                                                version="1.1"
                                                width="160px"
                                                height="160px"
                                            >
                                                <defs>
                                                    <linearGradient id="GradientColor">
                                                        <stop offset="0%" stop-color="#e91e63" />
                                                        <stop offset="100%" stop-color="#673ab7" />
                                                    </linearGradient>
                                                </defs>
                                                <circle cx="80" cy="80" r="65" stroke-linecap="round" />
                                            </svg>
                                        </div>
                                        <script>
                                            let number = document.getElementById("number");
                                            let counter = -1;
                                            setInterval(() => {{
                                            if (counter == {percentage[1]}) {{
                                                clearInterval();
                                            }} else {{
                                                counter += 1;
                                                number.innerHTML = counter + "%";
                                            }}
                                            }}, 30);
                                        </script>
                                    ''',height=200, width=200)
                    st.markdown(f'<style>{style}</style>', unsafe_allow_html=True)
                with col5:                        
                    st.write("Power")
                    components.html(f'''
                                        <style>
                                        {style}
                                        @keyframes anim {{
                                            100%{{
                                                stroke-dashoffset: {score[2]};
                                            }}
                                        }}
                                        </style>
                                        <div class="emotion">
                                            <div class="outer">
                                                <div class="inner">
                                                    <div id="number"></div>
                                                </div>
                                            </div>

                                            <svg class="icon"
                                                xmlns="http://www.w3.org/2000/svg"
                                                version="1.1"
                                                width="160px"
                                                height="160px"
                                            >
                                                <defs>
                                                    <linearGradient id="GradientColor">
                                                        <stop offset="0%" stop-color="#e91e63" />
                                                        <stop offset="100%" stop-color="#673ab7" />
                                                    </linearGradient>
                                                </defs>
                                                <circle cx="80" cy="80" r="65" stroke-linecap="round" />
                                            </svg>
                                        </div>
                                        <script>
                                            let number = document.getElementById("number");
                                            let counter = -1;
                                            setInterval(() => {{
                                            if (counter == {percentage[2]}) {{
                                                clearInterval();
                                            }} else {{
                                                counter += 1;
                                                number.innerHTML = counter + "%";
                                            }}
                                            }}, 30);
                                        </script>
                                    ''',height=200, width=200)
                    st.markdown(f'<style>{style}</style>', unsafe_allow_html=True)
                with col6:
                    st.write("Joy")
                    components.html(f'''
                                        <style>
                                        {style}
                                        @keyframes anim {{
                                            100%{{
                                                stroke-dashoffset: {score[3]};
                                            }}
                                        }}
                                        </style>
                                        <div class="emotion">
                                            <div class="outer">
                                                <div class="inner">
                                                    <div id="number"></div>
                                                </div>
                                            </div>

                                            <svg class="icon"
                                                xmlns="http://www.w3.org/2000/svg"
                                                version="1.1"
                                                width="160px"
                                                height="160px"
                                            >
                                                <defs>
                                                    <linearGradient id="GradientColor">
                                                        <stop offset="0%" stop-color="#e91e63" />
                                                        <stop offset="100%" stop-color="#673ab7" />
                                                    </linearGradient>
                                                </defs>
                                                <circle cx="80" cy="80" r="65" stroke-linecap="round" />
                                            </svg>
                                        </div>
                                        <script>
                                            let number = document.getElementById("number");
                                            let counter = -1;
                                            setInterval(() => {{
                                            if (counter == {percentage[3]}) {{
                                                clearInterval();
                                            }} else {{
                                                counter += 1;
                                                number.innerHTML = counter + "%";
                                            }}
                                            }}, 30);
                                        </script>
                                    ''',height=200, width=200)
                    st.markdown(f'<style>{style}</style>', unsafe_allow_html=True)
                with col7:
                    st.write("Calm")
                    components.html(f'''
                                        <style>
                                        {style}
                                        @keyframes anim {{
                                            100%{{
                                                stroke-dashoffset: {score[4]};
                                            }}
                                        }}
                                        </style>
                                        <div class="emotion">
                                            <div class="outer">
                                                <div class="inner">
                                                    <div id="number"></div>
                                                </div>
                                            </div>

                                            <svg class="icon"
                                                xmlns="http://www.w3.org/2000/svg"
                                                version="1.1"
                                                width="160px"
                                                height="160px"
                                            >
                                                <defs>
                                                    <linearGradient id="GradientColor">
                                                        <stop offset="0%" stop-color="#e91e63" />
                                                        <stop offset="100%" stop-color="#673ab7" />
                                                    </linearGradient>
                                                </defs>
                                                <circle cx="80" cy="80" r="65" stroke-linecap="round" />
                                            </svg>
                                        </div>
                                        <script>
                                            let number = document.getElementById("number");
                                            let counter = -1;
                                            setInterval(() => {{
                                            if (counter == {percentage[4]}) {{
                                                clearInterval();
                                            }} else {{
                                                counter += 1;
                                                number.innerHTML = counter + "%";
                                            }}
                                            }}, 30);
                                        </script>
                                    ''',height=200, width=200)
                    st.markdown(f'<style>{style}</style>', unsafe_allow_html=True)

                st.write("Recommended Songs")
                song_list = song_recommend(prediction.lower())
                st.table(song_list[1:])

                # st.write("Audio Model Output")
                # st.table(pred_dict)
     

    elif choice == "Recommendations":
        st.subheader("Show the stats")

    else:
        st.subheader("About Us")


# components.html('''
# <script>
# const element = window.parent.document.querySelectorAll('.e1tzin5v3')[1]
# let children = element.childNodes
# children[0].classList.add("joy")
# children[1].classList.add("romance")
# children[2].classList.add("sorrow")
# children[3].classList.add("calm")
# children[4].classList.add("power")
# </script>
# ''')

if __name__ == "__main__":
    main()
