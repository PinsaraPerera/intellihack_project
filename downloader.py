from pytube import YouTube
import pandas
import os
from youtubesearchpython import VideosSearch
import argparse

AUDIO_DOWNLOAD_DIR = "# audion download path"

def YoutubeAudioDownload(video_url):
    global FILE_PATH
    video = YouTube(video_url, use_oauth=True, allow_oauth_cache=True)
    audio = video.streams.get_audio_only().download(AUDIO_DOWNLOAD_DIR)
    try:
        FILE_PATH = os.path.abspath(audio)
        
    except:
        exit()
    
def get_link(name):
    videosSearch = VideosSearch(name, limit = 2)
    results = videosSearch.result()
    return results["result"][0]['link']

def get_path():
    return FILE_PATH
    
def main(name):
    name = name + ' audio only'
    link = get_link(name)
    if link:
        YoutubeAudioDownload(link)
        # ytmusic(name)
    else:
        None
