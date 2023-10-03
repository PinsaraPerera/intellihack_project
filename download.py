from subprocess import run
import wave
from pydub import AudioSegment
import tensorflow as tf
import downloader
tf.config.set_visible_devices([], 'GPU')
from final_model import final_pred
from ffmpy import FFmpeg
import os

def trim(dir):
    wav_file = wave.open(dir, 'r')
    sample_rate = wav_file.getframerate()
    num_channels = wav_file.getnchannels()
    audio_data = AudioSegment.from_wav(dir)
    trimmed_data = audio_data[0:180000]
    trimmed_data.export(dir)
    wav_file.close()

def get_data(name):
    Path = downloader.get_path()
    print(Path)
    input_file = Path
    output_file = "# save file location\{}.wav".format(name)
    print(output_file)
    run(["ffmpeg", "-i", input_file, "-vn", "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2", output_file])
    print("----------")
    trim(output_file)
    
    os.remove(Path)
    return output_file
    

def main(title,artist):
    try:
        request = title + " " + artist
        downloader.main(request)
        path = get_data(title)
        print(path)
        return final_pred(path) 
    except:
        print('Cannot Access', title)
        

pred = main("fix you","coldplay")
print(pred)