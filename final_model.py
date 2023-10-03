import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
import numpy as np
import librosa
from collections import Counter
import scipy

def reshape(d,shape=(26,65)):
    d = d.reshape(shape)
    d = np.expand_dims(d,axis=-1)
    return d
def get_frame_mfccs(path):
    audio, sr = librosa.load(path)
    frames = librosa.util.frame(audio, frame_length=sr*3, hop_length=sr*3)
    frame_mfccs = []
    for i in range(frames.shape[1]):
        mfccs = librosa.feature.mfcc(y=frames[:,i],sr=sr,n_mfcc=13,hop_length=512,n_fft=2048)
        frame_mfccs.append(mfccs)
    return frame_mfccs

audio_model = tf.keras.models.load_model("# audio model path")


classes = ['Calm','Joy','Power','Romance','Sorrow']
pred_song = []

def final_pred(path):
    fmccs = get_frame_mfccs(path)
    for frame in fmccs:
        pred_song.append(frame)
                
    pred_song = np.array(pred_song)
                
    pred_value = np.array(np.array([reshape(x) for x in pred_song]))
    pred_value.shape
    pred = audio_model.predict(pred_value)

    preds = []
    
    for i in pred:
        out = np.argmax(i)
        preds.append(out)

    final_prediction = {'Romance':0,'Sorrow':0,'Power':0,'Joy':0,'Calm':0}
    preds = dict(Counter(preds))
    for i, val in preds.items():
        final_prediction[classes[i]] = (val / len(pred))
    
    return final_prediction

pred = final_pred("# .wav file path to predict the emotion")
print(pred)