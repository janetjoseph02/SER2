from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import librosa
import math
import re
import librosa
import librosa.display
import matplotlib.pyplot as plt
import unicodedata
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment import SentimentIntensityAnalyzer
app = Flask(__name__)
import wave



@app.route('/')
def hello_world():
    return render_template('ser.html')

@app.route('/predict',methods=['POST','GET'])


def save_audio():
    if request.method == "POST":
        print("pOSt")
       # getting input with name = fname in HTML form
        audio_file = request.files["Audio"]
        audio_folder = 'audio_input/'
        if not os.path.exists(audio_folder):
            os.makedirs(audio_folder)
        wav_file_name=audio_file.filename
        # Save the audio file to the audio folder/
        path=os.path.join(audio_folder, wav_file_name)
        print(path)
        audio_file.save(path)
        aud= audio_vector(path)
        # return aud
        return render_template('result.html', result=aud)
        
        
        

    



def text_input():
    if request.method == "POST":
        text_file = request.form['Text']
        text_folder = 'text_input/'
        if not os.path.exists(text_folder):
            os.makedirs(text_folder)

    # Save the input text to a text file in the text folder
        with open(os.path.join(text_folder, 'input.txt'), 'w') as f:
            f.write(text_file)
    


def audio_vector(wav_file_name):
    # Load an audio file
    audio_file_path = wav_file_name
    audio_signal, sample_rate = librosa.load(audio_file_path)
    y=audio_signal
    # Compute RMS, Harmonic Mean, and Auto-correlation
    sig_mean = np.mean(abs(y))
    sig_std= np.std(y)
    rms = librosa.feature.rms(y=y + 0.0001)[0]
    rms_mean=np.mean(rms)
    rms_std=np.std(rms)
    silence = 0
    for e in rms:
        if e <= 0.4 * np.mean(rms):
            silence += 1
    silence /= float(len(rms))
    y=audio_signal
    y_harmonic = librosa.effects.hpss(y)[0]
    harmonic_mean = np.mean(y_harmonic)

    cl = 0.45 * sig_mean
    y=audio_signal
    center_clipped = []
    for s in (y):
        if s >= cl:
            center_clipped.append(s - cl)
        elif s <= -cl:
            center_clipped.append(s + cl)
        elif np.abs(s) < cl:
            center_clipped.append(0)
    auto_corrs = librosa.core.autocorrelate(np.array(center_clipped))
    auto_corr_max=1000 * np.max(auto_corrs)/len(auto_corrs)
    auto_corr_std=np.std(auto_corrs)


    # Stack the vectors vertically
    audio_vectors = np.stack((sig_mean,sig_std, rms_mean, rms_std, silence, harmonic_mean, auto_corr_max, auto_corr_std))

    feature_list=[wav_file_name]
    feature_list.append(sig_mean)  # sig_mean
    feature_list.append(np.std(y))
    feature_list.append(np.mean(rms))  # rms_mean
    feature_list.append(np.std(rms))  # rms_std
    feature_list.append(silence)  # silence
    feature_list.append(np.mean(y_harmonic) * 1000)  # harmonic (scaled by 1000)
    feature_list.append(1000 * np.max(auto_corrs)/len(auto_corrs))  # auto_corr_max (scaled by 1000)
    feature_list.append(np.std(auto_corrs))  # auto_corr_std
    columns = ['wav_file','sig_mean', 'sig_std', 'rms_mean', 'rms_std', 'silence', 'harmonic', 'auto_corr_max', 'auto_corr_std']
    df_features = pd.DataFrame(columns=columns)

    df_features = df_features.append(pd.DataFrame(feature_list, index=columns).transpose(), ignore_index=True)
    x=prediction(df_features)
    emotion_dict = {'ang': 0,
    'hap': 1,
    'exc': 2,
    'sad': 3,
    'fru': 4,
    'fea': 5,
    'sur': 6,
    'neu': 7,
    'xxx': 8,
    'oth': 8}
    if x==0:
        return 'angry'
    elif x==1:
        return 'happy'
    elif x==2:
        return 'excited'
    elif x==3:
        return 'sad'
    elif x==4:
        return 'frustrated'
    elif x==5:
        return 'fear'
    elif x==6:
        return 'surprised'
    elif x==7:
        return 'neutral'
    elif x==8:
        return 'other'






# #text classification starts here
# def unicodeToAscii(s):
#     return ''.join(
#         c for c in unicodedata.normalize('NFD', s)
#         if unicodedata.category(c) != 'Mn'
#     )

# # Lowercase, trim, and remove non-letter characters
# def normalizeString(s):
#     s = unicodeToAscii(s.lower().strip())
#     s = re.sub(r"([.!?])", r" \1", s)
#     s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
#     return s

# useful_regex = re.compile(r'^(\w+)', re.IGNORECASE)

# file2transcriptions = {}
# transcripts_path = 'C:\\Users\\HP\\Multimodal\\serwebsite\\text_input\\Ses01F_impro01.txt'
# with open(transcripts_path, 'r') as f:
#     all_lines = f.readlines()

# for l in all_lines:
#     audio_code = useful_regex.match(l).group()
#     transcription = l.split(':')[-1].strip()
#     # assuming that all the keys would be unique and hence no `try`
#     file2transcriptions[audio_code] = transcription

# wav_file_name=['Ses01F_impro01.wav'] #name of the text file
# text_data = pd.DataFrame()
# text_data['wav_file'] = wav_file_name

# text_data['transcription'] = [normalizeString('C:\\Users\\HP\\Multimodal\\serwebsite\\text_input\\input.txt')]
# # instead of really just take the text input from user


# # Get Text Features
# tfidf = TfidfVectorizer(sublinear_tf=True, min_df=1, max_features=2464, norm='l2', encoding='latin-1', ngram_range=(1, 24), stop_words='english')

# features = tfidf.fit_transform(text_data.transcription).toarray()

# text_data = features[:text_data.shape[0]]



# #text classification ends here

def prediction(audio_data):
    # load pre-trained model from pickle file
    with open('C:/Users/HP/Multimodal/multimodal-speech-emotion-recognition/trained_models/audio/RF.pkl', 'rb') as f:
        model1 = pickle.load(f)

    # load new data
    # new_data = audio_data
    input_features = audio_data[['sig_mean', 'sig_std', 'rms_mean','rms_std','silence','harmonic','auto_corr_max','auto_corr_std']]
    print('predict')
    # make predictions
    predictions_audio = model1.predict(input_features)

    print(predictions_audio)
    return predictions_audio
    # # load pre-trained model from pickle file
    # with open('C:/Users/HP/Multimodal/multimodal-speech-emotion-recognition/trained_models/text/RF.pkl', 'rb') as f:
    #     model2 = pickle.load(f)

    # # load new data
    # new_data = text_data


    # # make predictions
    # predictions_text = model2.predict(new_data)
    # print(predictions_text)





if __name__ == '__main__':
    app.run(debug=True)
