import os
import sys
import collections
sys.path.append(os.path.abspath('../Codebook'))

import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm

from feature_extraction.extract_mel_spectrogram import *

def get_spectrogram(y_id, audio_path, save_path, start_time, duration_sec=10):
    wav, sr = librosa.load(audio_path, sr=22050)
    
    length = sr*duration_sec
    y = np.zeros(length)
    if wav.shape[0] < length:
        y[:len(wav)] = wav
    else:
        y = wav[start_time:start_time+length]
    
    mel_spec = TRANSFORMS(y)
    np.save(os.path.join(save_path, y_id + '_mel.npy'), mel_spec)
    
phase_name = 'train'
audio_root = '/media/daftpunk2/home/jakeoneijk/221008_audio_caps/audiocaps_audio_dataset'
save_root = 'caps_full'

f_dict = {}
audio_path = os.path.join(audio_root, phase_name)
save_path = os.path.join(save_root, phase_name)

for f in os.listdir(audio_path):
    f_dict[f.split(']_')[0].strip('[')] = os.path.join(audio_path, f)

ordered_f_dict = collections.OrderedDict(sorted(f_dict.items()))
print('-'*5+'list sorting done!')

df = pd.read_csv(f'./{phase_name}.csv')

err_cnt = 0
for key, value in tqdm(list(ordered_f_dict.items())[45000:]):
    start_time = df.loc[df['youtube_id'] == key]['start_time'].item() 
    try:
        get_spectrogram(key, f_dict[key], save_path, start_time)
    except:
        print('Cannot load audio')
        err_cnt += 1
        
print('Error count: ', err_cnt)