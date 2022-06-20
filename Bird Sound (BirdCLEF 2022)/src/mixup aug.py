import os

import audiomentations
import torch
import torchvision, cv2
import numpy as np
import pandas as pd
import tqdm
import warnings
import librosa

warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 2000)
train_metadata = pd.read_csv('../inputs/train_metadata.csv')
print(train_metadata[train_metadata['scientific_name'] == 'Loxops mana'].sample(frac=1.))

# --- AUG  with Noises mix (water, wind, thunder) --- #
'''water, water_sr = librosa.load('water_noise_long.wav')
wind, wind_sr = librosa.load('wind_noise_long.wav')
water_wind, water_wind_sr = librosa.load('water&wind_noise_long.wav')
thunder, thunder_sr = librosa.load('thunder_noise_long.wav')


for class_dir in tqdm.tqdm(os.listdir('../inputs/train')):
    if class_dir in label_scores:
        label_list = []
        print(class_dir)
        if len(train_metadata[train_metadata['primary_label'] == class_dir])<200:
            info = train_metadata[(train_metadata['primary_label'] == class_dir) & (train_metadata['rating'] > 3.0)]

            for file in info['filename']:
                try:
                    sig, sr = librosa.load(f"../inputs/train/{file}")
                    sig_duration = sig.shape[0]
                    for noise in [water, wind, thunder, water_wind]:
                        cuted_noise = noise[:sig_duration]
                        aug = (cuted_noise + sig) / 2
                        label_list.append(np.array(aug, dtype=np.float16))
                except:
                    print('not saved:',file)
                    continue
            pd.to_pickle(label_list, f"../inputs/aug_audio/aug_{class_dir}.pkl")
print('augmenting mixup done...')'''


