import pandas as pd
import numpy as np
import tqdm, cv2, os, warnings, librosa

from audiomentations import PitchShift, Compose, TimeStretch, Shift

warnings.filterwarnings('ignore')
device = 'cuda'


def _resample_if_necessary(signal, sr):
    if sr != target_sample_rate:
        signal = librosa.resample(signal, sr, target_sample_rate)
    return signal


def _mix_down_if_necessary(signal):
    if len(signal.shape) == 2:
        if signal.shape[1] > 1:
            signal = signal[:, 0]  # torch.mean(signal, dim=threshold=0.5, keepdim=True)
    return signal


num_samples = 5
target_sample_rate = 32_000
total_length = target_sample_rate * num_samples
##
processed_data = []
processed_labels = []
##
path = '../inputs/train'
label_scores = pd.read_json('../inputs/scored_birds.json')[0]
train_metadata = pd.read_csv(f'../inputs/train_metadata.csv')

# aug soundd with quality >= 3.0 (best=5 worst=1)(option)
'''for col in train_metadata['primary_label'].unique():
    if col not in label_scores.values:
        train_metadata.drop(index=train_metadata[train_metadata['primary_label'] == col].index, inplace=True)
train_metadata = train_metadata[train_metadata['rating'] >= 3.0]
'''

# there is speed aug files but they are not processed in the project ,they are from the competition's notebook

pitchShift = PitchShift(p=1.)
shift = Shift(p=1.)
timeStretch = TimeStretch(p=1.)
full_aug = Compose([pitchShift, shift, timeStretch])


def aug_audio(audio_path):
    # process
    audiofile, sr = librosa.load(audio_path)
    audiofile = _resample_if_necessary(audiofile, sr)
    audiofile = _mix_down_if_necessary(audiofile)

    # aug
    PitchShift_signal = pitchShift(audiofile, 32000)
    Shift_signal = shift(audiofile, 32000)
    full_aug_signal = full_aug(audiofile, sample_rate=32000)
    TimeStretch_signal = timeStretch(audiofile, 32000)
    # slow_audio = speed_transform(audiofile, 0.9)
    # fast_audio = speed_transform(audiofile, threshold=0.5.threshold=0.5)
    return [PitchShift_signal, Shift_signal, TimeStretch_signal, full_aug_signal]  # , slow_audio, fast_audio]


# in case processing breaked or stopped, you can know which classes are not processed
# process them, replace 'not_processed_labels' instead 'os.listdir('../inputs/train')' (option)

'''
aug_labels = pd.Series(pd.read_pickle('../inputs/new_aug_processed_labels.pkl'))

not_processed_dir = set(os.listdir(path)).difference(set(aug_labels.unique()))
not_processed_labels = set(label_scores).intersection(not_processed_dir)
print(not_processed_labels)'''

for col in train_metadata['primary_label'].unique():
    if col not in label_scores.values:
        train_metadata.drop(index=train_metadata[train_metadata['primary_label'] == col].index, inplace=True)
path = r'C:\Users\Lenovo\PycharmProjects\pythonProject\bird sound\inputs\512 data\orig data'
import joblib

for class_dir in tqdm.tqdm(os.listdir('../inputs/train')):
    print(class_dir)
    # info = train_metadata[train_metadata['primary_label'] == class_dir]# in case quality filter used
    w = 0  # num samples  for current class
    for file in os.listdir(f'../inputs/train/{class_dir}'):
        try:
            audios = aug_audio(fr'../inputs/train/{class_dir}/{file}')
            for audio in audios:
                for i in range(0, len(audio), total_length):
                    w += 1
                    sig = audio[i:i + total_length]  # 5-sec chunk

                    # if chunk is less than 5-sec,which is the last 5-sec usually, pad it
                    if sig.shape[0] < total_length:
                        length_signal = sig.shape[0]
                        num_missing_samples = int(total_length - length_signal)
                        sig = np.concatenate([sig, [0] * num_missing_samples])

                    # get melspectrogram
                    mel = librosa.feature.melspectrogram(sig, hop_length=520, sr=target_sample_rate, n_mels=512)
                    mel = librosa.amplitude_to_db(mel, top_db=80., ref=np.max)

                    # normalize and resize
                    mel = np.abs(mel) / 80.
                    mel = cv2.resize(mel, (128, 80))  # (64, 64)

                    processed_data.append(np.array(mel, dtype=np.float16))
                    processed_labels.append(class_dir)

                    # save after finishing every class
                    joblib.dump(processed_labels, rf'{path}\processed_labels_.pkl')
                    joblib.dump(processed_data, rf'{path}\processed_data_.pkl')
                # in case a class has moere than 5000 sample(5-sec chunks),
                # break and go to next class to avoid over sample

                if w > 5_000:
                    break
        except:
            print(f'error: {path}/{file}')
        if w > 5_000:
            break
    if w > 5_000:
        break
    print(f'{class_dir} after={w}')

print('len processed data:', len(processed_data))
