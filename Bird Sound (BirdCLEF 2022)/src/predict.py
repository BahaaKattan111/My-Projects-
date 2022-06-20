import os,warnings,cv2,librosa
import numpy as np
import pandas as pd
import torch, deep_utils

warnings.filterwarnings('ignore')

# ---submitting--- #
pred = {'row_id': [], 'target': []}

num_samples = 5
target_sample_rate = 32_000
total_length = num_samples * target_sample_rate

def _resample_if_necessary(signal, sr):
    if sr != target_sample_rate:
        signal = librosa.resample(signal, sr, target_sample_rate)
    return signal


def _mix_down_if_necessary(signal):
    if len(signal.shape) == 2:
        if signal.shape[1] > 1:
            signal = signal[:, 0]  # torch.mean(signal, dim=threshold=0.5, keepdim=True)
    return signal
test_audio_dir = '../inputs/test/'
file_list = [f.split('.')[0] for f in sorted(os.listdir(test_audio_dir))]

scored_birds = {0: 'akiapo', 1: 'aniani', 2: 'apapan', 3: 'barpet', 4: 'crehon', 5: 'elepai', 6: 'ercfra', 7: 'hawama',
                8: 'hawcre', 9: 'hawgoo', 10: 'hawhaw', 11: 'hawpet1', 12: 'houfin', 13: 'iiwi', 14: 'jabwar',
                15: 'maupar', 16: 'omao', 17: 'puaioh', 18: 'skylar', 19: 'warwhe1', 20: 'yefcan'}

path = os.listdir('run files')[0]
print(path)
best_state = torch.load(rf'..\src\run files\{path}', map_location=torch.device('cuda'))

best_model = deep_utils.CNNNetwork(21)
best_model.load_state_dict(best_state)
best_model.to('cuda')

pred_frame = {'row_id': [], 'target': []}
for afile in file_list:
    # read files
    path = test_audio_dir + afile + '.ogg'
    audio, sr = librosa.load(path)

    audio = _resample_if_necessary(audio, sr)
    audio = _mix_down_if_necessary(audio)

    # chunk audio to 5sec
    chunks = []
    # for i in range(0, len(audio), total_length):
    for i in range(0, len(audio), total_length):

        sig = audio[i:i + total_length]
        if sig.shape[0] < total_length:
            length_signal = sig.shape[0]
            num_missing_samples = int(total_length - length_signal)
            sig = np.concatenate([sig, [0] * num_missing_samples])
        mel = librosa.feature.melspectrogram(sig, hop_length=520, sr=target_sample_rate, n_mels=512)
        mel = librosa.amplitude_to_db(mel, top_db=80., ref=np.max)
        mel = np.abs(mel) / 80.
        mel = cv2.resize(mel, (128, 80))  # (64, 64)
        mel = np.array(mel).reshape((1, 128, 80))

        sig = np.zeros((3, 128, 80))
        sig[0, :, :] = mel
        sig[1, :, :] = mel
        sig[2, :, :] = mel
        chunks.append(np.array(sig, dtype=np.float32))

    test = np.stack(chunks)
    test = torch.tensor(test).to('cuda')

    with torch.no_grad():
        outputs = torch.softmax(best_model(test), 1).detach().cpu().numpy()

    for idx in range(len(outputs)):  # difference
        pred = outputs[idx]
        print(np.round(pred,3))
        chunk_end_time = (idx + 1) * 5
        for idx_2 in range(len(pred) ):

            bird = scored_birds[idx_2]
            try:
                y = pred[idx_2] > 0.5
                row_id = afile + '_' + bird + '_' + str(chunk_end_time)
                pred_frame['row_id'].append(row_id)
                pred_frame['target'].append(y)
            except:
                row_id = afile + '_' + bird + '_' + str(chunk_end_time)
                pred_frame['row_id'].append(row_id)
                pred_frame['target'].append(False)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
s = pd.DataFrame(pred_frame, columns=['row_id', 'target'])
print(s[s['target'] == True])
print(s)
