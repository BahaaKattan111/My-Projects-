# import pickle
import joblib
import pandas as pd
from torch.utils.data import DataLoader
import torch, torchaudio
# from sklearn.model_selection import train_test_split
import numpy as np
# from sklearn.preprocessing import StandardScaler
import deep_utils

pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 2000)


def seed_everything(seed=10):
    import random, os
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


torch.backends.cudnn.benchmark = True  # for fast computing
seed_everything()

# ------- # AUG DATA # ------ #

print('concatenating training data and labels...')
# ---Competition Data aug---
data = np.array(joblib.load(r'..\inputs\512 data\aug data\processed_data.pkl'), dtype=np.float16)# (dtype=np.float16 is to reduce array/pic size)
labels = pd.Series(joblib.load(r'..\inputs\512 data\aug data\processed_labels.pkl')).reset_index(drop=True)# (.reset_index(drop=True) so we can concat)

# ---Uploaded(from internet) Data aug---
data_2 = np.array(joblib.load(r'..\inputs\512 data\aug data\aug_uploaded_processed_data.pkl'), dtype=np.float16)
labels_2 = pd.Series(joblib.load(r'..\inputs\512 data\aug data\aug_uploaded_processed_labels.pkl')).reset_index(drop=True)

# ---Concatenate---
data = np.concatenate([data, data_2] ).reshape((-1, 128 * 80)) # because we can't concat without reduceing the dim ,(a, b, c, d) -> (a, b)
labels = pd.concat([labels, labels_2]).reset_index(drop=True)

df = pd.concat([pd.DataFrame(data), labels], axis=1).sample(frac=1)
del data, labels
# after concat, get back dim
data = np.array(df.iloc[:, :-1], dtype=np.float16).reshape((-1, 1, 128 , 80))
labels = df.iloc[:, -1]
del df

# ------- # ORIGINAL DATA # ------ #
print('concatenating validation data and labels...')
# ---Competition Data orig---
val_data = np.array(joblib.load(r'..\inputs\512 data\orig data\processed_data_full.pkl'), dtype=np.float16) # .reshape((-1, 128 * 80 ))
val_labels = pd.Series(joblib.load(r'..\inputs\512 data\orig data\processed_labels_full.pkl'))

# ---Uploaded(from internet) Data orig---
val_data_2 = np.array(joblib.load(r'..\inputs\512 data\orig data\uploaded_processed_data.pkl'), dtype=np.float16)# .reshape((-1, 128 * 80 ))
val_labels_2 = pd.Series(joblib.load(r'..\inputs\512 data\orig data\uploaded_processed_labels.pkl'))

# concat
val_data = np.concatenate([val_data, val_data_2]).reshape((-1, 128 * 80 ))
val_labels = pd.concat([val_labels, val_labels_2]).reset_index(drop=True)

df = pd.concat([pd.DataFrame(val_data), val_labels], axis=1).sample(frac=1) # .reset_index(drop=True)
del val_data, val_labels

val_data = np.array(df.iloc[:, :-1], dtype=np.float16).reshape((-1, 1, 128, 80))
val_labels = df.iloc[:, -1]
del df
print('TRAIN_SET labels:', len(labels))
print(labels.value_counts())
print()
print('VAL_SET labels:', len(val_labels))
print(val_labels.value_counts())
print()
print('num labels:', val_labels.nunique())
print()

# ---Other Data Files Not Used(option)---
"""#aug_data_1 = np.array(pd.read_pickle(r'../inputs/all_aug files/aug_processed_data.pkl'))
aug_labels_1 = pd.Series(pd.read_pickle(r'../inputs/all_aug files/aug_processed_labels.pkl'))
print(aug_labels_1.value_counts())

aug_data_2 = np.array(pd.read_pickle(r'../inputs/all_aug files/aug_speed_processed_data.pkl'))
aug_labels_2 = pd.Series(pd.read_pickle(r'../inputs/all_aug files/aug_speed_processed_labels (1).pkl'))

'''aug_data_3 = np.array(pd.read_pickle(r'../inputs/all_aug files/noise_aug_data.pkl'))
aug_labels_3 = pd.Series(pd.read_pickle(r'../inputs/all_aug files/noise_aug_labels.pkl'))

aug_data_6 = np.array(pd.read_pickle(r'../inputs/all_aug files/aug_processed_RainOnPond_noise_data.pkl'))
aug_labels_6 = pd.Series(pd.read_pickle(r'../inputs/all_aug files/aug_processed_RainOnPond_noise_label.pkl'))

aug_data_4 = np.array(pd.read_pickle(r'../inputs/all_aug files/aug_processed_RainOnPond_2_noise_data.pkl'))
aug_labels_4 = pd.Series(pd.read_pickle(r'../inputs/all_aug files/aug_processed_RainOnPond_2_noise_label.pkl'))
'''
aug_data_5 = np.array(pd.read_pickle(r'../inputs/all_aug files/aug_uploaded_processed_data.pkl'))
aug_labels_5 = pd.Series(pd.read_pickle(r'../inputs/all_aug files/aug_uploaded_processed_labels.pkl'))

data = np.concatenate([ aug_data_2, aug_data_5]).reshape((-1, 1 * 64 * 64))
labels = pd.concat([ aug_labels_2, aug_labels_5]).reset_index(drop=True)
print(labels.value_counts())

df = pd.concat([pd.DataFrame(data), labels], axis=1).sample(frac=1).reset_index(drop=True)

skylar_df = df[df.iloc[:, -1] == 'skylar'].sample(500)
houfin_df = df[df.iloc[:, -1] == 'houfin'].sample(500)
jabwar_df = df[df.iloc[:, -1] == 'jabwar'].sample(500)
df.drop(index=df[df.iloc[:, -1] == 'houfin'].index, inplace=True)
df.drop(index=df[df.iloc[:, -1] == 'skylar'].index, inplace=True)
df.drop(index=df[df.iloc[:, -1] == 'jabwar'].index, inplace=True)
df = pd.concat([df, skylar_df, houfin_df, jabwar_df]).sample(frac=1).reset_index(drop=True)
##
'''df.iloc[:, -threshold=0.5] = df.iloc[:, -threshold=0.5].apply(lambda x: 'noise' if x == 'noise' else 'bird')
bird_df = df[df.iloc[:, -threshold=0.5] == 'bird'].sample(9000)
df.drop(index=df[df.iloc[:, -threshold=0.5] == 'bird'].index, inplace=True)
df = pd.concat([df, bird_df]).sample(frac=threshold=0.5).reset_index(drop=True)'''
#
data = np.array(df.iloc[:, :-1], dtype=np.float16).reshape((-1, 1, 64, 64))
labels = df.iloc[:, -1]

# ############## #

data_1 = np.array(pd.read_pickle(r'../inputs/orig_processed_data.pkl'))
labels_1 = pd.Series(pd.read_pickle(r'../inputs/orig_processed_labels.pkl'))
# noise
'''data_2 = np.array(pd.read_pickle(r'../inputs/all_aug files/noise_orig_data.pkl'))
labels_2 = pd.Series(pd.read_pickle(r'../inputs/all_aug files/noise_orig_labels.pkl'))

data_3 = np.array(pd.read_pickle(r'../inputs/all_aug files/processed_RainOnPond_2_noise_data.pkl'))
labels_3 = pd.Series(pd.read_pickle(r'../inputs/all_aug files/processed_RainOnPond_2_noise_label.pkl'))

data_5 = np.array(pd.read_pickle(r'../inputs/all_aug files/processed_RainOnPond_noise_data.pkl'))
labels_5 = pd.Series(pd.read_pickle(r'../inputs/all_aug files/processed_RainOnPond_noise_label.pkl'))
'''

data_4 = np.array(pd.read_pickle(r'../inputs/all_aug files/uploaded_processed_data.pkl'))
labels_4 = pd.Series(pd.read_pickle(r'../inputs/all_aug files/uploaded_processed_labels.pkl'))

val_data = np.concatenate([data_1, data_4]).reshape((-1, 1 * 64 * 64))
val_labels = pd.concat([labels_1, labels_4]).reset_index(drop=True)

df = pd.concat([pd.DataFrame(val_data), val_labels], axis=1).sample(frac=1).reset_index(drop=True)
skylar_df = df[df.iloc[:, -1] == 'skylar'].sample(500)
houfin_df = df[df.iloc[:, -1] == 'houfin'].sample(500)
df.drop(index=df[df.iloc[:, -1] == 'houfin'].index, inplace=True)
df.drop(index=df[df.iloc[:, -1] == 'skylar'].index, inplace=True)
df = pd.concat([df, skylar_df, houfin_df]).sample(frac=1).reset_index(drop=True)

###

val_data = np.array(df.iloc[:, :-1], dtype=np.float16).reshape((-1, 1, 64, 64))
val_labels = df.iloc[:, -1]

del data_1, data_4, aug_data_2, aug_data_5, aug_labels_2, aug_labels_5 # aug_data_1, aug_labels_1
del labels_1, labels_4, houfin_df, skylar_df
"""
# --- use only 21 classes out of 121 classes(option)---#
'''scored_labels_dict = pd.read_json('../inputs/scored_birds.json').values

cat = pd.get_dummies(labels)

for col in cat.columns:
    if col not in scored_labels_dict:
        cat.drop(col, axis=threshold=0.5, inplace=True)
cat = pd.Series(cat.columns[np.where(cat != 0)[threshold=0.5]], name='label')
data['label'] = cat

# scored_data = data[data['label'].isna() == False].reset_index(drop=True)
non_scored_data = data[data['label'].isna() == True].reset_index(drop=True)[:20000]
# non_scored_data['label'] = 'noise'

# data= pd.concat([scored_data, non_scored_data]).sample(frac=threshold=0.5., random_state=292).reset_index(drop=True)
# labels = scored_data['label']
'''

########

print('audio mel shape:', val_data[1].shape)
# Create dict for classes(option)
'''class_dict = dict(enumerate(pd.read_json('../inputs/scored_birds.json')[0]))
class_dict[21] = 'noise' 
class_dict = {k: v for v, k in enumerate(class_dict.values())}  
pd.to_pickle(class_dict, 'class_dict.pkl', protocol=pickle.DEFAULT_PROTOCOL)

class_dict = {'akiapo': 0, 'aniani': 1, 'apapan': 2, 'barpet': 3, 'crehon': 4, 'elepai': 5, 'ercfra': 6, 'hawama': 7,
              'hawcre': 8, 'hawgoo': 9, 'hawhaw': 10, 'hawpet1': 11, 'houfin': 12, 'iiwi': 13, 'jabwar': 14,
              'maupar': 15, 'omao': 16, 'puaioh': 17, 'skylar': 18, 'warwhe1': 19, 'yefcan': 20}
print(class_dict)'''

# --- Dummy Data(test code is running correct, real data are heavy)(option) --- #
'''data_1 = np.random.uniform(-0.2, 0.2, (np.array(data).shape))
data_2 = np.random.uniform(-0.8, 0.8, (np.array(data).shape))
data = np.concatenate([data_1, data_2])
labels_1 = np.ones((10000))
labels_2 = np.zeros((10000))
labels = np.concatenate([labels_1, labels_2])'''
# --- split data(in our case we will train on aug data and validate on orig data so no need to split) --- #
'''
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=.2, random_state=32, stratify=labels)
# X_train = np.concatenate([aug_data, X_train])
# y_train = np.concatenate([aug_labels, y_train])

del data, labels
'''

# ----- TRAINING ----- #

print(f'shape: X_train={data.shape} | X_test={val_data.shape}')
labels = pd.get_dummies(labels)
val_labels = pd.get_dummies(val_labels)

train_set = deep_utils.SoundDataset_TRAIN(data, labels, class_dict=None) # class_dict=None since labels are one hot encoded
val_set = deep_utils.SoundDataset_VAL(val_data, val_labels, class_dict=None)
del data, val_data, labels, val_labels

# in case split is used(option)
'''
train_set = deep_utils.SoundDataset(X_train, y_train, class_dict)
val_set = deep_utils.SoundDataset(X_test, y_test, class_dict)
'''

train_loader = DataLoader(train_set, batch_size=100, drop_last=True, shuffle=True)
val_loader = DataLoader(val_set, batch_size=100, drop_last=True, shuffle=True)

model = deep_utils.CNNNetwork(21)
model.to(deep_utils.device)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


optimizer = torch.optim.AdamW(model.parameters(), 0.001)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 2)
print(f'The model has {count_parameters(model):,} trainable parameters')
# best reached score is F1-score 94% on orig(val) data
deep_utils.run(model, train_loader, optimizer, val_loader, scheduler, 1,epochs=100)

# example of running the file

r'''
concatenating training data and labels...
concatenating validation data and labels...
TRAIN_SET labels: 39371
skylar     5014
houfin     5001
jabwar     3524
hawcre     3140
elepai     2712
warwhe1    2284
apapan     2160
akiapo     2000
yefcan     1976
iiwi       1820
aniani     1624
crehon     1396
barpet     1140
hawgoo     1104
omao        980
maupar      972
hawpet1     864
hawama      644
puaioh      476
hawhaw      324
ercfra      216
Name: 0, dtype: int64

VAL_SET labels: 15922
skylar     5001
houfin     3582
jabwar      881
hawcre      785
elepai      678
warwhe1     571
apapan      540
akiapo      500
yefcan      494
iiwi        455
aniani      406
crehon      349
barpet      285
hawgoo      276
omao        245
maupar      243
hawpet1     216
hawama      161
puaioh      119
hawhaw       81
ercfra       54
dtype: int64

TRAIN_SET num labels: 21
VAL_SET num labels: 21
audio mel shape: (1, 128, 80)
{'akiapo': 0, 'aniani': 1, 'apapan': 2, 'barpet': 3, 'crehon': 4, 'elepai': 5, 'ercfra': 6, 'hawama': 7, 'hawcre': 8, 'hawgoo': 9, 'hawhaw': 10, 'hawpet1': 11, 'houfin': 12, 'iiwi': 13, 'jabwar': 14, 'maupar': 15, 'omao': 16, 'puaioh': 17, 'skylar': 18, 'warwhe1': 19, 'yefcan': 20}
shape: X_train=(39371, 1, 128, 80) | X_test=(15922, 1, 128, 80)
The model has 1,145,233 trainable parameters
TRAINING STARTED...
BEST EPOCH(1) time=(0.09)hours: loss=(train=2.2259 | val=1.3908) | F1-score=(train=0.198 | val=0.321) | Accuracy=(train=[0.315] | val=[0.555])
BEST EPOCH(2) time=(0.03)hours: loss=(train=1.6273 | val=1.2247) | F1-score=(train=0.36 | val=0.417) | Accuracy=(train=[0.484] | val=[0.61])
BEST EPOCH(3) time=(0.03)hours: loss=(train=1.4705 | val=1.1747) | F1-score=(train=0.433 | val=0.486) | Accuracy=(train=[0.531] | val=[0.632])
BEST EPOCH(4) time=(0.03)hours: loss=(train=1.3355 | val=1.1165) | F1-score=(train=0.508 | val=0.539) | Accuracy=(train=[0.577] | val=[0.638])
BEST EPOCH(5) time=(0.03)hours: loss=(train=1.222 | val=1.0002) | F1-score=(train=0.563 | val=0.583) | Accuracy=(train=[0.609] | val=[0.679])
BEST EPOCH(6) time=(0.03)hours: loss=(train=1.1198 | val=0.908) | F1-score=(train=0.605 | val=0.627) | Accuracy=(train=[0.64] | val=[0.705])
BEST EPOCH(7) time=(0.03)hours: loss=(train=1.0574 | val=0.897) | F1-score=(train=0.633 | val=0.657) | Accuracy=(train=[0.66] | val=[0.712])
BEST EPOCH(8) time=(0.03)hours: loss=(train=0.9773 | val=0.9098) | F1-score=(train=0.658 | val=0.671) | Accuracy=(train=[0.684] | val=[0.71])
BEST EPOCH(9) time=(0.03)hours: loss=(train=0.9021 | val=0.825) | F1-score=(train=0.692 | val=0.686) | Accuracy=(train=[0.709] | val=[0.739])
BEST EPOCH(10) time=(0.03)hours: loss=(train=0.8458 | val=0.8551) | F1-score=(train=0.706 | val=0.711) | Accuracy=(train=[0.725] | val=[0.731])
BEST EPOCH(11) time=(0.04)hours: loss=(train=0.7789 | val=0.886) | F1-score=(train=0.736 | val=0.728) | Accuracy=(train=[0.749] | val=[0.733])
BEST EPOCH(12) time=(0.03)hours: loss=(train=0.7411 | val=0.8115) | F1-score=(train=0.748 | val=0.739) | Accuracy=(train=[0.76] | val=[0.761])
BEST EPOCH(13) time=(0.06)hours: loss=(train=0.7035 | val=0.7549) | F1-score=(train=0.762 | val=0.751) | Accuracy=(train=[0.772] | val=[0.769])
epoch(14) time=(0.07)hours: loss=(train=0.6915 | val=0.8817) | F1-score=(train=0.763 | val=0.743) | Accuracy=(train=[0.775] | val=[0.742])
BEST EPOCH(15) time=(0.07)hours: loss=(train=0.6106 | val=0.7922) | F1-score=(train=0.794 | val=0.753) | Accuracy=(train=[0.801] | val=[0.77])
epoch(16) time=(0.06)hours: loss=(train=0.5853 | val=0.8342) | F1-score=(train=0.8 | val=0.747) | Accuracy=(train=[0.808] | val=[0.759])
BEST EPOCH(17) time=(0.03)hours: loss=(train=0.5717 | val=0.811) | F1-score=(train=0.809 | val=0.758) | Accuracy=(train=[0.817] | val=[0.77])
BEST EPOCH(18) time=(0.03)hours: loss=(train=0.5475 | val=0.8116) | F1-score=(train=0.814 | val=0.786) | Accuracy=(train=[0.82] | val=[0.771])
epoch(19) time=(0.03)hours: loss=(train=0.5148 | val=0.8419) | F1-score=(train=0.829 | val=0.773) | Accuracy=(train=[0.832] | val=[0.763])
epoch(20) time=(0.08)hours: loss=(train=0.4778 | val=0.8021) | F1-score=(train=0.842 | val=0.777) | Accuracy=(train=[0.844] | val=[0.779])
BEST EPOCH(21) time=(0.08)hours: loss=(train=0.4735 | val=0.7602) | F1-score=(train=0.843 | val=0.802) | Accuracy=(train=[0.846] | val=[0.79])
epoch(22) time=(0.08)hours: loss=(train=0.4534 | val=0.7571) | F1-score=(train=0.85 | val=0.797) | Accuracy=(train=[0.853] | val=[0.792])
epoch(23) time=(0.08)hours: loss=(train=0.4358 | val=0.8793) | F1-score=(train=0.855 | val=0.781) | Accuracy=(train=[0.858] | val=[0.78])
BEST EPOCH(24) time=(0.08)hours: loss=(train=0.4109 | val=0.7878) | F1-score=(train=0.868 | val=0.808) | Accuracy=(train=[0.868] | val=[0.798])
epoch(25) time=(0.07)hours: loss=(train=0.3872 | val=0.8379) | F1-score=(train=0.875 | val=0.803) | Accuracy=(train=[0.875] | val=[0.79])
BEST EPOCH(26) time=(0.08)hours: loss=(train=0.3678 | val=0.8377) | F1-score=(train=0.881 | val=0.815) | Accuracy=(train=[0.881] | val=[0.799])
BEST EPOCH(27) time=(0.08)hours: loss=(train=0.3625 | val=0.7533) | F1-score=(train=0.883 | val=0.825) | Accuracy=(train=[0.883] | val=[0.816])
epoch(28) time=(0.08)hours: loss=(train=0.3558 | val=0.7572) | F1-score=(train=0.886 | val=0.819) | Accuracy=(train=[0.884] | val=[0.812])
epoch(29) time=(0.08)hours: loss=(train=0.3494 | val=0.8711) | F1-score=(train=0.886 | val=0.815) | Accuracy=(train=[0.886] | val=[0.797])
BEST EPOCH(30) time=(0.08)hours: loss=(train=0.3094 | val=0.7546) | F1-score=(train=0.901 | val=0.827) | Accuracy=(train=[0.9] | val=[0.817])
BEST EPOCH(31) time=(0.03)hours: loss=(train=0.308 | val=0.8747) | F1-score=(train=0.9 | val=0.831) | Accuracy=(train=[0.9] | val=[0.8])
BEST EPOCH(32) time=(0.03)hours: loss=(train=0.3055 | val=0.8228) | F1-score=(train=0.9 | val=0.846) | Accuracy=(train=[0.9] | val=[0.816])
epoch(33) time=(0.03)hours: loss=(train=0.2855 | val=0.8412) | F1-score=(train=0.907 | val=0.826) | Accuracy=(train=[0.906] | val=[0.811])
epoch(34) time=(0.03)hours: loss=(train=0.2847 | val=0.8763) | F1-score=(train=0.911 | val=0.841) | Accuracy=(train=[0.908] | val=[0.812])
epoch(35) time=(0.05)hours: loss=(train=0.264 | val=0.9864) | F1-score=(train=0.913 | val=0.839) | Accuracy=(train=[0.913] | val=[0.805])
epoch(36) time=(0.03)hours: loss=(train=0.2399 | val=0.8886) | F1-score=(train=0.925 | val=0.821) | Accuracy=(train=[0.923] | val=[0.813])
epoch(37) time=(0.03)hours: loss=(train=0.2361 | val=1.0423) | F1-score=(train=0.928 | val=0.835) | Accuracy=(train=[0.924] | val=[0.788])
epoch(38) time=(0.03)hours: loss=(train=0.2198 | val=0.9516) | F1-score=(train=0.931 | val=0.841) | Accuracy=(train=[0.929] | val=[0.806])
BEST EPOCH(39) time=(0.03)hours: loss=(train=0.2304 | val=0.9071) | F1-score=(train=0.93 | val=0.847) | Accuracy=(train=[0.925] | val=[0.814])
epoch(40) time=(0.03)hours: loss=(train=0.2178 | val=0.8497) | F1-score=(train=0.934 | val=0.844) | Accuracy=(train=[0.93] | val=[0.824])
epoch(41) time=(0.03)hours: loss=(train=0.2235 | val=0.9141) | F1-score=(train=0.932 | val=0.846) | Accuracy=(train=[0.928] | val=[0.815])
epoch(42) time=(0.03)hours: loss=(train=0.2135 | val=0.9487) | F1-score=(train=0.936 | val=0.839) | Accuracy=(train=[0.931] | val=[0.811])
epoch(43) time=(0.03)hours: loss=(train=0.2058 | val=0.9416) | F1-score=(train=0.937 | val=0.832) | Accuracy=(train=[0.933] | val=[0.813])
BEST EPOCH(44) time=(0.03)hours: loss=(train=0.195 | val=0.9543) | F1-score=(train=0.941 | val=0.851) | Accuracy=(train=[0.936] | val=[0.816])
epoch(45) time=(0.03)hours: loss=(train=0.1971 | val=0.9866) | F1-score=(train=0.941 | val=0.85) | Accuracy=(train=[0.936] | val=[0.823])
epoch(46) time=(0.03)hours: loss=(train=0.1789 | val=1.087) | F1-score=(train=0.947 | val=0.847) | Accuracy=(train=[0.942] | val=[0.817])
epoch(47) time=(0.03)hours: loss=(train=0.1737 | val=1.0353) | F1-score=(train=0.95 | val=0.845) | Accuracy=(train=[0.944] | val=[0.823])
epoch(48) time=(0.03)hours: loss=(train=0.1653 | val=1.0929) | F1-score=(train=0.951 | val=0.844) | Accuracy=(train=[0.946] | val=[0.824])
BEST EPOCH(49) time=(0.03)hours: loss=(train=0.1562 | val=0.9995) | F1-score=(train=0.956 | val=0.852) | Accuracy=(train=[0.95] | val=[0.833])
BEST EPOCH(50) time=(0.03)hours: loss=(train=0.1662 | val=1.0785) | F1-score=(train=0.95 | val=0.853) | Accuracy=(train=[0.946] | val=[0.825])
epoch(51) time=(0.03)hours: loss=(train=0.1592 | val=1.0234) | F1-score=(train=0.955 | val=0.843) | Accuracy=(train=[0.949] | val=[0.831])
epoch(52) time=(0.03)hours: loss=(train=0.1602 | val=1.1328) | F1-score=(train=0.952 | val=0.834) | Accuracy=(train=[0.948] | val=[0.819])
epoch(53) time=(0.03)hours: loss=(train=0.1482 | val=1.0636) | F1-score=(train=0.957 | val=0.84) | Accuracy=(train=[0.952] | val=[0.83])
BEST EPOCH(54) time=(0.04)hours: loss=(train=0.1471 | val=0.9258) | F1-score=(train=0.955 | val=0.86) | Accuracy=(train=[0.953] | val=[0.847])
epoch(55) time=(0.03)hours: loss=(train=0.1704 | val=1.1115) | F1-score=(train=0.951 | val=0.848) | Accuracy=(train=[0.946] | val=[0.822])
BEST EPOCH(56) time=(0.03)hours: loss=(train=0.1441 | val=0.9723) | F1-score=(train=0.959 | val=0.867) | Accuracy=(train=[0.955] | val=[0.836])
epoch(57) time=(0.03)hours: loss=(train=0.1421 | val=1.1446) | F1-score=(train=0.96 | val=0.846) | Accuracy=(train=[0.955] | val=[0.824])
epoch(58) time=(0.09)hours: loss=(train=0.1307 | val=1.1396) | F1-score=(train=0.963 | val=0.863) | Accuracy=(train=[0.959] | val=[0.829])
epoch(59) time=(0.09)hours: loss=(train=0.1293 | val=1.0535) | F1-score=(train=0.964 | val=0.861) | Accuracy=(train=[0.959] | val=[0.837])
epoch(60) time=(0.08)hours: loss=(train=0.1353 | val=1.0234) | F1-score=(train=0.963 | val=0.853) | Accuracy=(train=[0.958] | val=[0.834])
epoch(61) time=(0.07)hours: loss=(train=0.1409 | val=1.2582) | F1-score=(train=0.959 | val=0.852) | Accuracy=(train=[0.955] | val=[0.814])
epoch(62) time=(0.03)hours: loss=(train=0.1157 | val=1.1276) | F1-score=(train=0.97 | val=0.864) | Accuracy=(train=[0.964] | val=[0.841])
epoch(63) time=(0.09)hours: loss=(train=0.1218 | val=1.1944) | F1-score=(train=0.965 | val=0.864) | Accuracy=(train=[0.961] | val=[0.83])
epoch(64) time=(0.09)hours: loss=(train=0.1168 | val=1.1813) | F1-score=(train=0.969 | val=0.857) | Accuracy=(train=[0.964] | val=[0.825])
epoch(65) time=(0.08)hours: loss=(train=0.1182 | val=1.3539) | F1-score=(train=0.969 | val=0.851) | Accuracy=(train=[0.962] | val=[0.815])
BEST EPOCH(66) time=(0.08)hours: loss=(train=0.1035 | val=1.1643) | F1-score=(train=0.973 | val=0.876) | Accuracy=(train=[0.968] | val=[0.844])
epoch(67) time=(0.08)hours: loss=(train=0.1083 | val=1.3476) | F1-score=(train=0.971 | val=0.866) | Accuracy=(train=[0.966] | val=[0.829])
BEST EPOCH(68) time=(0.08)hours: loss=(train=0.0977 | val=1.2553) | F1-score=(train=0.975 | val=0.88) | Accuracy=(train=[0.969] | val=[0.842])
epoch(69) time=(0.08)hours: loss=(train=0.0996 | val=1.3612) | F1-score=(train=0.974 | val=0.867) | Accuracy=(train=[0.968] | val=[0.833])
epoch(70) time=(0.08)hours: loss=(train=0.0995 | val=1.4532) | F1-score=(train=0.975 | val=0.862) | Accuracy=(train=[0.969] | val=[0.827])
BEST EPOCH(71) time=(0.09)hours: loss=(train=0.0904 | val=1.2895) | F1-score=(train=0.979 | val=0.89) | Accuracy=(train=[0.972] | val=[0.851])
epoch(72) time=(0.09)hours: loss=(train=0.0927 | val=1.3058) | F1-score=(train=0.978 | val=0.881) | Accuracy=(train=[0.972] | val=[0.843])
epoch(73) time=(0.07)hours: loss=(train=0.0958 | val=1.5657) | F1-score=(train=0.976 | val=0.865) | Accuracy=(train=[0.97] | val=[0.825])
epoch(74) time=(0.05)hours: loss=(train=0.0866 | val=1.4231) | F1-score=(train=0.98 | val=0.876) | Accuracy=(train=[0.973] | val=[0.837])
epoch(75) time=(0.06)hours: loss=(train=0.0846 | val=1.4589) | F1-score=(train=0.98 | val=0.873) | Accuracy=(train=[0.974] | val=[0.839])
epoch(76) time=(0.04)hours: loss=(train=0.0851 | val=1.3941) | F1-score=(train=0.98 | val=0.882) | Accuracy=(train=[0.974] | val=[0.845])
epoch(77) time=(0.04)hours: loss=(train=0.0852 | val=1.3841) | F1-score=(train=0.979 | val=0.885) | Accuracy=(train=[0.973] | val=[0.848])
epoch(78) time=(0.04)hours: loss=(train=0.0864 | val=1.4711) | F1-score=(train=0.978 | val=0.879) | Accuracy=(train=[0.973] | val=[0.846])
epoch(79) time=(0.04)hours: loss=(train=0.0835 | val=1.4875) | F1-score=(train=0.98 | val=0.88) | Accuracy=(train=[0.975] | val=[0.847])
epoch(80) time=(0.04)hours: loss=(train=0.0817 | val=1.3754) | F1-score=(train=0.981 | val=0.888) | Accuracy=(train=[0.975] | val=[0.853])
epoch(81) time=(0.04)hours: loss=(train=0.0825 | val=1.6174) | F1-score=(train=0.981 | val=0.868) | Accuracy=(train=[0.975] | val=[0.841])
epoch(82) time=(0.04)hours: loss=(train=0.0805 | val=1.4995) | F1-score=(train=0.982 | val=0.882) | Accuracy=(train=[0.975] | val=[0.848])
epoch(83) time=(0.04)hours: loss=(train=0.0796 | val=1.5538) | F1-score=(train=0.982 | val=0.879) | Accuracy=(train=[0.975] | val=[0.846])
epoch(84) time=(0.05)hours: loss=(train=0.0793 | val=1.5023) | F1-score=(train=0.982 | val=0.887) | Accuracy=(train=[0.975] | val=[0.851])
epoch(85) time=(0.04)hours: loss=(train=0.0797 | val=1.7101) | F1-score=(train=0.982 | val=0.874) | Accuracy=(train=[0.975] | val=[0.84])
epoch(86) time=(0.04)hours: loss=(train=0.0783 | val=1.561) | F1-score=(train=0.982 | val=0.886) | Accuracy=(train=[0.976] | val=[0.85])
epoch(87) time=(0.04)hours: loss=(train=0.0787 | val=1.5149) | F1-score=(train=0.982 | val=0.888) | Accuracy=(train=[0.976] | val=[0.852])
epoch(88) time=(0.04)hours: loss=(train=0.078 | val=1.6469) | F1-score=(train=0.982 | val=0.878) | Accuracy=(train=[0.976] | val=[0.844])
epoch(89) time=(0.04)hours: loss=(train=0.0778 | val=1.6041) | F1-score=(train=0.983 | val=0.885) | Accuracy=(train=[0.976] | val=[0.848])
epoch(90) time=(0.04)hours: loss=(train=0.0773 | val=1.6752) | F1-score=(train=0.982 | val=0.883) | Accuracy=(train=[0.976] | val=[0.848])
epoch(91) time=(0.04)hours: loss=(train=0.0768 | val=1.7233) | F1-score=(train=0.983 | val=0.875) | Accuracy=(train=[0.976] | val=[0.844])
epoch(92) time=(0.05)hours: loss=(train=0.0768 | val=1.706) | F1-score=(train=0.983 | val=0.88) | Accuracy=(train=[0.976] | val=[0.847])
epoch(93) time=(0.05)hours: loss=(train=0.0767 | val=1.677) | F1-score=(train=0.983 | val=0.879) | Accuracy=(train=[0.976] | val=[0.847])
epoch(94) time=(0.04)hours: loss=(train=0.0763 | val=1.6115) | F1-score=(train=0.983 | val=0.884) | Accuracy=(train=[0.977] | val=[0.851])
epoch(95) time=(0.07)hours: loss=(train=0.077 | val=1.6889) | F1-score=(train=0.983 | val=0.886) | Accuracy=(train=[0.976] | val=[0.849])
epoch(96) time=(0.04)hours: loss=(train=0.0767 | val=1.6999) | F1-score=(train=0.983 | val=0.88) | Accuracy=(train=[0.976] | val=[0.847])
epoch(97) time=(0.04)hours: loss=(train=0.0764 | val=1.7089) | F1-score=(train=0.982 | val=0.881) | Accuracy=(train=[0.976] | val=[0.847])
epoch(98) time=(1.87)hours: loss=(train=0.0763 | val=1.685) | F1-score=(train=0.983 | val=0.885) | Accuracy=(train=[0.976] | val=[0.848])
epoch(99) time=(0.03)hours: loss=(train=0.0763 | val=1.718) | F1-score=(train=0.982 | val=0.881) | Accuracy=(train=[0.976] | val=[0.846])
TRAINING time=6.62

Process finished with exit code 0


'''