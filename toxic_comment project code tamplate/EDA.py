import matplotlib.pyplot as plt
import seaborn as sb

"""
log = pd.read_csv('../finished models/model_1 0.95/lstm_with_tfidf_training_log.csv')

plt.title('loss')
plt.plot(log['train_loss_log'], label='train_loss')
plt.plot(log['val_loss_log'], label='val_loss')

plt.legend()
plt.show()"""

import numpy as np
import pandas as pd

matrix = pd.DataFrame(pd.read_pickle('embedding utils/big_matrix.pkl'))
from sklearn.decomposition import PCA, IncrementalPCA

pca = IncrementalPCA(batch_size=32).fit(matrix)

# plot PCA explained_variance_ratio_
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.axhline(y=.90, color='r', linestyle='--')
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')

# create dataframe of explained_variance_ratio_ values
cumsum = pd.DataFrame(np.cumsum(pca.explained_variance_ratio_))
print('cumulative explained variance values: ')
print(cumsum)

# extract components the give 90% info of data, and use the first one
best_component = cumsum[(cumsum >= .90) & (cumsum < .91)].dropna().index[0]
# (cumsum >= .90) & (cumsum  < .92) is because we may not find a component of varaince .90, so we will take any
# component between .90 and 91


print(f'\n total most common words: {len(matrix.columns)}')

print('Best Component that keep 90% info:', best_component)
print(f'\n total most common words after reduction: {best_component}')
