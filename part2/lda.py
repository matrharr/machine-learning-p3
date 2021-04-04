import numpy as np
import matplotlib.pyplot as plt


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import preprocessing


from data.load_data import load_bankrupt_data, load_brain_tumor_data

x_train_brain, x_test_brain, y_train_brain, y_test_brain, raw_brain, y_brain = load_brain_tumor_data()
x_train_bank, x_test_bank, y_train_bank, y_test_bank, raw_bank, y_bank = load_bankrupt_data()

scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
raw_brain = scaler.fit_transform(raw_brain)
raw_bank = scaler.fit_transform(raw_bank)

lda_brain = LinearDiscriminantAnalysis().fit(raw_brain, y_brain)
lda_brain_data = lda_brain.transform(raw_brain)

# fig, ax = plt.subplots()
# for color, i, target_name in zip(['r'], [0], [0]):
#     plt.scatter(lda_brain_data[y_brain == i, 0], lda_brain_data[y_brain == i, 1], color=color, alpha=.8, lw=2,
#                 label=target_name)
# plt.legend(loc='best', shadow=False, scatterpoints=1)
# plt.title('lda')
# fig.savefig('figures/part2_lda_scatter_brain.png')
# plt.show()

with open('part2/lda-brain.npy', 'wb') as f:
    np.save(f, lda_brain_data)


lda_bankrupt = LinearDiscriminantAnalysis().fit(raw_bank, y_bank)
lda_bankrupt_data = lda_bankrupt.transform(raw_bank)

# fig, ax = plt.subplots()
# for color, i, target_name in zip(['r'], [0], [0]):
#     plt.scatter(lda_bankrupt_data[y_bank == i, 0], lda_bankrupt_data[y_bank == i, 1], color=color, alpha=.8, lw=2,
#                 label=target_name)
# plt.legend(loc='best', shadow=False, scatterpoints=1)
# plt.title('lda')
# fig.savefig('figures/part2_lda_scatter_bank.png')
# plt.show()

with open('part2/lda-bankrupt.npy', 'wb') as f:
    np.save(f, lda_bankrupt_data)