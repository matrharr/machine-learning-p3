import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


from sklearn.decomposition import PCA
from sklearn import preprocessing


from data.load_data import load_bankrupt_data, load_brain_tumor_data
from visualize.plot_graphs import plot_pca2d

x_train_brain, x_test_brain, y_train_brain, y_test_brain, raw_brain, y_brain = load_brain_tumor_data()
x_train_bank, x_test_bank, y_train_bank, y_test_bank, raw_bank, y_bank = load_bankrupt_data()

scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
raw_brain = scaler.fit_transform(raw_brain)
raw_bank = scaler.fit_transform(raw_bank)

pca_brain = PCA(
    n_components=2,
    copy=True,
    whiten=True,
    svd_solver='auto',
    tol=0.0,
    iterated_power='auto',
    random_state=None
).fit(raw_brain)

pca_brain_data = pca_brain.transform(raw_brain)

print('PCA components:', pca_brain.components_)
print('PCA explained variance:', pca_brain.explained_variance_)
print('PCA explained variance ratio:', pca_brain.explained_variance_ratio_)

# scatter
fig, ax = plt.subplots()
for color, i, target_name in zip(['r', 'b', 'y'], [0, 1, 2], [0, 1, 2]):
    plt.scatter(pca_brain_data[y_brain == i, 0], pca_brain_data[y_brain == i, 1], color=color, alpha=.8, lw=2,
                label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('PCA')
fig.savefig('figures/part2_pca_scatter_brain.png')

# plot_pca2d(pca_brain)

# bar graph, almost all variation in comp 1
# per_var = np.round(pca_brain.explained_variance_ratio_* 100, decimals=1)
# labels = ['PC' + str(x) for x in range(1, len(per_var)+1)]
# plt.bar(x=range(1,len(per_var)+1), height=per_var, tick_label=labels)
# plt.ylabel('Percentage of Explained Variance')
# plt.xlabel('Principal Component')
# plt.title('Scree Plot')
# plt.show()
with open('part2/pca-brain.npy', 'wb') as f:
    np.save(f, pca_brain_data)

pca_bankrupt = PCA(
    n_components=2,
    copy=True,
    whiten=False,
    svd_solver='auto',
    tol=0.0,
    iterated_power='auto',
    random_state=None
).fit(raw_bank)

pca_bankrupt_data = pca_bankrupt.transform(raw_bank)

# scatter
fig, ax = plt.subplots()
for color, i, target_name in zip(['r', 'b'], [0, 1], [0, 1]):
    plt.scatter(pca_bankrupt_data[y_bank == i, 0], pca_bankrupt_data[y_bank == i, 1], color=color, alpha=.8, lw=2,
                label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('PCA')
fig.savefig('figures/part2_pca_scatter_bank.png')

# plot_pca2d(pca_bankrupt)

# bar graph, almost all variation in comp 1
# per_var = np.round(pca_bankrupt.explained_variance_ratio_* 100, decimals=1)
# labels = ['PC' + str(x) for x in range(1, len(per_var)+1)]
# plt.bar(x=range(1,len(per_var)+1), height=per_var, tick_label=labels)
# plt.ylabel('Percentage of Explained Variance')
# plt.xlabel('Principal Component')
# plt.title('Scree Plot')
# plt.show()
with open('part2/pca-bankrupt.npy', 'wb') as f:
    np.save(f, pca_bankrupt_data)
