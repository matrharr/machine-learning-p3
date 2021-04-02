import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data.load_data import load_bankrupt_data, load_brain_tumor_data
from visualize.plot_graphs import plot_pca2d

from sklearn.random_projection import GaussianRandomProjection

x_train_brain, x_test_brain, y_train_brain, y_test_brain, raw_brain, y_brain = load_brain_tumor_data()
x_train_bank, x_test_bank, y_train_bank, y_test_bank, raw_bank, y_bank = load_bankrupt_data()


gauss_brain = GaussianRandomProjection(
    n_components=2,
    eps=0.1,
    random_state=None
).fit(raw_brain)

gauss_brain_data = gauss_brain.transform(raw_brain)

fig, ax = plt.subplots()
for color, i, target_name in zip(['r', 'b'], [0, 1], [0, 1]):
    plt.scatter(gauss_brain_data[y_brain == i, 0], gauss_brain_data[y_brain == i, 1], color=color, alpha=.8, lw=2,
                label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('Random Projection Brain Tumor')
fig.savefig('figures/part2_rp_scatter_brain.png')
# plt.show()

with open('part2/rand-proj-brain.npy', 'wb') as f:
    np.save(f, gauss_brain_data)

gauss_bankrupt = GaussianRandomProjection(
    n_components=2,
    eps=0.1,
    random_state=None
).fit(raw_bank)

gauss_bankrupt_data = gauss_bankrupt.transform(raw_bank)

fig, ax = plt.subplots()
for color, i, target_name in zip(['r', 'b'], [0, 1], [0, 1]):
    plt.scatter(gauss_bankrupt_data[y_bank == i, 0], gauss_bankrupt_data[y_bank == i, 1], color=color, alpha=.8, lw=2,
                label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('Random Projection Bankruptcy')
fig.savefig('figures/part2_rp_scatter_bank.png')
# plt.show()

with open('part2/rand-proj-bank.npy', 'wb') as f:
    np.save(f, gauss_bankrupt_data)

