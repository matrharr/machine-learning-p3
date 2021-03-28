import numpy as np
import matplotlib.pyplot as plt


from sklearn.decomposition import FastICA

from data.load_data import load_bankrupt_data, load_brain_tumor_data

x_train_brain, x_test_brain, y_train_brain, y_test_brain, raw_brain, y_brain = load_brain_tumor_data()
x_train_bank, x_test_bank, y_train_bank, y_test_bank, raw_bank, y_bank = load_bankrupt_data()


# ica_brain = FastICA(
#     n_components=None,
#     algorithm='parallel',
#     whiten=False,
#     fun='logcosh',
#     fun_args=None,
#     max_iter=200,
#     tol=1e-4,
#     w_init=None,
#     random_state=None
# ).fit(x_train_brain)
ica_brain = FastICA(n_components=2).fit(x_train_brain)
ica_brain_data = ica_brain.transform(x_train_brain)
# for color, i, target_name in zip(['r', 'b'], [0, 1], [0, 1]):
#     plt.scatter(ica_brain_data[y_train_brain == i, 0], ica_brain_data[y_train_brain == i, 1], color=color, alpha=.8, lw=2,
#                 label=target_name)
# plt.legend(loc='best', shadow=False, scatterpoints=1)
# plt.title('ICA')
# plt.show()

with open('part2/ica-brain.npy', 'wb') as f:
    np.save(f, ica_brain_data)

print(type(ica_brain))

# ica_bank = FastICA(
#     n_components=None,
#     algorithm='parallel',
#     whiten=False,
#     fun='logcosh',
#     fun_args=None,
#     max_iter=200,
#     tol=1e-4,
#     w_init=None,
#     random_state=None
# ).fit(x_train_bank)
