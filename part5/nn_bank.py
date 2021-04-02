import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from data.load_data import load_bankrupt_data
from visualize.plot_graphs import plot_learning_curve
from metrics import get_metrics
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split, ShuffleSplit
from sklearn.metrics import accuracy_score
from sklearn.model_selection import learning_curve



x_train, x_test, y_train, y_test, raw, y = load_bankrupt_data()
features = []

print(x_train.head())

with open('part3/kmeans_bank_pca.npy', 'rb') as f:
    file = np.load(f)
    features.append([file, 'kmeans_pca'])

with open('part3/kmeans_bank_ica.npy', 'rb') as f:
    file = np.load(f)
    features.append([file, 'kmeans_ica'])

with open('part3/kmeans_bank_rand-proj.npy', 'rb') as f:
    file = np.load(f)
    features.append([file, 'kmeans_rp'])

with open('part3/gm_bank_pca.npy', 'rb') as f:
    file = np.load(f)
    features.append([file, 'gm_pca'])

with open('part3/gm_bank_ica.npy', 'rb') as f:
    file = np.load(f)
    features.append([file, 'gm_ica'])

with open('part3/gm_bank_rand-proj.npy', 'rb') as f:
    file = np.load(f)
    features.append([file, 'gm_rp'])



for feature in features:
    cluster_dimred_name = feature[1]
    data = feature[0]
    x_train, x_test, y_train, y_test = train_test_split(data, y, test_size=0.2)


    print('---------------Neural Network bank --------------')
    print('dim reduction: ', cluster_dimred_name)
    nn_model = MLPClassifier(
        solver="lbfgs",
        hidden_layer_sizes=(3,),
        activation='relu',
        batch_size='auto',
        alpha=0.0001,
        learning_rate='constant',
        learning_rate_init=0.0001,
        power_t=0.5,
        max_iter=400,
        shuffle=True,
        random_state=3,
        tol=0.0001,
        verbose=False,
        warm_start=False,
        momentum=0.9,
        nesterovs_momentum=True,
        early_stopping=False,
        validation_fraction=0.1,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-8,
        n_iter_no_change=10,
        max_fun=15000 # only use w/lbfgs
    )

    nn_model.fit(x_train, y_train)

    y_train_pred = nn_model.predict(x_train)

    print('Training Score: ', accuracy_score(y_train, y_train_pred))

    y_test_pred = nn_model.predict(x_test)

    print('Testing Score: ', accuracy_score(y_test, y_test_pred))

    metrics = get_metrics(nn_model, x_train, y_train, x_test, y_test, y_test_pred, data, y)
    print('loss: ', nn_model.loss)


    fig, axes = plt.subplots(3, 2, figsize=(10, 15))
    title = f'Learning Curve using {cluster_dimred_name}'
    # Cross validation with 100 iterations to get smoother mean test and train
    # score curves, each time with 20% data randomly selected as a validation set.
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=5)
    axes=axes[:, 0]
    estimator = nn_model
    ylim=(0.1, 1.01)
    train_sizes=np.linspace(.1, 1.0, 5)
    print('here')
    # plot_learning_curve(
    #     estimator,
    #     title,
    #     x_all,
    #     y_all,
    #     axes=axes[:, 0],
    #     ylim=(0.7, 1.01),
    #     cv=cv,
    #     n_jobs=4
    # )
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, data, y, cv=cv,
                       train_sizes=train_sizes,
                       return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)
    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    print(train_scores_mean)
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    print(test_scores_mean)
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes[0].legend(loc="best")
    # plt.show()
    fig.savefig(f'figures/part5_nn_bank_{cluster_dimred_name}_learning_curve')
