import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns


from data.load_data import load_brain_tumor_data
from visualize.plot_graphs import density_est_gmm
from sklearn.cluster import KMeans
from sklearn import mixture

features = []

x_train_brain, x_test_brain, y_train_brain, y_test_brain, raw_brain, y_brain = load_brain_tumor_data()

with open('part2/pca-brain.npy', 'rb') as f:
    file = np.load(f)
    features.append([file, 'pca'])

with open('part2/ica-brain.npy', 'rb') as f:
    file = np.load(f)
    features.append([file, 'ica'])

with open('part2/rand-proj-brain.npy', 'rb') as f:
    file = np.load(f)
    features.append([file, 'rand-proj'])


for feature in features:
    dim_red_name = feature[1]
    data = feature[0]

    print('---------------Kmeans--------------')
    print('dim reduction: ', dim_red_name)
    
    kmeans = KMeans(
        n_clusters=4,
        init='k-means++',
        n_init=10,
        max_iter=300,
        tol=1e-4,
        precompute_distances='auto',
        verbose=0,
        random_state=None,
        copy_x=True,
        n_jobs=None,
        algorithm="auto"
    ).fit(
        data, y=None, sample_weight=None
    )

    kmeans_data = kmeans.transform(data)

    labels = pd.DataFrame(kmeans.labels_) #This is where the label output of the KMeans we just ran lives. Make it a dataframe so we can concatenate back to the original data
    og_data = pd.concat((pd.DataFrame(raw_brain), labels),axis=1)
    og_data = og_data.rename({0:'labels'},axis=1)

    print(og_data.head())
    for col in og_data.columns:
        print(col)

    with open(f'part3/kmeans_brain_{dim_red_name}.npy', 'wb') as f:
        np.save(f, kmeans_data)
    # plot 2d
    sns.lmplot(x='Correlation',y='Skewness',data=og_data,hue='labels',fit_reg=False)

    # slow
    # all pairs
    sns_pair = sns.pairplot(og_data,hue='labels')
    sns_pair.savefig(f'figures/part3_kmeans_brain_{dim_red_name}.png')

    # very slow
    # strip plots
    # og_data['Constant'] = "Data"
    # f, axes = plt.subplots(4, 5, figsize=(20, 25), sharex=False) 
    # f.subplots_adjust(hspace=0.2, wspace=0.7)
    # for i in range(0,len(list(og_data))-2):
    #     col = og_data.columns[i]
    #     if i < 5:
    #         ax = sns.swarmplot(x=og_data['Constant'],y=og_data[col].values,hue=og_data['labels'],ax=axes[0,(i)])
    #         ax.set_title(col)
    #     elif i >= 5 and i<10:
    #         ax = sns.swarmplot(x=og_data['Constant'],y=og_data[col].values,hue=og_data['labels'],ax=axes[1,(i-5)])
    #         ax.set_title(col)
    #     elif i >= 10 and i<15:
    #         ax = sns.swarmplot(x=og_data['Constant'],y=og_data[col].values,hue=og_data['labels'],ax=axes[2,(i-10)])
    #         ax.set_title(col)
    #     elif i >= 15:
    #         ax = sns.swarmplot(x=og_data['Constant'],y=og_data[col].values,hue=og_data['labels'],ax=axes[3,(i-15)])
    #         ax.set_title(col)

    # plt.show()

for feature in features:
    dim_red_name = feature[1]
    data = feature[0]

    print('---------------Exp Maximization--------------')
    print('dim reduction: ', dim_red_name)


    gm = mixture.GaussianMixture(
        n_components=2,
        covariance_type='full',
        tol=1e-3,
        reg_covar=1e-6,
        max_iter=100,
        n_init=1,
        init_params='kmeans',
        weights_init=None,
        means_init=None,
        precisions_init=None,
        random_state=None,
        warm_start=False,
        verbose=0,
        verbose_interval=10
    ).fit(
        data
    )

    gm_data = gm.predict_proba(data)

    with open(f'part3/gm_brain_{dim_red_name}.npy', 'wb') as f:
        np.save(f, gm_data)

    # density_est_gmm(gm)