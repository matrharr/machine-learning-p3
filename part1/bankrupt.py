# Run the clustering algorithms on the datasets and describe what you see.
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn import mixture
from sklearn import preprocessing
import seaborn as sns

from data.load_data import load_bankrupt_data, load_brain_tumor_data
from visualize.plot_graphs import plot_kmeans, plot_gm, density_est_gmm


x_train, x_test, y_train, y_test, raw, y = load_bankrupt_data()

scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

kmeans = KMeans(
    n_clusters=4,
    init='k-means++',
    n_init=10,
    max_iter=300,
    tol=1e-4,
    # precompute_distances='auto',
    verbose=0,
    random_state=None,
    copy_x=True,
    n_jobs=None,
    algorithm="auto"
).fit(
    x_train, y=None, sample_weight=None
)

labels = pd.DataFrame(kmeans.labels_) #This is where the label output of the KMeans we just ran lives. Make it a dataframe so we can concatenate back to the original data
og_data = pd.concat((pd.DataFrame(raw), labels),axis=1)
og_data = og_data.rename({0:'labels'},axis=1)

print(og_data.head())
for col in og_data.columns:
    print(col)

# plot 2d
# sns.lmplot(x='Correlation',y='Skewness',data=og_data,hue='labels',fit_reg=False)

# slow
# all pairs
sns_pair = sns.pairplot(og_data,hue='labels')
sns_pair.savefig('figures/part1_kmeans_bankrupt.png')

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

y_kmeans = kmeans.predict(x_train)

print('----------------Bankrupt data-------------------')
print('kmeans cluster centers:', kmeans.cluster_centers_)
print('kmeans number of clusters:', len(kmeans.cluster_centers_))
print('kmeans labels:',  kmeans.labels_)
print('kmeans num of iterations:', kmeans.n_iter_)

gm = mixture.GaussianMixture(
    n_components=1,
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
    x_train
)
