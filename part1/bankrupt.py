# Run the clustering algorithms on the datasets and describe what you see.
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn import mixture
from sklearn import preprocessing

from data.load_data import load_bankrupt_data, load_brain_tumor_data
from visualize.plot_graphs import plot_kmeans, plot_gm


x_train, x_test, y_train, y_test, raw = load_bankrupt_data()

scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

kmeans = KMeans(
    n_clusters=8,
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
    x_train, y=None, sample_weight=None
)

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

print('-----------------------------Bankruptcy Data----------------------------')
print('kmeans cluster centers:', kmeans.cluster_centers_)
print('kmeans number of clusters:', len(kmeans.cluster_centers_))
print('kmeans labels:',  kmeans.labels_)
print('kmeans num of iterations:', kmeans.n_iter_)

# plot points
# plot_scatter(
#     kmeans.cluster_centers_, 
#     x_train, x_train, y_kmeans
# )
# plot_gm(gm, x_train, y_train, x_test, y_test)