from sklearn.neural_network import MLPClassifier
from part3.load_data import load_brain_dim_red, load_bank_dim_red

brain_ica, brain_pca, brain_rand_proj = load_brain_dim_red()
bank_ica, bank_pca, bank_rand_proj = load_bank_dim_red()

brain_data = [
    brain_ica, brain_pca, brain_rand_proj
]

bank_data = [
    bank_ica, bank_pca, bank_rand_proj
]

data = []

for d in brain_data:
    y = d['Class']
    x = d.drop('Class', axis=1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    data.append((x_train, x_test, y_train, y_test))

for d in bank_data:
    y = d['Bankrupt?']
    x = d.drop('Bankrupt?', axis=1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    data.append((x_train, x_test, y_train, y_test))


for d in data:
    x_train = d[0]
    y_train = d[2]
    nn = MLPClassifier(
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

    nn.fit(x_train, y_train)
