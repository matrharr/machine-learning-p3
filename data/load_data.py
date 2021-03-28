import pandas as pd

from sklearn.model_selection import train_test_split


def load_brain_tumor_data():
    brain = pd.read_csv('data/brain-tumor.csv')
    # brain = brain.loc[:2000]
    brain.drop(['Image'], axis=1, inplace=True)
    y = brain['Class']
    x = brain.drop('Class', axis=1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    return x_train, x_test, y_train, y_test, x, y


def load_bankrupt_data():
    bankrupt = pd.read_csv('data/company-bankrupt.csv')
    y = bankrupt['Bankrupt?']
    x = bankrupt.drop('Bankrupt?', axis=1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    return x_train, x_test, y_train, y_test, x, y