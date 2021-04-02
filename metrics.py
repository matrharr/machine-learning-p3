import numpy as np

from sklearn.metrics import mean_squared_error, confusion_matrix, precision_score, recall_score, roc_curve, roc_auc_score, f1_score, accuracy_score
from sklearn.model_selection import cross_val_predict, cross_val_score


def get_metrics(
    model, x_train, y_train, x_test, y_test, predictions,
    x_all, y_all
):
    return {
        'mse': get_mse(y_test, predictions),
        'rmse': get_rmse(y_test, predictions),
        'confusion_matrix': get_confusion_matrix(model, y_test, x_train, predictions),
        'cross_val_score': get_cross_val_score(model, x_all, y_all),
        'precision_score': get_precision_score(y_test, predictions),
        'recall_score': get_recall_score(y_test, predictions),
        'accuracy_score': get_accuracy_score(y_test, predictions),
        'f1_score': get_f1_score(y_test, predictions),
        'roc_auc_score': get_roc_auc_score(y_test, predictions),
    }

def get_mse(y_train, predicted):
    mse = mean_squared_error(y_train, predicted)
    print('mse: ', mse)
    return mse

def get_rmse(y_train, predicted):
    rmse = np.sqrt(get_mse(y_train, predicted))
    print('rmse: ', rmse)
    return rmse

def get_confusion_matrix(model, y_test, x_train, predicted):
    # pred = cross_val_predict(model, x_train, y_train, cv=3)
    conf_mtx = confusion_matrix(y_test, predicted)
    print('confusion matrix: ', conf_mtx)
    return conf_mtx

def get_cross_val_score(model, x_all, y_all):
    cvs = cross_val_score(model, x_all, y_all, cv=5)
    print('cross validation score: ', cvs)
    print('cross validation mean: ', cvs.mean())
    print('cross validation std: ', cvs.std())
    print('cross validation rmse: ', np.sqrt(-cvs))
    return cvs

def get_accuracy_score(y_test, predictions):
    acc = accuracy_score(y_test, predictions)
    print('accuracy score: ', acc)
    return acc

def get_precision_score(y_train, predictions):
    ps = precision_score(y_train, predictions)
    print('precision score: ', ps)
    return ps

def get_recall_score(y_train, predictions):
    rs = recall_score(y_train, predictions)
    print('recall score: ', rs)
    return rs

def get_f1_score(y_train, predictions):
    f1 = f1_score(y_train, predictions)
    print('f1 score: ', f1)
    return f1

def get_roc_auc_score(y_test, predictions):
    # pred = cross_val_predict(
    #     model, x_train, y_train, cv=3,
    #     # method="decision_function"
    # )
    ras = roc_auc_score(y_test,predictions)
    print('roc auc score: ', ras)
    return ras