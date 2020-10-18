from apl.data import read_data
from apl.prep.prelim import xcolms, ycolm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def get_data():
    from sklearn.model_selection import train_test_split
    from apl.prep.prep import sample
    df = read_data()
    df.drop_duplicates(inplace=True)
    df = df[xcolms+[ycolm]]
    data = sample(df, ycolm, 3, random_state=1234)
    x,y = data, data['Failure']
    xtrain, xtest, ytrain, ytest = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y)
    return df, xtrain, xtest, ytrain, ytest

def rms_error(model, X, y):
    y_pred = model.predict(X)
    return np.sqrt(np.mean((y - y_pred) ** 2))

def plot_learning_curve(clf, xtrain, ytrain):
    from sklearn.model_selection import learning_curve

    train_sizes, train_scores, test_scores, = learning_curve(
                           clf, xtrain, ytrain, cv=5,
                           n_jobs=2, scoring=rms_error,
                           train_sizes=np.linspace(.1, 1.0, 5),
                           verbose=1)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.figure(figsize=(12,6))
    plt.xlabel('size')
    plt.ylabel('error')
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training error")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation error")
    plt.legend(loc="best");

def bias_variance(clf, xtrain, ytrain):
    # check for regularization paramter
    from sklearn.model_selection import GridSearchCV
    parameters = {'C':np.logspace(-2, 8, 8)}
    search = GridSearchCV(clf, parameters, cv=5, n_jobs=2,
                          scoring=rms_error, verbose=1,
                         return_train_score=True)
    search.fit(xtrain, ytrain)
    df = pd.DataFrame(search.cv_results_)
    df = df[['param_C', 'mean_train_score', 'mean_test_score']]
    plt.figure(figsize=(12,6))
    plt.xlabel(r'regularization parameter ($\lambda$)')
    plt.ylabel('error')
    px, py = 1/df['param_C'], df['mean_train_score']
    plt.plot(px,py, 'o-', label='train')
    px, py = 1/df['param_C'], df['mean_test_score']
    plt.plot(px,py, 'o-', label='test')
    plt.xscale('log')
    plt.legend();
