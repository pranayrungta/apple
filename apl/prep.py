import pandas as pd


# def rms_error(model, X, y):
#     y_pred = model.predict(X)
#     return np.sqrt(np.mean((y - y_pred) ** 2))

# val_train, val_test = validation_curve(PolynomialRegression(), X, y,
#                                        'polynomialfeatures__degree',
#                                        degree, cv=7, scoring=rms_error)
def sample(df, ycolm='Failure', n:int=3, **kwargs):
    nve = df[df[ycolm]==0]
    pve = df[df[ycolm]==1]
    npve, nnve = n*len(pve), (n+1)*len(pve)
    data = [pve.sample(npve, replace=True, **kwargs),
            pve,
            nve.sample(nnve, replace=True, **kwargs)]
    data = pd.concat(data, ignore_index=True)
    return data


from apl.data import read_data
from apl.prelim import xcolms, ycolm
df = read_data()
df.drop_duplicates(inplace=True)
df = df[xcolms+[ycolm]]
data = sample(df, ycolm, 3,random_state=1234)

