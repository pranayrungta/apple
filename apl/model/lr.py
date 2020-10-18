from apl.data import read_data
from apl.prep.ohe import fitted_ohe
from apl.prep.prelim import xcolms, ycolm
from apl.prep.prep import preprocess, operations, sample
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

df = read_data()
df.drop_duplicates(inplace=True)
df = df[xcolms+[ycolm]]
# fit ohe over all data
ohe = fitted_ohe(operations['oneHotEncode'], df)

data = sample(df, ycolm, 3, random_state=1234)
x,y = data, data[['Failure']]
p = preprocess(operations, ohe, scale=True)
p.fit(x,y)
x = p.transform(x,y)



clf = LogisticRegression(penalty='l1', solver='liblinear')
clf.fit(x, y.Failure)
p = clf.predict_proba(x)

plt.figure()
zero = p[y.Failure==0]
one = p[y.Failure==1]
plt.hist(zero[:,0], label='0s', color='b', edgecolor='k', bins=50, alpha=0.5, range=(0,1))
plt.hist(one[:,0], label='1s', color='r', edgecolor='k', bins=50, alpha=0.5, range=(0,1))
plt.legend()

plt.figure()
p = x.dot(clf.coef_[0])
zero = p[y.Failure==0]
one = p[y.Failure==1]
plt.hist(zero.values, label='0s',color='b',  edgecolor='k', bins=50, alpha=0.5)#, range=(0,1))
plt.hist(one.values, label='1s', color='r', edgecolor='k', bins=50, alpha=0.5)#, range=(0,1))
plt.legend()



# investigating gaussian
good_ones = x.loc[one[one>1].index]
good_zero = x.loc[zero[zero<1].index]
one_vals = clf.coef_*good_ones
zero_vals = clf.coef_*good_zero
plt.hist(zero_vals.sum(axis=1), label='0s',color='b',  edgecolor='k', bins=50, alpha=0.5)#, range=(0,1))
plt.hist(one_vals.sum(axis=1), label='1s', color='r', edgecolor='k', bins=50, alpha=0.5)#, range=(0,1))
plt.legend()

i = one_vals.mean().reset_index()
i.sort_values(0, ascending=False, inplace=True)



# investigating feature
p = data.MODULE2_X3
zero = p[y.Failure==0]
one = p[y.Failure==1]
plt.hist(zero.values, label='0s', edgecolor='k', bins=50, alpha=0.5)#, range=(0,1))
plt.hist(one.values, label='1s', edgecolor='k', bins=50, alpha=0.5)#, range=(0,1))
plt.legend()




p = x.LINE_ID_7
zero = p[y.Failure==0]
one = p[y.Failure==1]
plt.hist(zero.values, label='0s', edgecolor='k', bins=50, alpha=0.5)#, range=(0,1))
plt.hist(one.values, label='1s', edgecolor='k', bins=50, alpha=0.5)#, range=(0,1))
plt.legend()

# def rms_error(model, X, y):
#     y_pred = model.predict(X)
#     return np.sqrt(np.mean((y - y_pred) ** 2))

# val_train, val_test = validation_curve(PolynomialRegression(), X, y,
#                                        'polynomialfeatures__degree',
#                                        degree, cv=7, scoring=rms_error)
