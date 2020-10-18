from apl.data import read_data
from apl.prep.ohe import fitted_ohe
from apl.prep.prelim import xcolms, ycolm
from apl.prep.prep import preprocess, operations, sample
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

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


clf = DecisionTreeClassifier()
clf.fit(x, y)
p = clf.predict(x)

plt.hist(p[:,0], label='0s', edgecolor='k', bins=50)



# def rms_error(model, X, y):
#     y_pred = model.predict(X)
#     return np.sqrt(np.mean((y - y_pred) ** 2))

# val_train, val_test = validation_curve(PolynomialRegression(), X, y,
#                                        'polynomialfeatures__degree',
#                                        degree, cv=7, scoring=rms_error)
