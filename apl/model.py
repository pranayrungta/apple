# def rms_error(model, X, y):
#     y_pred = model.predict(X)
#     return np.sqrt(np.mean((y - y_pred) ** 2))

# val_train, val_test = validation_curve(PolynomialRegression(), X, y,
#                                        'polynomialfeatures__degree',
#                                        degree, cv=7, scoring=rms_error)
