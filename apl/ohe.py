import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def fitted_ohe(colms, df):
    ohe = {}
    for colm in colms:
        ohe[colm] = OneHotEncoder().fit(df[[colm]])
    return ohe

class ohe_transformer:
    def __init__(self, ohe):
        self.ohe = ohe

    def fit(self):
        return self

    def transform(self, df):
        ans = []
        for colm in self.ohe.keys():
            enc = self.ohe[colm]
            m = enc.transform(df[[colm]])
            m = pd.DataFrame(m.todense(),
                    columns=enc.categories_[0], dtype=int)
            ans.append(m)
        ans = pd.concat(ans, axis=1)
        return ans
