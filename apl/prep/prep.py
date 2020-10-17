import pandas as pd

def sample(df, ycolm='Failure', n:int=3, **kwargs):
    nve = df[df[ycolm]==0]
    pve = df[df[ycolm]==1]
    npve, nnve = n*len(pve), (n+1)*len(pve)
    data = [pve.sample(npve, replace=True, **kwargs),
            pve,
            nve.sample(nnve, replace=True, **kwargs)]
    data = pd.concat(data, ignore_index=True)
    return data

def extract_int(df:pd.DataFrame, colms:list):
    tsf_df = pd.DataFrame(index=df.index)
    for colm in colms:
        tsf_df[colm] = df[colm].str.extract(rf'{colm}_(\d+)')
    tsf_df[colms] = tsf_df[colms].astype(int)
    return tsf_df


class preprocess:
    def __init__(self, operations, fitted_ohe_dict:dict, scale=True):
        from apl.prep.ohe import ohe_transformer
        self.operations = operations
        self.ohe = ohe_transformer(fitted_ohe_dict)
        self.scale = scale

    def fit(self, x, y=None):
        print('fitting...')
        from sklearn.preprocessing import StandardScaler
        colms = self.operations['extract_int']
        selected = x.loc[x.Failure==0, colms]
        extract = extract_int(selected, colms)
        self.int_scaler = StandardScaler().fit(extract)

        colms = self.operations['standardise']
        selected = x.loc[x.Failure==0, colms]
        self.scaler = StandardScaler().fit(selected)
        return self

    def transform(self, x, y=None):
        print('transforming...')
        colms = self.operations['extract_int']
        extract = extract_int(x, colms)
        if self.scale:
            extract = self.int_scaler.transform(extract)
            extract = pd.DataFrame(extract, columns=colms)

        encoded = self.ohe.transform(x)

        colms = self.operations['standardise']
        other = x[colms]
        if self.scale:
            other = self.scaler.transform(other)
            other = pd.DataFrame(other, columns=colms)
        ans = pd.concat([extract, encoded, other], axis=1)
        return ans

operations = {
'extract_int' : ['STATION_ID', 'MACHINEID',
                 'MACHINEID_TESTER', 'MODULE2_FACTORY',
                 'MODULE2_BUILD', 'MODULE3_TOOL',
                 'MODULE3_SUBMOD2', 'MODULE3_CTool'],

'oneHotEncode':['PRODUCT', 'LINE_ID',
                'MODULE1_Vendor', 'MODULE2_Vendor',
                'MODULE2_CODE', 'MODULE3_Vendor',
                'MODULE3_SUBMOD2_Config', 'MODULE3_PHASE',],

'standardise': [ 'MODULE2_X1', 'MODULE2_X2',
                 'MODULE2_X3', 'MODULE2_X4',
                 'MODULE2_X5', 'MODULE3_SUBMOD1',]   }
