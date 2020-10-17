import pandas as pd
from apl.data import read_data
from apl.prelim import xcolms, ycolm

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

# if __name__=='__main__':
df = read_data()
df.drop_duplicates(inplace=True)
df = df[xcolms+[ycolm]]

data = sample(df, ycolm, 3, random_state=1234)

colms = operations['extract_int']
ts = extract_int(data, colms)


from apl.ohe import fitted_ohe, transform
# fit ohe over all data
ohe = fitted_ohe(operations['oneHotEncode'], df)

t = transform(ohe)
encoded = t.transform(data)



from sklearn.preprocessing import StandardScaler
colms = operations['standardise']
x = data.loc[data.Failure==0, colms]
scaler = StandardScaler().fit(x)
scaled = scaler.transform(data[colms])
scaled = pd.DataFrame(scaled, columns=colms)
