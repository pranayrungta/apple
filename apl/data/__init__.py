import pandas as pd


def read_data():
    import pathlib
    path = pathlib.Path(__file__).parent
    path = path/'data_exercise2.csv'
    print('reading ', path)
    df = pd.read_csv(path)
    df['STARTTIME'] = pd.to_datetime(df.STARTTIME,
                                     dayfirst=False)
    return df
