'''
preliminary analysis:
    1. High data imbalance : need over, under sampling
    2. 6 fields to be removed : very unique or constant to be statistically significant
'''


def unique_values(df):
    vals = df.apply(lambda s : set(s.dropna().unique()))
    vals.name = 'unique_vals'
    vals.index.name = 'colm_name'
    vals = vals.to_frame()
    vals.insert(0, 'counts', df.apply(lambda s: s.nunique() ) )
    vals = vals.reset_index()
    vals.sort_values('counts', inplace=True)
    return vals

remove = [
 'COLOR', # same throughout
 'SERIALNUMBER',# almost unique
 'MODULE3_SEQ', # almost unique
 'MODULE1_SN',# almost unique
 'MODULE2_SN',# almost unique
 'MODULE3_SN',# almost unique
 'STARTTIME',# logical reason
 'MODULE1_WoM',# logical reason
 'MODULE2_WoM',# logical reason
 'TESTDURATION',# logical reason
 ]

ycolm = 'Failure'

xcolms = [
 'PRODUCT', #categorical -> 4
 'STATION_ID',# extract int -> 97
 'MACHINEID',# extract int -> 80
 'MACHINEID_TESTER',# extract int -> 172
 'LINE_ID',#categorical -> 35

 'MODULE1_Vendor',#categorical -> 3

 'MODULE2_Vendor',#categorical -> 10
 'MODULE2_CODE',#categorical -> 4
 'MODULE2_FACTORY',# extract int -> 10
 'MODULE2_X1', # int -> 42
 'MODULE2_X2', # int -> 19
 'MODULE2_X3', # int -> 24
 'MODULE2_X4', # int -> 56
 'MODULE2_X5', # int -> 24
 'MODULE2_BUILD', # extract int -> 9

 'MODULE3_Vendor',#categorical -> 3
 'MODULE3_TOOL', # extract int
 'MODULE3_SUBMOD1', # int -> 2
 'MODULE3_SUBMOD2', # extract int -> 103
 'MODULE3_SUBMOD2_Config',#categorical
 'MODULE3_PHASE',#categorical -> 5
 'MODULE3_CTool' # extract int -> 113
 ]
