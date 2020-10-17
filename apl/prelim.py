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
 'PRODUCT', #categorical
 'STATION_ID',#categorical
 'MACHINEID',#categorical
 'MACHINEID_TESTER',#categorical
 'LINE_ID',#categorical

 'MODULE1_Vendor',#categorical

 'MODULE2_Vendor',#categorical
 'MODULE2_CODE',#categorical
 'MODULE2_FACTORY',# extract int
 'MODULE2_X1', # int
 'MODULE2_X2', # int
 'MODULE2_X3', # int
 'MODULE2_X4', # int
 'MODULE2_X5', # int
 'MODULE2_BUILD', # extract int

 'MODULE3_Vendor',#categorical
 'MODULE3_TOOL', # extract int
 'MODULE3_SUBMOD1', # int
 'MODULE3_SUBMOD2', # extract int
 'MODULE3_SUBMOD2_Config',#categorical
 'MODULE3_PHASE',#categorical
 'MODULE3_CTool' # extract int
 ]
