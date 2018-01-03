import os
import pandas as pd
import pandas_access as mdb
import glob
import re
import numpy as np

def load_dir(dir, **kwargs):
    files = [os.path.join(dir, f) for f in sorted(os.listdir(dir)) if f.lower().endswith('.plt')]
    data = load_files(files, **kwargs)
    return data

def load_files(file_list, barcode_re='_(A\\d{6})\\.PLT$', **kwargs):
    regex = re.compile(barcode_re)
    plates = {}
    for filename in file_list:
        print(filename)
        org = load_file(filename)
        match = regex.search(filename)
        barcode = match.group(1) if match else None
        org['barcode'] = barcode
        if not barcode in plates:
            plates[barcode] = []
        org['file'] = len(plates[barcode])
        plates[barcode].append(org)
    ll = []
    for barcode in plates:
        plate = pd.concat(plates[barcode], ignore_index=True)
        plate = normalize_plate(plate)
        ll.append(plate)
    df = pd.concat(ll, ignore_index=True)
    grouped = df.groupby(['barcode', 'well'], group_keys=False)
    df['oci'] = grouped['org'].apply(lambda s: s - s.iloc[0]) / 15
    df['ci'] = grouped['oci'].apply(spike_filter)
    df['nci'] = grouped.apply(lambda gr: gr['ci'] / gr.iloc[np.where(gr.time <= 0)[0][-1]]['ci'])
    # df['nci'] = df['ci']
    return df

def normalize_plate(plate, zerotime_file=1, zerotime_point=3, zerotime_offset=120):
    zerotime = plate[(plate.file == zerotime_file) & (plate.tp == zerotime_point)].dt.iloc[0]
    plate['time'] = pd.to_numeric(plate['dt'] - zerotime) / 3.6e12 + zerotime_offset / 3600
    # print(zerotime)
    return plate

def spike_filter(s, threshold=3):
    s[(s - s.shift(1) > threshold) & (s - s.shift(-1) > threshold)] = np.nan
    return s

def load_file(filename):
    if not os.path.isfile(filename):
        raise FileNotFoundError(filename)
    errors = mdb.read_table(filename, 'ErrLog')
    if len(errors) > 0:
        print('Errors reported in file %s:' % filename)
        print(errors)
    messages = mdb.read_table(filename, 'Messages')
    messages = messages[messages.MsgID == 'w01']
    if len(messages) > 0:
        messages.Message = messages.Message.replace(
            regex='Plate Scanned. Please check the connection on positions:\\s*', 
            value='Connection problem: ')
        print('Connection problems reported in file %s:' % filename)
        print(messages.drop(['MsgID', 'MsgOrder'], axis=1))
    ttimes = mdb.read_table(filename, 'TTimes')
    ttimes = ttimes[['TimePoint', 'TestTime']]
    ttimes.columns = ['tp', 'dt']
    ttimes['dt'] = pd.to_datetime(ttimes['dt'])
    ttimes = ttimes.set_index('tp')
    org = mdb.read_table(filename, 'Org10K').drop('TestOrder', axis=1)
    org = org.rename({'TimePoint':'tp'}, axis=1).set_index(['tp', 'Row'])
    n_cols = org.shape[1]
    org.columns = pd.MultiIndex.from_product([['org'], range(n_cols)], names=['value', 'col'])
    org = org.unstack('tp').sort_index()
    org = org.reset_index(drop=True)
    org.index.name = 'row'
    org = org.stack(['tp', 'col'])
    org.org = pd.to_numeric(org.org)
    org = ttimes.join(org)
    org = org.reset_index()
    org['well'] = org['row'] * n_cols + org['col']
    org = org.sort_values(by=['tp', 'well'])
    org = org[['tp', 'row', 'col', 'well', 'dt', 'org']]
    return org

# def scan():
#     for dirpath, dirnames, files in os.walk('data'):
#         for name in files:
#             if name.lower().endswith('.plt'):
#                 filename = os.path.join(dirpath, name)
#                 load_file(filename)




if __name__ == '__main__':
    # scan()
    load_file('data/170202_dose_response.plt')
    # load_file('data/2017-11-22_cytostatics/1711171708HT1_A115534.PLT')