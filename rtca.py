import os
import pandas as pd
import pandas_access as mdb
import glob
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

def load_dir(dir, **kwargs):
    files = [os.path.join(dir, f) for f in sorted(os.listdir(dir)) if f.lower().endswith('.plt')]
    data = load_files(files, **kwargs)
    return data

def load_files(file_list, barcode_re='_(A\\d{6})\\.PLT$', **kwargs):
    regex = re.compile(barcode_re)
    rows_series = np.array(list('ABCDEFGHIJKLMNOP'))
    plates = {}
    ll = []
    for filename in file_list:
        print(filename)
        org = load_file(filename)
        match = regex.search(filename)
        barcode = match.group(1) if match else np.nan
        org['barcode'] = barcode
        if not barcode in plates:
            plates[barcode] = [0, 0]
        org['otp'] = org['tp']
        org['tp'] += plates[barcode][1]
        plates[barcode][1] += max(org['tp'])

        plates[barcode][0] += 1
        org['file'] = plates[barcode][0]
        org['well'] = rows_series[org['row']] + (org['col'] + 1).astype(str)
        if barcode is not np.nan:
            org['well'] = barcode + '_' + org['well']
        ll.append(org)
    df = pd.concat(ll, ignore_index=True)
    df = df.groupby('barcode').apply(normalize_plate)
    df = df.groupby('barcode').apply(normalize_plate)
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12,4))
    df = df.groupby('well').apply(lambda x: normalize_well(x, fig))
    df = df.drop(['dt', 'org', 'file', 'otp'], axis=1)
    df = df.sort_values(['tp', 'row', 'col'])
    df = df.reset_index(drop=True)
    return df

def normalize_plate(plate, zerotime_file=2, zerotime_point=3, zerotime_offset=120):
    zerotime = plate[(plate.file == zerotime_file) & (plate.otp == zerotime_point)].dt.iloc[0]
    plate['time'] = pd.to_numeric(plate['dt'] - zerotime) / 3.6e12 + zerotime_offset / 3600
    return plate

def normalize_well(well, fig, spike_threshold=3):
    ax0, ax1 = fig.get_axes()
    s = well.org
    well['ci'] = (s - s.iloc[0]) / 15
    outliers = well.loc[abs(well.ci) > 100].copy()
    if not outliers.empty:
        outliers['blocks'] = ((outliers.tp - outliers.tp.shift(1)) > 1).cumsum()
        ax0.plot(well.tp, well.ci, label=outliers.well.iloc[0])
        ax0.scatter(outliers.tp, outliers.ci, facecolors='none', edgecolors='r', label=None)
        ax0.legend(); ax0.set_xlabel('tp'); ax0.set_xlabel('ci'); ax0.set_title('outliers')
        def fix_outlier(ol):
            fix = (well[well.tp == ol.tp.min()-1].ci.iloc[0] + well[well.tp == ol.tp.max()+1].ci.iloc[0]) /2
            well.loc[ol.index, 'ci'] = fix
        outliers.groupby('blocks').filter(fix_outlier)
    s = well.ci
    spikes = well[(s - s.shift(1) > spike_threshold) & (s - s.shift(-1) > spike_threshold)]
    if not spikes.empty:
        ax1.plot(well.tp, well.ci, label=spikes.well.iloc[0])
        ax1.scatter(spikes.tp, spikes.ci, facecolors='none', edgecolors='r', label=None)
        ax1.legend(); ax1.set_xlabel('tp'); ax1.set_ylabel('ci'); ax1.set_title('spikes')
        for ii, ol in spikes.iterrows():
            fix = (well[well.tp == ol.tp-1].ci.iloc[0] + well[well.tp == ol.tp+1].ci.iloc[0]) /2
            well.loc[ii, 'ci'] = fix
            
    well['nci'] = well['ci'] / well.iloc[np.where(well.time <= 0)[0][-1]]['ci']
    return well

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
    org = org[['tp', 'row', 'col', 'dt', 'org']]
    org = org.sort_values(by=['tp', 'row', 'col'])
    org = org.reset_index(drop=True)
    return org

# def scan():
#     for dirpath, dirnames, files in os.walk('data'):
#         for name in files:
#             if name.lower().endswith('.plt'):
#                 filename = os.path.join(dirpath, name)
#                 load_file(filename)

def plot_overview(df, x='time', y='nci', group='barcode'):
    fig, ax = plt.subplots(1, 1, figsize=(24, 16))
    ax.set_prop_cycle(mpl.rcParams['axes.prop_cycle'][:3])
    r_max, c_max = df.row.max(), df.col.max()
    x_min, y_min, x_max, y_max = df[x].min(), df[y].min(), df[x].max(), df[y].max()
    x_offset = (x_max - x_min)
    y_offset = (y_max - y_min)
    _, ylim = ax.set_ylim([0, (r_max + 2) * y_offset * 1.1])
    _, xlim = ax.set_xlim([0, (c_max + 2) * x_offset * 1.1])
    plt.setp(ax, 'frame_on', False)
    ax.set_xticks([])
    ax.set_yticks([])
    rows, cols, grs = df.row.unique(), df.col.unique(), df[group].unique()
    bcg = []
    for row in rows:
        rr = df[df.row==row]
        y_pos = ylim - (row + 2) * y_offset * 1.1
        # row label
        ax.text(0.75*x_offset, y_pos+.5*y_offset, row, size=20, ha='right', va='center')
        for col in cols:
            cc = rr[rr.col==col]
            x_pos = (col + 1) * x_offset * 1.1
            bcg.append(mpl.patches.Rectangle((x_pos, y_pos), x_offset, y_offset))
            # col label
            if row == 0:
                ax.text(x_pos+0.5*x_offset, y_pos+1.25*y_offset, col, size=20, ha='center')
            for gr in grs:
                sf = cc[cc[group]==gr]
                ax.plot(sf[x] + x_pos - x_min, sf[y] + y_pos - y_min, '-')
    pc = mpl.collections.PatchCollection(bcg, facecolor='#f0f0f0')
    ax.add_collection(pc)

def plot(df, x='time', y='nci', color=None):
    from collections import OrderedDict
    fig, ax = plt.subplots(figsize=(18,12))
    # df = df.sort_values(x)

    if color is None:
        for well, group in df.groupby('well'):
            ax.plot(group[x], group[y], color='k')
        handles, labels = ax.get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
    else:
        groups = df[color].unique()
        cmap = plt.get_cmap('gist_rainbow')
        # cmap = plt.get_cmap('viridis')
        color_map = dict(zip(groups, cmap((groups - groups.min()) / (groups.max()-groups.min()))))
        
        for (cc, well), group in df.groupby([color, 'well']):
            ax.plot(group[x], group[y], color=color_map[cc]*.75, label=cc)
        handles, labels = ax.get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), title=color)


if __name__ == '__main__':
    # scan()
    load_file('data/170202_dose_response.plt')
    # load_file('data/2017-11-22_cytostatics/1711171708HT1_A115534.PLT')