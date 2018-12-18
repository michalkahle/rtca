import os
import pandas as pd
import pandas_access as mdb
import glob
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.decomposition import PCA
import warnings
# from IPython.core.debugger import set_trace

def load_dir(directory, **kwargs):
    files = [os.path.join(directory, f) for f in sorted(os.listdir(directory)) if f.lower().endswith('.plt')]
    data = load_files(files, **kwargs)
    if callable(kwargs.get('fix')):
        data = kwargs['fix'](data)
    data = normalize(data, **kwargs)
    return data

# barcode_re='_(A\\d{6})\\.PLT$'
def load_files(file_list, barcode_re=None, **kwargs):
    if barcode_re is not None:
        barcode_re = re.compile(barcode_re)
    rows_series = np.array(list('ABCDEFGHIJKLMNOP'))
    plates = {}
    ll = []
    for filename in file_list:
        # print(filename)
        org = load_file(filename)
        if barcode_re is not None:
            match = barcode_re.search(filename)
            if match is not None:
                barcode = match.group(1)
            else:
                raise Exception(
                    'barcdode_re="%s" not found in file name: "%s"' % (barcode_re.pattern, filename))
        else:
            barcode = np.nan

        org['barcode'] = barcode
        if not barcode in plates:
            plates[barcode] = [0, 0]
        org['otp'] = org['tp']
        org['tp'] += plates[barcode][1]
        plates[barcode][1] = org['tp'].max()
        plates[barcode][0] += 1
        org['file'] = plates[barcode][0]
        org['well'] = rows_series[org['row']] + (org['col'] + 1).astype(str) #.str.zfill(2)
        if barcode is not np.nan:
            org['well'] = barcode + '_' + org['well']
        ll.append(org)
    return pd.concat(ll, ignore_index=True)

def normalize(df, layout=None, **kwargs):
    # if barcode is not np.nan:
    #     df = df.groupby('barcode').apply(lambda x: normalize_plate(x, **kwargs))
    # else:
    #     df = normalize_plate(df, **kwargs)
    df = df.groupby('barcode').apply(lambda x: normalize_plate(x, **kwargs))

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12,4))
    df = df.groupby('well').apply(lambda x: normalize_well(x, fig))
    df = df.drop(kwargs.get('drop', ['dt', 'file', 'org', 'otp']), axis=1)
    if layout is not None:
        df = pd.merge(df, layout)
    df = df.sort_values(['tp', 'row', 'col'])
    df = df.reset_index(drop=True)
    return df

# zerotime_file=2, zerotime_point=4, zerotime_offset=120
def normalize_plate(plate, zerotime_file=1, zerotime_point=1, zerotime_offset=0, **kwargs):
    if plate.file.max() < zerotime_file:
        raise ValueError('Not enough files. Zero time point should be in file %i but only %i file(s) present.' % (zerotime_file, plate.file.max()))
    zerotime = plate[(plate.file == zerotime_file) & (plate.otp == zerotime_point)].dt.iloc[0]
    plate['time'] = pd.to_numeric(plate['dt'] - zerotime) / 3.6e12 + zerotime_offset / 3600
    return plate

def normalize_well(well, fig, spike_threshold=3):
    ax0, ax1 = fig.get_axes(); ax0.set_title('outliers'); ax1.set_title('spikes')
    well['ci'] = (well.org - well.org.loc[well.tp.idxmin()]) / 15
    outliers = well.loc[abs(well.ci) > 100].copy()
    if not outliers.empty:
        outliers['blocks'] = ((outliers.tp - outliers.tp.shift(1)) > 1).cumsum()
        ax0.plot(well.tp, well.ci, label=outliers.well.iloc[0])
        ax0.scatter(outliers.tp, outliers.ci, facecolors='none', edgecolors='r', label=None)
        ax0.legend(); ax0.set_xlabel('tp'); ax0.set_ylabel('ci')
        def fix_outlier(ol):
            try:
                fix = (well[well.tp == ol.tp.min()-1].ci.iloc[0] + well[well.tp == ol.tp.max()+1].ci.iloc[0]) /2
            except IndexError:
                if well.tp.min() < ol.tp.min():
                    fix = well[well.tp == ol.tp.min()-1].ci.iloc[0]
                else:
                    fix = well[well.tp == ol.tp.max()+1].ci.iloc[0]
            well.loc[ol.index, 'ci'] = fix
        outliers.groupby('blocks').filter(fix_outlier)
    s = well.ci
    spikes = well[(s - s.shift(1) > spike_threshold) & (s - s.shift(-1) > spike_threshold)]
    if not spikes.empty:
        ax1.plot(well.tp, well.ci, label=spikes.well.iloc[0])
        ax1.scatter(spikes.tp, spikes.ci, facecolors='none', edgecolors='r', label=None)
        ax1.legend(); ax1.set_xlabel('tp'); ax1.set_ylabel('ci')
        for ii, ol in spikes.iterrows():
            fix = (well[well.tp == ol.tp-1].ci.iloc[0] + well[well.tp == ol.tp+1].ci.iloc[0]) /2
            well.loc[ii, 'ci'] = fix

    ci_for_log = well['ci'].copy() + 1
    ci_for_log[ci_for_log < 0.5] = 0.5
    well['lci'] = np.log2(ci_for_log)

    if well.time.min() < 0:
        norm_point = np.where(well.time <= 0)[0][-1]
        norm_value = well.iloc[norm_point]['ci']
        if norm_value < 0.1: # negative values flip the curve and small values make it grow too fast
            warnings.warn('Negative or small CI at time zero. Well %s removed.' % well.well.iloc[0])
            return None
        well['nci'] = well['ci'] / norm_value
        nci = well.nci.copy() + 1
        nci[nci < 0.5] = 0.5
        well['lnci'] = np.log2(nci)
    else:
        well['nci'] = well['ci']
        well['lnci'] = well['lci']
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
    assert org.shape[0] > 0, '%s contains no data!' % filename
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

def plate_96():
    ll = pd.DataFrame(dict(
        row = np.repeat(np.arange(8),12),
        col = np.tile(np.arange(12),8),
        well = np.arange(96),
        edge=0))
    ll.edge[ll.row < 1] += 1
    ll.edge[ll.col < 1] += 1
    ll.edge[ll.row > ll.row.max() - 1] += 1
    ll.edge[ll.col > ll.col.max() - 1] += 1
    return ll

def plate_384():
    ll = pd.DataFrame(dict(row = np.repeat(np.arange(16),24), col = np.tile(np.arange(24),16), edge=0))
    for m in [1,2]:
        ll.edge[ll.row < m] += 1
        ll.edge[ll.col < m] += 1
        ll.edge[ll.row > ll.row.max() - m] += 1
        ll.edge[ll.col > ll.col.max() - m] += 1
    return ll

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


def plot3d(dd, color=None, factor=False, cmap='tab10', hover='well', publish=False):
    import plotly
    import plotly.plotly as py
    import plotly.graph_objs as go

    trace_params = {'mode': 'markers', 'hoverinfo':'name+text'}
    marker = {'colorscale': 'Jet', 'opacity': 1, 'size': 3}
    layout = {'height': 600, 'margin': {'b': 0, 'l': 0, 'r': 0, 't': 0}, 'paper_bgcolor': '#f0f0f0', 'width': 800,
             'scene': {'xaxis':{'title':'PC1'}, 'yaxis':{'title':'PC2'}, 'zaxis':{'title':'PC3'}}}

    # this is a hack to fix the hover texts on the first trace (we make a fake one which is broken)
    # https://github.com/plotly/plotly.py/issues/952
    traces = [go.Scatter3d(x=[0], y=[0], z=[0],
        marker={'color':'rgb(0, 0, 0)', 'opacity': 1, 'size': 0.1}, showlegend=False)]

    if factor: #(dd[color].dtype == np.dtype('O')) or
        tab10 = plt.get_cmap(cmap)
        def get_plotly_color(cm, n):
            return 'rgb' + str(cm(n, bytes=True)[:3])
        for ii, (name, sg) in enumerate(dd.groupby(color)):
            marker['color'] = get_plotly_color(tab10, ii)
            trace_params['marker'] = marker
            trace_params['name'] = name
            trace = go.Scatter3d(x=sg.PC1, y=sg.PC2, z=sg.PC3, hovertext=sg[hover], **trace_params)
            traces.append(trace)
            layout['showlegend'] = True
    else:
        marker['color'] = dd[color].astype('category').cat.codes.values
        marker['colorbar'] = go.ColorBar(title=color, thickness=10, len=.3, y=.8)
        trace_params['marker'] = marker
        trace = go.Scatter3d(x=dd.PC1, y=dd.PC2, z=dd.PC3, hovertext=dd.id, **trace_params)
        traces.append(trace)
        layout['showlegend'] = False

    fig = go.Figure(data=traces, layout=go.Layout(layout))
    if publish:
        plotly.iplot(fig)
    else:
        plotly.offline.iplot(fig)

def plot3d_tsne(dd, color=None, factor=False, cmap='tab10', hover='well', publish=False):
    import plotly
    import plotly.plotly as py
    import plotly.graph_objs as go

    trace_params = {'mode': 'markers', 'hoverinfo':'name+text'}
    marker = {'colorscale': 'Jet', 'opacity': 1, 'size': 3}
    layout = {'height': 600, 'margin': {'b': 0, 'l': 0, 'r': 0, 't': 0}, 'paper_bgcolor': '#f0f0f0', 'width': 800,
             'scene': {'xaxis':{'title':'tSNE1'}, 'yaxis':{'title':'tSNE2'}, 'zaxis':{'title':'tSNE3'}}}

    # this is a hack to fix the hover texts on the first trace (we make a fake one which is broken)
    # https://github.com/plotly/plotly.py/issues/952
    traces = [go.Scatter3d(x=[0], y=[0], z=[0],
        marker={'color':'rgb(0, 0, 0)', 'opacity': 1, 'size': 0.1}, showlegend=False)]

    if factor: #(dd[color].dtype == np.dtype('O')) or
        tab10 = plt.get_cmap(cmap)
        def get_plotly_color(cm, n):
            return 'rgb' + str(cm(n, bytes=True)[:3])
        for ii, (name, sg) in enumerate(dd.groupby(color)):
            marker['color'] = get_plotly_color(tab10, ii)
            trace_params['marker'] = marker
            trace_params['name'] = name
            trace = go.Scatter3d(x=sg.tSNE1, y=sg.tSNE2, z=sg.tSNE3, hovertext=sg[hover], **trace_params)
            traces.append(trace)
            layout['showlegend'] = True
    else:
        marker['color'] = dd[color].astype('category').cat.codes.values
        marker['colorbar'] = go.ColorBar(title=color, thickness=10, len=.3, y=.8)
        trace_params['marker'] = marker
        trace = go.Scatter3d(x=dd.tSNE1, y=dd.tSNE2, z=dd.tSNE3, hovertext=dd.id, **trace_params)
        traces.append(trace)
        layout['showlegend'] = False

    fig = go.Figure(data=traces, layout=go.Layout(layout))
    if publish:
        plotly.iplot(fig)
    else:
        plotly.offline.iplot(fig)

def pca(dfl, n=3, plot=True, x='lnci'):
    dfw = prepare_unstack(dfl, x=x).unstack('tp')
    pca_m = PCA(n_components=n)
    X_pca = pca_m.fit_transform(dfw.values)
    columns = ['PC' + str(x) for x in range(1,n+1)]
    X_pca_df = pd.DataFrame(X_pca, index=dfw.index, columns=columns).reset_index()

    if plot:
        plot_explained_variance(pca_m)
    # print('residual = %.3f' % (1-pca_m.explained_variance_ratio_.sum()))
    return pca_m, X_pca_df

def pca_filter(dfl, n=3, plot=True, x='lnci'):
    dfl = dfl.sort_values(['well', 'tp']).reset_index(drop=True)
    dfl = prepare_unstack(dfl, x=x)
    dfw = dfl.unstack('tp')
    pca_m = PCA(n_components=n)
    X_pca = pca_m.fit_transform(dfw.values)

    X_components_df = pd.DataFrame(pca_m.components_.T,
        index=dfw.columns.droplevel(),
        columns=pd.MultiIndex.from_product([['w'], range(1,n+1)], names=[None, 'pc'])
        ).stack().reset_index()

    filtered = pca_m.inverse_transform(X_pca)
    dfl['filtered'] = pd.DataFrame(filtered, index=dfw.index, columns=dfw.columns).stack('tp')
    dfl['residual'] = dfl[x] - dfl['filtered']

    if plot:
        plot_explained_variance(pca_m)
    print('residual = %.3f' % (1-pca_m.explained_variance_ratio_.sum()))
    return dfl.reset_index(), X_components_df

def prepare_unstack(dfw, x='lnci'):
    to_drop = {'time', 'ci', 'nci', 'lci', 'lnci'} & set(dfw.columns)
    to_drop.remove(x)
    dfw = dfw.drop(to_drop, axis=1)
    i_cols = set(dfw.columns)
    i_cols.discard(x)
    i_cols.discard('tp')
    return dfw.set_index(list(i_cols) + ['tp'])

def plot_explained_variance(pca_m):
    evr = pca_m.explained_variance_ratio_
    residual = 1-evr.sum()
    n = pca_m.n_components
    ind = np.arange(n+1, 0, -1)
    col = np.r_[np.tile('g', n), np.array(['r'])]
    evr = np.r_[evr, np.array([residual])]
    fig, ax = plt.subplots(figsize=(4,1))
    ax.barh(ind, evr, color=col)
    ax.set_yticks(ind)
    ax.set_yticklabels(['PC' + str(x) for x in range(1, n + 1)] + ['residual'])
    ax.set_xlim(0,1)
    ax.set_xlabel('explained variance')
    ax.set_ylabel('components')

def calculate_z_score(p, components=None):
    res = np.zeros_like(p.PC1)
    for pc in p.columns[p.columns.str.contains('PC')][:components]:
        res += ((p[pc] - p[pc].mean())/ p[pc].std())**2
    return np.sqrt(res)

def annotate(df):
    import requests
    import urllib3
    def escape(x):
        if isinstance(x, (tuple, list)):
            return '[%s]' % (','.join(map(escape, x)))
        return '"%s"' % x
    def parse(rec):
        plate, well, dd = rec
        dd['plate'] = plate
        dd['well'] = well
        return dd
    df = df.copy()
    urllib3.disable_warnings()
    login_url = 'https://dbtest.screenx.cz/accounts/login/'
    s = requests.Session()
    s.get(login_url, verify=False)
    login_data = {
        'username' : 'kahle',
        'password' : 'tajne heslo',
        'csrfmiddlewaretoken' : s.cookies['csrftoken']
    }
    r1 = s.post(login_url, login_data, headers={'Referer' : login_url}, verify=False)
    rows_series = np.array(list('ABCDEFGHIJKLMNOP'))
    df['well'] = rows_series[df['row']] + (df['col'] + 1).astype(str)
    df['plate'] = df.cp.apply(lambda cp: 'CP-0%s-00' % cp)
    ll = []
    for ii in range(0, df.shape[0], 200):
        query = df[['plate', 'well']].iloc[ii: ii+200].values.tolist()
        r2 = s.get('https://dbtest.screenx.cz/api/get_samples?query=%s' % escape(query), verify=False)
        anot = r2.json()
        res_df = pd.DataFrame(list(map(parse, anot)))
        ll.append(res_df)
    res_df = pd.concat(ll)
    return pd.merge(res_df, df).drop(['plate', 'well'], axis=1)

if __name__ == '__main__':
    pass