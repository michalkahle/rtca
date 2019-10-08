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
from math import sqrt
from functools import partial
# from IPython.core.debugger import set_trace

def load_dir(directory, **kwargs):
    files = [os.path.join(directory, f) for f in sorted(os.listdir(directory)) if f.lower().endswith('.plt')]
    data = load_files(files, **kwargs)
    if callable(kwargs.get('fix')):
        data = kwargs['fix'](data)
    data = normalize(data, **kwargs)
    return data

# barcode_re='_(A\\d{6})\\.PLT$'
def load_files(file_list, barcode_re, **kwargs):
    barcode_re = re.compile(barcode_re)
    plates = {}
    ll = []
    for filename in file_list:
        # print(filename)
        org = load_file(filename)
        match = barcode_re.search(filename)
        if match is not None:
            barcode = int(match.group(1))
        else:
            raise Exception(
                'barcdode_re="%s" not found in file name: "%s"' % (barcode_re.pattern, filename))
        org['ap'] = barcode
        if not barcode in plates:
            plates[barcode] = [0, 0]
        org['otp'] = org['tp']
        org['tp'] += plates[barcode][1]
        plates[barcode][1] = org['tp'].max()
        plates[barcode][0] += 1
        org['file'] = plates[barcode][0]
        ll.append(org)
    return pd.concat(ll, ignore_index=True).reset_index(drop=True)

def normalize(df, layout=None, **kwargs):
    df = df.groupby('ap').apply(lambda x: normalize_plate(x, **kwargs))

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12,4))
    df = df.groupby(['ap', 'well']).apply(lambda x: normalize_well(x, fig))
    df = df.drop(kwargs.get('drop', ['dt', 'file', 'org', 'otp']), axis=1)
    if layout is not None:
        df = pd.merge(df, layout)
    # df = df.sort_values(['tp', 'well'])
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
        if norm_value < 0.1: # negative values here flip the curve and small values make it grow too fast
            warnings.warn('Negative or small CI at time zero. Well %s removed.' % well.well.iloc[0])
            return None
        well['nci'] = well['ci'] / norm_value
        nci = well['nci'].copy() + 1
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
    org = mdb.read_table(filename, 'Org10K').drop('TestOrder', axis=1)
    assert org.shape[0] > 0, '%s contains no data!' % filename
    org['Row'] = org['Row'].map(ord) - 65
    org = org.rename({'TimePoint':'tp', 'Row':'row'}, axis=1).set_index(['tp', 'row']).sort_index()
    n_cols = org.shape[1]
    org.columns = pd.MultiIndex.from_product([['org'], range(n_cols)], names=[None, 'col'])
    org = org.stack('col').reset_index()
    org['org'] = org['org'].astype(float)
    org['well'] = org['row'] * (org['col'].max() + 1) + org['col']
    org.drop(['row', 'col'], axis=1, inplace=True)
    org = org.merge(ttimes)
    org = org[['tp', 'well', 'dt', 'org']]
    return org

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
    # ll['spiral'] = spiral(16, 24).flatten()
    return ll

def spiral(m, n):
    M = np.zeros([m, n], dtype=int)
    i, j = 0, 0 # location of "turtle"
    di, dj = 0, 1 # direction of movement
    h = (np.min([m,n]))/2
    for ii in range(m * n):
        M[i, j] = ii
        if (i < h and (i == j+1 or i+1 == n-j)) or (i >= m-h and (m-i == n-j or m-i == j+1)):
            di, dj = dj, -di # turn clockwise
        i, j = i + di, j + dj
    return M

def plot_overview(df, x='time', y='nci', group='ap'):
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


def plot3d(dd, color=None, factor=False, cmap='tab10', hover='well', publish=False, projection='PCA'):
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

    if projection == 'PCA':
        xc, yc, zc = 'PC1', 'PC2', 'PC3'
    elif projection == 'tSNE':
        xc, yc, zc = 'tSNE1', 'tSNE2', 'tSNE3'

    if factor: #(dd[color].dtype == np.dtype('O')) or
        tab10 = plt.get_cmap(cmap)
        def get_plotly_color(cm, n):
            return 'rgb' + str(cm(n, bytes=True)[:3])
        for ii, (name, sg) in enumerate(dd.groupby(color)):
            marker['color'] = get_plotly_color(tab10, ii)
            trace_params['marker'] = marker
            trace_params['name'] = name
            trace = go.Scatter3d(x=sg[xc], y=sg[yc], z=sg[zc], hovertext=sg[hover], **trace_params)
            traces.append(trace)
            layout['showlegend'] = True
    else:
        marker['color'] = dd[color].values#.astype('category').cat.codes.values
        marker['colorbar'] = dict(title=color, thickness=10, len=.3, y=.8)
        marker['showscale'] = True
        trace_params['marker'] = marker
        trace = go.Scatter3d(x=dd[xc], y=dd[yc], z=dd[zc], hovertext=dd.compound, **trace_params)
        traces.append(trace)
        layout['showlegend'] = False

    fig = go.Figure(data=traces, layout=go.Layout(layout))
    if publish:
        plotly.iplot(fig)
    else:
        plotly.offline.iplot(fig)

def pca(dfl, n=3, plot=True, x='lnci', t='tp'):
    dfw = prepare_unstack(dfl, x=x, t=t).unstack(t)
    pca_m = PCA(n_components=n)
    X_pca = pca_m.fit_transform(dfw.values)
    columns = ['PC' + str(x) for x in range(1,n+1)]
    X_pca_df = pd.DataFrame(X_pca, index=dfw.index, columns=columns).reset_index()

    if plot:
        plot_explained_variance(pca_m)
    # print('residual = %.3f' % (1-pca_m.explained_variance_ratio_.sum()))
    return pca_m, X_pca_df

def pca_fit(dfl, n=3, plot=True, x='lnci', t='tp'):
    dfw = prepare_unstack(dfl, x=x, t=t).unstack(t)
    pca_m = PCA(n_components=n)
    X_pca = pca_m.fit_transform(dfw.values)

    # columns = ['PC' + str(x) for x in range(1,n+1)]
    # X_pca_df = pd.DataFrame(X_pca, index=dfw.index, columns=columns).reset_index()

    if plot:
        plot_explained_variance(pca_m)
    # print('residual = %.3f' % (1-pca_m.explained_variance_ratio_.sum()))
    return pca_m

def pca_components(model, time_index=None):
    shape = model.components_.shape
    if time_index is None:
        time_index = pd.Index(range(1, shape[1]+1), name='tp')
    else:
        time_index = pd.Index(time_index, name='time')
    return pd.DataFrame(model.components_.T,
        index=time_index,
        columns=pd.MultiIndex.from_product([['w'], range(1,shape[0]+1)], names=[None, 'PC'])
        ).stack().reset_index()

def pca_reconstruct(model, df_pca, time_index=None):
    shape = model.components_.shape
    if time_index is None:
        time_index = range(1, shape[1]+1)
    time_index = pd.MultiIndex.from_product([['lnci'], time_index], names=[None, 'time'])
    df_pca.set_index(list(df_pca.columns[~df_pca.columns.str.contains('PC')]), inplace=True)
    filtered = model.inverse_transform(df_pca.values)
    df = pd.DataFrame(filtered, index=df_pca.index, columns=time_index)
    return df.stack().reset_index()

def pca_filter(dfl, n=3, plot=True, x='lnci', t='tp'):
    dfl = dfl.sort_values(['well', t]).reset_index(drop=True)
    dfl = prepare_unstack(dfl, x=x, t=t)
    dfw = dfl.unstack(t)
    pca_m = PCA(n_components=n)
    X_pca = pca_m.fit_transform(dfw.values)

    X_components_df = pd.DataFrame(pca_m.components_.T,
        index=dfw.columns.droplevel(),
        columns=pd.MultiIndex.from_product([['w'], range(1,n+1)], names=[None, 'pc'])
        ).stack().reset_index()

    filtered = pca_m.inverse_transform(X_pca)
    dfl['filtered'] = pd.DataFrame(filtered, index=dfw.index, columns=dfw.columns).stack(t)
    dfl['residual'] = dfl[x] - dfl['filtered']

    if plot:
        plot_explained_variance(pca_m)
    # print('residual = %.3f' % (1-pca_m.explained_variance_ratio_.sum()))
    return dfl.reset_index(), X_components_df

def prepare_unstack(dfw, x='lnci', t='tp'):
    to_drop = {'time', 'ci', 'nci', 'lci', 'lnci'} & set(dfw.columns)
    to_drop.remove(x)
    if t == 'time':
        to_drop.remove(t)
    dfw = dfw.drop(to_drop, axis=1)
    dfw.set_index([cc for cc in dfw.columns if cc != x], inplace=True)
    return dfw

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

def add_tsne(df, dims=2, perplexity=30):
    from sklearn.manifold import TSNE
    X = df.loc[:, df.columns.str.contains('PC')].values
    X_embedded = TSNE(n_components=dims, perplexity=perplexity).fit_transform(X)
    for n in range(dims):
        label = 'tSNE%s' % (n+1)
        df[label] = X_embedded[:, n]
    return df

def calculate_z_score(p, components=None):
    res = np.zeros_like(p.PC1)
    for pc in p.columns[p.columns.str.contains('PC')][:components]:
        res += ((p[pc] - p[pc].mean())/ p[pc].std())**2
    return np.sqrt(res)

def annotate(df, verbose=False):
    import requests
    import urllib3
    import json
    import screenx_credentials
    urllib3.disable_warnings()
    login_url = 'https://db.screenx.cz/accounts/login/'
    s = requests.Session()
    s.get(login_url, verify=False)
    login_data = {
        'username' : screenx_credentials.username,
        'password' : screenx_credentials.password,
        'csrfmiddlewaretoken' : s.cookies['csrftoken']
    }
    r1 = s.post(login_url, login_data, headers={'Referer' : login_url}, verify=False)

    df = df.copy()
    cols = df.columns
    to_drop = []

    if ('library' in cols or 'sourcename' in cols) and ('compound' in cols or 'samplename' in cols):
        if not 'sourcename' in cols:
            df['sourcename'] = df['library']
            to_drop.append('sourcename')
        if not 'samplename' in cols:
            df['samplename'] = df['compound']
            to_drop.append('samplename')
        dq = df[['sourcename', 'samplename']].copy()
    elif ('cp' in cols or 'plate' in cols or 's_plate' in cols) and ('well' in cols or 'well_an' in cols or 's_plate' in cols):
        if 'plate' in cols:
            pass
        elif 'cp' in cols:
            df['plate'] = df['cp'].apply(lambda cp: 'CP-0%s-00' % cp)
            to_drop.append('plate')
        elif 's_plate' in cols:
            df['plate'] = df['s_plate']
            to_drop.append('plate')

        if 'well_an' in cols:
            pass
        elif 'well_an' in cols:
            df['well_an'] = np.array(list('ABCDEFGHIJKLMNOP'))[df['well']//24] + (df['well']%24 + 1).astype(str)
            to_drop.append('well_an')
        elif 's_well' in cols:
            df['well_an'] = df['s_well']
            to_drop.append('well_an')
        dq = df[['plate', 'well_an']].rename({'well_an':'well'}, axis=1)
    else:
        raise ValueError('DataFrame does not contain necessary columns.')

    dq = dq.drop_duplicates()
    ll = []
    for ii in range(0, dq.shape[0], 50):
        query = dq.iloc[ii: ii+50].to_dict('records')
        query = json.dumps(query)
        if verbose:
            print(query)
        r2 = s.get('https://db.screenx.cz/api/get_samples?query=%s' % query, verify=False)
        if verbose:
            print(r2.text)
        ll.append(pd.DataFrame(r2.json()))
    res_df = pd.concat(ll).rename({'well':'well_an'}, axis=1)
    res_df = df.merge(res_df, how='left').drop(to_drop, axis=1) # , errors='ignore'
    return res_df


if __name__ == '__main__':
    pass