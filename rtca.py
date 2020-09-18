import os
import json
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
import scipy.cluster.hierarchy
from sklearn import metrics
scipy.cluster.hierarchy.set_link_color_palette(['gray', 'goldenrod'])
# scipy.cluster.hierarchy.set_link_color_palette(['salmon', 'skyblue'])
# from IPython.core.debugger import set_trace
from plotnine import *
from time import time
from scipy.optimize import least_squares
from shutil import which

def warning_on_one_line(message, category, filename, lineno, line=None):
    return ' %s:%s: %s: %s\n' % (filename, lineno, category.__name__, message)
warnings.formatwarning = warning_on_one_line

# Load RTCA files ####################################################

def load_dir(directory, layout=None, **kwargs):
    """Load a directory of RTCA files.
    """
    if not which('mdb-schema'):
        raise Exception('`mdbtools` not installed. ')
    files = [f for f in sorted(os.listdir(directory)) if f.endswith('.PLT')]
    files = [os.path.join(directory, f) for f in files]
    data = load_files(files, **kwargs)
    if callable(kwargs.get('fix')):
        data = kwargs['fix'](data)
    data = normalize(data, **kwargs)
    if layout is not None:
        data = pd.merge(data, layout)
    return data

def load_files(file_list, barcode_re='_A(\\d{6})\\.PLT$', **kwargs):
    plates = {}
    ll = []
    for filename in file_list:
        org = load_file(filename)
        if barcode_re is None:
            barcode = 0
        else:
            match = re.search(barcode_re, filename)
            if match is not None:
                barcode = int(match.group(1))
            else:
                raise Exception(
                    'barcdode_re="%s" not found in file name: "%s"'
                    % (barcode_re, filename))
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

def normalize(df, **kwargs):
    df = df.groupby('ap').apply(lambda x: normalize_plate(x, **kwargs))

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12,4))
    ax0.set_title('outliers'); ax1.set_title('spikes')
    df = df.groupby(['ap', 'welln']).apply(lambda x: normalize_well(x, fig))
    df = df.drop(kwargs.get('drop', ['dt', 'org', 'otp']), axis=1) # 'file'
    df = df.reset_index(drop=True)
    return df

# t0_file=2, t0_point=4, t0_offset=120
def normalize_plate(plate, t0_file=1, t0_point=1, t0_offset=0, **kwargs):
    if plate.file.max() < t0_file:
        raise ValueError('Not enough files. Zero time point should be in file '
        '%i but only %i file(s) present.' % (t0_file, plate.file.max()))
    t0 = plate[(plate.file == t0_file) & (plate.otp == t0_point)].dt.iloc[0]
    plate['time'] = pd.to_numeric(plate['dt'] - t0) / 3.6e12 + t0_offset / 3600
    return plate

def normalize_well(well, fig, spike_threshold=3):
    ax0, ax1 = fig.get_axes()
    well['ci'] = (well['org'] - well['org'].loc[well['tp'].idxmin()]) / 15
    outliers = well.loc[abs(well['ci']) > 100].copy()
    if not outliers.empty:
        outliers['blocks'] = ((outliers['tp'] - outliers['tp'].shift(1)) > 1).cumsum()
        label = '%s-%s' % (outliers['ap'].iloc[0], welln2well_384(outliers['welln'].iloc[0]))
        ax0.plot(well['tp'], well['ci'], label=label)
        ax0.scatter(outliers['tp'], outliers['ci'], facecolors='none', edgecolors='r', label=None)
        ax0.legend(); ax0.set_xlabel('tp'); ax0.set_ylabel('ci')
        def fix_outlier(ol):
            try:
                fix = (well[well['tp'] == ol['tp'].min()-1]['ci'].iloc[0] + well[well['tp'] == ol['tp'].max()+1]['ci'].iloc[0]) /2
            except IndexError:
                if well['tp'].min() < ol['tp'].min():
                    fix = well[well['tp'] == ol['tp'].min()-1]['ci'].iloc[0]
                else:
                    fix = well[well['tp'] == ol['tp'].max()+1]['ci'].iloc[0]
            well.loc[ol.index, 'ci'] = fix
        outliers.groupby('blocks').filter(fix_outlier)
    s = well['ci']
    spikes = well[(s - s.shift(1) > spike_threshold) & (s - s.shift(-1) > spike_threshold)]
    if not spikes.empty:
        label = '%s-%s' % (spikes['ap'].iloc[0], welln2well_384(spikes['welln'].iloc[0]))
        ax1.plot(well['tp'], well['ci'], label=label)
        ax1.scatter(spikes['tp'], spikes['ci'], facecolors='none', edgecolors='r', label=None)
        ax1.legend(); ax1.set_xlabel('tp'); ax1.set_ylabel('ci')
        for ii, ol in spikes.iterrows():
            fix = (well[well['tp'] == ol['tp']-1]['ci'].iloc[0] + well[well['tp'] == ol['tp']+1]['ci'].iloc[0]) /2
            well.loc[ii, 'ci'] = fix

    ci_for_log = well['ci'].copy() + 1
    ci_for_log[ci_for_log < 0.5] = 0.5
    well['lci'] = np.log2(ci_for_log)

    if well['time'].min() < 0:
        norm_point = np.where(well['time'] <= 0)[0][-1]
        norm_value = well.iloc[norm_point]['ci']
        if norm_value < 0.1: # negative values here flip the curve and small values make it grow too fast
            warnings.warn('Negative or small CI at time zero. Well %s removed.' % welln2well_384(well['welln'].iloc[0]))
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
    org['welln'] = org['row'] * (org['col'].max() + 1) + org['col']
    org.drop(['row', 'col'], axis=1, inplace=True)
    org = org.merge(ttimes)
    org = org[['tp', 'welln', 'dt', 'org']]
    return org

_rows = np.array([chr(x) for x in range(65, 91)] + ['A' + chr(x) for x in range(65, 71)])

# Plotting ####################################################

def plot_overview(df, x='time', y='nci', group='ap', format=384):
    fig, ax = plt.subplots(1, 1, figsize=(24, 16))
    ax.set_prop_cycle(mpl.rcParams['axes.prop_cycle'][:3])
    n_cols = int(sqrt(format/2*3))
    r_max = (df['welln'] // n_cols).max() + 1
    c_max = (df['welln'] % n_cols).max() + 1

    x_min, y_min, x_max, y_max = df[x].min(), df[y].min(), df[x].max(), df[y].max()
    x_offset = (x_max - x_min)
    y_offset = (y_max - y_min)
    _, ylim = ax.set_ylim([0, (r_max + 2) * y_offset * 1.1])
    _, xlim = ax.set_xlim([0, (c_max + 2) * x_offset * 1.1])
    plt.setp(ax, 'frame_on', False)
    ax.set_xticks([])
    ax.set_yticks([])

    bcg = []
    grs = df[group].unique()
    for welln in range(format):
        well = df[df['welln']==welln]
        row = welln // n_cols
        col = welln % n_cols
        y_pos = ylim - (row + 2) * y_offset * 1.1
        # row label
        ax.text(0.75*x_offset, y_pos+.5*y_offset, _rows[row], size=20, ha='right', va='center')
        x_pos = (col + 1) * x_offset * 1.1
        bcg.append(mpl.patches.Rectangle((x_pos, y_pos), x_offset, y_offset))
        # col label
        if row == 0:
            ax.text(x_pos+0.5*x_offset, y_pos+1.25*y_offset, col + 1, size=20, ha='center')
        for gr in grs:
            sf = well[well[group]==gr]
            ax.plot(sf[x] + x_pos - x_min, sf[y] + y_pos - y_min, '-')
    pc = mpl.collections.PatchCollection(bcg, facecolor='#f0f0f0')
    ax.add_collection(pc)

def plot(df, x='time', y='nci', color=None):
    from collections import OrderedDict
    fig, ax = plt.subplots(figsize=(18,12))
    # df = df.sort_values(x)

    if color is None:
        for well, group in df.groupby('welln'):
            ax.plot(group[x], group[y], color='k')
        handles, labels = ax.get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
    else:
        groups = df[color].unique()
        cmap = plt.get_cmap('gist_rainbow')
        # cmap = plt.get_cmap('viridis')
        color_map = dict(zip(groups, cmap((groups - groups.min()) / (groups.max()-groups.min()))))

        for (cc, well), group in df.groupby([color, 'welln']):
            ax.plot(group[x], group[y], color=color_map[cc]*.75, label=cc)
        handles, labels = ax.get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), title=color)

def plot3d(dd, color=None, factor=None, cmap=None, hover=None, projection='UMAP'):
    import plotly
    import plotly.graph_objs as go
    if color is None:
        raise Exception('Column name for color must be specified.')
    
    if projection == 'PCA':
        projection = 'PC'
    xc, yc, zc = (projection + str(x) for x in range(1, 4))

    trace_params = {'mode': 'markers', 'hoverinfo':'name+text'}
    marker = {'colorscale': 'Jet', 'opacity': 1, 'size': 3}
    layout = {'height': 600, 'margin': {'b': 0, 'l': 0, 'r': 0, 't': 0}, 
              'paper_bgcolor': '#f0f0f0', 'width': 800,
              'scene': {'xaxis':{'title':xc}, 'yaxis':{'title':yc}, 'zaxis':{'title':zc}}}

    traces = []


    if factor == False or dd[color].dtype == 'float64':
        if dd[color].dtype.name == 'category':
            marker['color'] = dd[color].cat.codes.values
        else:
            marker['color'] = dd[color].values
        marker['colorbar'] = dict(title=color, thickness=10, len=.3, y=.8)
        marker['showscale'] = True
        trace_params['marker'] = marker
        trace = go.Scatter3d(x=dd[xc], y=dd[yc], z=dd[zc], hovertext=dd['compound'], **trace_params)
        traces.append(trace)
        layout['showlegend'] = False
    else:
        n_colors = len(dd[color].unique())
        if cmap:
            pass
        elif n_colors <= 10:
            cmap = 'tab10'
        else:
            cmap = 'tab20'

        cm = plt.get_cmap(cmap)
        def get_plotly_color(cm, n):
            return 'rgb' + str(cm(n, bytes=True)[:3])
        for ii, (name, sg) in enumerate(dd.groupby(color)):
            marker['color'] = get_plotly_color(cm, ii)
            trace_params['marker'] = marker
            trace_params['name'] = name
            trace = go.Scatter3d(x=sg[xc], y=sg[yc], z=sg[zc], hovertext=sg[hover], **trace_params)
            traces.append(trace)
            layout['showlegend'] = True

    fig = go.Figure(data=traces, layout=go.Layout(layout))
#     if publish:
#         plotly.iplot(fig)
#     else:
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

def pca_wide(dfw, n=3, plot=True):
    X = extract_data(dfw)
    pca_m = PCA(n_components=n)
    X_pca = pca_m.fit_transform(X)
    columns = ['PC' + str(x) for x in range(1,n+1)]
    X_pca_df = pd.DataFrame(X_pca, index=dfw.index, columns=range(n))#.reset_index()

    if plot:
        plot_explained_variance(pca_m)
    # print('residual = %.3f' % (1-pca_m.explained_variance_ratio_.sum()))
    return pca_m, X_pca_df#.reset_index()

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
    dfl = dfl.sort_values(['welln', t]).reset_index(drop=True)
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
    if x in to_drop:
        to_drop.remove(x)
    if t == 'time':
        to_drop.remove(t)
    dfw = dfw.drop(to_drop, axis=1)
    dfw.set_index(
        [cc for cc in dfw.columns if (cc != x) & (cc != t)] + [t], inplace=True)
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

def extract_data(df):
    ints = [col for col in df.columns if type(col) == int]
    if len(ints) > 2:
        selection = df[ints]
    elif df.columns.str.contains('PC').sum() > 3:
        selection = df.loc[:, df.columns.str.contains('PC')]
    elif df.columns.str.contains('TDRC').sum() > 3:
        selection = df.loc[:, df.columns.str.contains('TDRC')]
    else:
        raise ValueError('Neither integers, PC or TDRC found in columns.')
    X = selection.values
    return X

def add_tsne(df, dims=2, perplexity=30):
    from sklearn.manifold import TSNE
    X = extract_data(df)
    X_embedded = TSNE(n_components=dims, perplexity=perplexity).fit_transform(X)
    for n in range(dims):
        label = 'tSNE%s' % (n+1)
        df[label] = X_embedded[:, n]
    return df

def add_umap(df, dims=3, **kwargs):
    import umap
    X = extract_data(df)
    embedding = umap.UMAP(n_components=dims, **kwargs).fit_transform(X)
    for n in range(dims):
        df['UMAP' + str(n+1)] = embedding[:, n]
    return df


def calculate_z_score(p, components=None):
    res = np.zeros_like(p.PC1)
    for pc in p.columns[p.columns.str.contains('PC')][:components]:
        res += ((p[pc] - p[pc].mean())/ p[pc].std())**2
    return np.sqrt(res)

def add_znorm(df):
    znorm = np.zeros(df.shape[0])
    for pc in [col for col in df.columns if 'PC' in col]:
        znorm += (df[pc] / df[pc].std())**2
    df['znorm'] = np.sqrt(znorm)

tts = np.cumsum(np.concatenate([[0], 1.05**np.arange(43) * .5]))

def resample_plate(plate, tts=tts, column='lnci'):
    well = plate.query('welln == 0')
    source_t = well['time'].values
    ii = np.searchsorted(source_t, tts)
    t1, t2 = source_t[ii-1], source_t[ii]

    index = [c for c in
        ['cl', 'cp', 'ap', 'welln', 'library', 'compound', 'sid', 'log_c', 'tp']
        if c in plate.columns]
    plate2 = plate.set_index(index)[column].unstack()
    tps = well['tp'].iloc[ii]
    c1 = plate2[tps - 1].values
    c2 = plate2[tps].values
    res = c1 + (c2 - c1) * ((tts - t1)/(t2 - t1))
    columns = pd.MultiIndex.from_product([[column], tts], names=[None, 'time'])
    return pd.DataFrame(res, index=plate2.index, columns=columns).stack().reset_index()

def well2welln_384(wells, form=384):
    form = int(form)
    if form not in [96, 384, 1536]:
        raise ValueError('Only formats 96, 384 and 1536 supported.')
    n_cols = int(sqrt(form/2*3))
    wells = wells if type(wells) == np.ndarray else np.array(wells, dtype=np.str)
    _well_regex = re.compile('^([A-Z]{1,2})(\d{1,2})')
    def _w2wn(well, n_cols):
        match = _well_regex.match(well)
        if not match:
            raise ValueError('Well not recognized: "%s"' % well)
        rr, cc = match.group(1), match.group(2)
        rrn = ord(rr) - 65 if len(rr) == 1 else ord(rr[1]) - 39
        ccn = int(cc) - 1
        return rrn * n_cols + ccn
    _vw2wn = np.vectorize(_w2wn, excluded=('n_cols'))
    wns = _vw2wn(wells, n_cols)
    if np.any(wns >= form) or np.any(wns < 0):
        raise ValueError('welln out of range')
    return wns

def hierarchical_clustering(df, k=10, plot=False, figsize=(8, 8)):
    X = extract_data(df)
    Z = scipy.cluster.hierarchy.linkage(X, 'ward')
    clusters = scipy.cluster.hierarchy.fcluster(Z, t=k, criterion='maxclust')
    max_d = Z[-k+1, 2]
    if plot:
        fig, ax = plt.subplots(figsize=figsize)
        ax.axvline(x=max_d, c='silver')
        R = scipy.cluster.hierarchy.dendrogram(Z,
                       labels=df['compound'].values,
                       color_threshold=max_d,
                       leaf_font_size=8,
                       above_threshold_color='gray',
                       orientation='left',
                       ax=ax)
        if 'moa' in df.columns:
            cm = plt.cm.get_cmap("tab10")
            for lbl in ax.get_ymajorticklabels():
                iloc = np.where(df['compound'] == lbl._text)[0][0]
                moa = df['moa'].cat.codes.iloc[iloc]
                if moa > -1:
                    lbl.set_color(cm(int(moa)))
                else:
                    lbl.set_color('silver')
            proxy_artists, labels = [], []
            for moa in df['moa'].cat.categories:
                proxy_artists.append(mpl.lines.Line2D([0], [0]))
                labels.append(moa)
            legend = ax.legend(proxy_artists, labels, handletextpad=0, handlelength=0)
            for n, text in enumerate(legend.texts):
                text.set_color(cm(n))


        [ax.spines[key].set_visible(False) for key in ['left', 'right', 'top', 'bottom']]
        ax.invert_yaxis()
        ax.set_xlabel('euclidean distance')
        ax.set_title('k=' + str(k))
        #leaves = pd.DataFrame({'compound':R['ivl'], 'leaf':np.arange(len(R['ivl']))})
        #df = df.merge(leaves)    

    df['cluster'] = pd.Categorical(clusters)
    #return fig, ax

def evaluate_clustering(df, dr_method, k_min=10, k_max=30):
    fn = 'clustering_comparisons.csv'
    
    if os.path.isfile(fn):
        ec = pd.read_csv(fn)
    else:
        ec = pd.DataFrame()
        
        
    X = extract_data(df)
    Z = scipy.cluster.hierarchy.linkage(X, 'ward')
    loc = df['moa'] != 'other'
    true = df.loc[loc, 'moa']
    ll = []
    for k in range(k_min, k_max+1):
        predicted = scipy.cluster.hierarchy.fcluster(Z, t=k, criterion='maxclust')[loc]
        ll.append([dr_method, k, 'ARI', metrics.adjusted_rand_score(true, predicted)])
        ll.append([dr_method, k, 'AMI', metrics.adjusted_mutual_info_score(true, predicted)])
    new = pd.DataFrame(ll, columns=['dr', 'k', 'index', 'value'])
    ec = (pd.concat([ec, new])
          .drop_duplicates(['dr', 'k', 'index'],keep='last')
          .sort_values(['dr', 'k', 'index']))
    ec.to_csv(fn, index=False) 
    return ec

def plot_comparisons():
    ec = pd.read_csv('clustering_comparisons.csv')
    (ggplot(ec)
     + aes('k', 'value', color='index')
     + geom_line()
     + facet_grid('~dr')
     + theme(figure_size=(8, 2))
    ).draw()
    
# TDRC ####################################################
inflectionShift = 6
slopeFactorHill = 10

cc = np.array([-8.7, -8.4, -8.1, -7.8, -7.5, -7.2, -6.9, -6.6, -6.3, -6. , -5.7, -5.4, -5.1, -4.8, -4.5, -4.2])
cc1 = np.expand_dims(cc, 0)

x = np.linspace(-1,1,44)
T = np.zeros([44,10])
for n, r in enumerate(np.identity(10)):
    T[:,n] = -np.polynomial.chebyshev.chebval(-x, r)
    
def f_tdrc_hill(p):
    Q = T @ p.reshape([3,10]).T
    max_effect = Q[:,0:1]
    inflection = Q[:,1:2] - inflectionShift
    slope = Q[:,2:3] * slopeFactorHill
    slope = slope * (slope > 0)
    Yhat = max_effect / (1 + (cc1 / inflection)**slope)
    return Yhat

def f_tdrc_logistic(p):
    Q = T @ p.reshape([3,10]).T
    m = Q[:,0:1]
    i = Q[:,1:2] - inflectionShift
    s = Q[:,2:3]
    s = s * (s > 0)
    Yhat = m / (1 + np.exp(-s*(cc1 - i)))
    return Yhat

def costf_residuals(Y, f_tdrc):
    return lambda p: (f_tdrc(p) - Y).flatten()

def costf_regularized(Y, f_tdrc):
    return lambda p: np.concatenate([(f_tdrc(p) - Y).flatten(), p])

def costf_potency_invariant(Y, f_tdrc):
    return lambda p: np.concatenate([(f_tdrc(p) - Y).flatten(), p[:10], p[11:]])

def fit_tdrc(row, cf=costf_potency_invariant, f_tdrc=f_tdrc_hill, verbose=False, **kwargs):
    Y = row.values.reshape([16,44]).T
    costf = cf(Y, f_tdrc)
    t0 = time()
    res = least_squares(costf, np.zeros(30), jac='2-point', **kwargs) # method='lm',)
    res['time'] = time() - t0
    res['success'] = 'success' if res['success'] else 'fail'
    if verbose:
        print('{time:.2f}s\t{success}\t cost={cost:.2f}\t nfev={nfev}'.format(**res))
    return pd.Series(res.x)

def plot_tdrc(Y, ax=None):
    jet = plt.get_cmap('jet')
    if ax is None: ax = plt.gca()
    l = Y.shape[0]
    for i in range(l):
        ax.plot(cc, Y[i,:], color=jet(i/l), linewidth=1)

def dg_tdrc(row, lim=False, return_p=False, f_tdrc=f_tdrc_hill, **kwargs):
    Y = row.values.reshape([16,44]).T
    p = fit_tdrc(row, f_tdrc=f_tdrc, **kwargs)
    
    
    fig = plt.figure(tight_layout=True, figsize=(10, 5))
    ax1 = plt.subplot(231)
    plot_tdrc(Y, ax1)
    ax1.set_title('{1}, {2}'.format(*row.name))

    ax2 = plt.subplot(232, sharey=ax1)
    plot_tdrc(f_tdrc(p.values), ax2)
    
    ax3 = plt.subplot(233)
    q = p.values.reshape([3,10]).T
    ax3.axhline(color='gray')
    ax3.plot(q[:,0], '.-', label='max_effect')
    ax3.plot(q[:,1], '.-', label='inflection')
    ax3.plot(q[:,2] , '.-', label='slope')
    ax3.legend()
    
    ax4 = plt.subplot(234, sharey=ax1)    
    Q = T @ q
    ax4.plot(Q[:,0:1], '.-', color='C0')
    ax4.set_title('max_effect')
    
    ax5 = plt.subplot(235)
    ax5.plot(Q[:,1:2] - inflectionShift, '.-', color='C1')
    ax5.set_title('inflection')

    ax6 = plt.subplot(236)
    slope = Q[:,2:3] #* slopeFactor
    slope = slope * (slope > 0)
    ax6.plot(slope, '.-', color='C2')
    ax6.set_title('slope')
    
    if lim:
        ax1.set_ylim((-4, 1))
        ax3.set_ylim((-1, 1))
        ax5.set_ylim((-9, -4))
        ax6.set_ylim((0, 40))
        
    return p.values if return_p else None

def compare_tdrc_group(raw, tdrc, compounds, drc, title=None, concentration_independent=True, **kwargs):
    fig0, ax0 = plt.subplots()
    plt.scatter('tSNE1', 'tSNE2', data=compounds)
    for r, row in enumerate(compounds.iloc):
        plt.annotate(row['compound'], (row['tSNE1'], row['tSNE2']), color='C' + str(r))
    if title:
        plt.title(title)
    
#     return None
    
    if drc == 'Hill':
        f_tdrc = f_tdrc_hill
    elif drc == 'logistic':
        f_tdrc = f_tdrc_logistic
        
    else:
        raise ValueError('Parameter `drc` must be specified as either `Hill` or `logistic`.')
    
    cls = raw.index.unique(level='cl')
    fig1, ax1 = plt.subplots(len(compounds)+1, 
                           2*len(cls)+1, 
                           sharex=True, sharey=True, 
                           tight_layout=True, 
                           figsize=(18, len(compounds) + 1))
    ax1[0, -1].axis('off')
    
    fig2, ax2 = plt.subplots(6, 6, sharey='row', tight_layout=True, figsize=(18, 18))
    X = np.arange(30).reshape([3,10])
   
    for r, compound in enumerate(compounds['compound']):
        ax = ax1[r+1, -1]
        ax.text(0, 0.5, compound, 
                       verticalalignment='center', horizontalalignment='left',
                       transform=ax.transAxes, fontsize=15, color='C' + str(r))
        ax.axis('off')
        for c, cl in enumerate(cls):
            if r == 0:
                ax1[0, c*2].text(0, 0.5, cl, 
                               verticalalignment='center', horizontalalignment='left',
                               transform=ax1[0, c*2].transAxes, fontsize=15)
                ax1[0, c*2].axis('off')
                ax1[0, c*2+1].axis('off')
                
            row = raw.query('cl == @cl & compound == @compound').iloc[0]
            Y = row.values.reshape([16,44]).T
            plot_tdrc(Y, ax1[r+1, c * 2])

#             if fit:
#                 p = fit_tdrc(row, **kwargs).values
#             else:
            p = tdrc.query('cl == @cl & compound == @compound').iloc[:,3:].copy().values.flatten()
            if concentration_independent:
                p[10] = 0
            plot_tdrc(f_tdrc(p), ax1[r+1, c * 2 + 1])

            q = p.copy().reshape([3,10]).T
#             q[0,1] = 0
            Q = T @ q
        
            ax2[0, c].set_title(cl)
            ax2[0, c].plot(Q[:,0:1], '.-')
            ax2[1, c].axhline(color='gray')
            ax2[1, c].plot(X[0], q[:,0], '.-')
            ax2[1, c].set_ylim(-3, 3)
            
            ax2[2, c].plot(Q[:,1:2] - inflectionShift, '.-')
            ax2[3, c].axhline(color='gray')
            ax2[3, c].plot(X[1], q[:,1], '.-')
            ax2[3, c].set_ylim(-3, 3)
            
            slope = Q[:,2:3]
            slope = slope * (slope > 0)
            ax2[4, c].plot(slope, '.-')
            ax2[5, c].axhline(color='gray')
            ax2[5, c].plot(X[2], q[:,2] , '.-')
            ax2[5, c].set_ylim(-3, 3)
    ax2[0, 0].set(ylabel='max. effect')
    ax2[2, 0].set(ylabel='inflection')
    ax2[4, 0].set(ylabel='slope')
    return None

if __name__ == '__main__':
    pass
