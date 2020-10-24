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
# from IPython.core.debugger import set_trace
from plotnine import *
from time import time
from scipy.optimize import least_squares
from shutil import which
import fastcluster

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
    df = df.groupby(['ap', 'welln'], as_index=False).apply(lambda x: normalize_well(x, fig))
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

_rows = np.array([chr(x) for x in range(65, 91)] + ['A' + chr(x) for x in range(65, 71)])

def welln2well_384(wells, form=384):
    form = int(form)
    if form not in [96, 384, 1536]:
        raise ValueError('Only formats 96, 384 and 1536 supported.')
    n_cols = int(sqrt(form/2*3))

    wells = wells if type(wells) == np.ndarray else np.array(wells, dtype=np.int)
    if np.any(wells >= form) or np.any(wells < 0):
        raise ValueError('welln out of range')
    rr = _rows[wells // n_cols]
    cc = (wells % n_cols + 1).astype(str)
    return np.core.defchararray.add(rr, cc)

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
    plotly.offline.iplot(fig)

# helper functions ###########################################################################

def extract_data(df):
    ints = [col for col in df.columns if type(col) == int]
    if len(ints) > 2:
        selection = df[ints]
    elif df.columns.str.contains('PC').sum() > 3:
        selection = df.loc[:, df.columns.str.contains('PC')]
    elif df.columns.str.contains('UMAP').sum() > 1:
        selection = df.loc[:, df.columns.str.contains('UMAP')]
    else:
        raise ValueError('Neither integers, PC or UMAP found in columns.')
    X = selection.values
    return X

def add_tsne(df, dims=2, perplexity=30, seed=None):
    from sklearn.manifold import TSNE
    X = extract_data(df)
    X_embedded = TSNE(n_components=dims, perplexity=perplexity, random_state=seed).fit_transform(X)
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


# Clustering ######################################################################################

def replace_with_rainbow_text(lbl, strings, colors, ax=None, **kwargs):
    ax = ax if ax else plt.gca()
    t = lbl.get_transform()
    x, y = lbl.get_position()
    renderer = ax.figure.canvas.get_renderer()
    ll = []
    for s, c in zip(strings, colors):
        text = ax.text(x, y, s , color=c, transform=t, **kwargs)
        text.draw(renderer)
        ex = text.get_window_extent()
        t = mpl.transforms.offset_copy(text.get_transform(), x=ex.width, units='dots')
        ll.append(text)
    lbl.set_visible(False)
    return ll

def hierarchical_clustering(df, k=10, metric='euclidean', truncate=False, add_clusters=False, cm=None):
    X = extract_data(df)
    Z = fastcluster.linkage(X, 'ward', metric=metric)
    max_d = Z[-k+1, 2]
    truncate_mode = 'lastp' if truncate else None
    labels = df['compound'].values
    groups = df['moa'] if 'moa' in df.columns else None
    if truncate and groups is not None:
        figsize = (8, 1 + 0.184 * len(groups.unique()))
    else:
        figsize = (8, 1 + 0.27 * len(labels))
    
    fig, ax = plt.subplots(figsize=figsize)
    R = scipy.cluster.hierarchy.dendrogram(
        Z,
        labels=labels,
        truncate_mode=truncate_mode,
        p=k,
        color_threshold=max_d,
        leaf_font_size=8,
        above_threshold_color='gray',
        orientation='left',
        ax=ax,
        #show_contracted=show_contracted,
    )
    [ax.spines[key].set_visible(False) for key in ['left', 'right', 'top', 'bottom']]
    ax.invert_yaxis()
    ax.set_xlabel(metric + ' distance')
    ax.set_title('k=' + str(k))

    if groups is not None:
        gn = groups.unique().shape[0]
        if cm is not None:
            pass
        elif gn <= 10:
            cm = plt.cm.get_cmap('tab10')
        elif gn <= 20:
            cm = plt.cm.get_cmap('tab20')
        else:
            cm = plt.cm.get_cmap('tab20')
        if truncate:
            lg = len(groups)
            G = np.zeros([2 * lg - 1, len(groups.cat.categories)])
            for ii, code in enumerate(groups.cat.codes):
                G[ii, code] = 1
            for ii, row in enumerate(Z):
                G[lg + ii] = G[int(row[0])] + G[int(row[1])]
            fig.canvas.draw() # recompute autoscaled limits
            for ii, lbl in enumerate(ax.get_ymajorticklabels()):
                leave = R['leaves'][ii]
                gg = G[leave]
                tt, cc = [], []
                for jj, x in enumerate(gg):
                    if not (x == 0.0 or jj == gn-1):
                        tt.append('\u2b24' * int(x))
                        cc.append(cm(jj))
                jj, x = gn-1, gg[gn-1]
                if 0 < x <= 3:
                    tt.append('\u2b24' * int(x))
                    cc.append(cm(jj))
                elif 2 < x:
                    tt.append(str(int(x)) + '\u00d7' + '\u2b24')
                    cc.append(cm(jj))
                replace_with_rainbow_text(lbl, tt, cc, ax=ax, size=9, va='center_baseline', ha='left')    
        else:
            for lbl in ax.get_ymajorticklabels():
                iloc = np.where(labels == lbl._text)[0][0]
                moa = groups.cat.codes.iloc[iloc]
                if moa < gn-1:
                    #lbl.set_backgroundcolor(cm(int(moa)))
                    lbl.set_bbox(dict(facecolor=cm(int(moa)), edgecolor='none', boxstyle='square,pad=0.1'))
                else:
                    pass
                    #lbl.set_color('silver')
            ax.axvline(x=max_d, c='silver')
        proxy_artists, labs = [], []
        for moa in groups.cat.categories:
            proxy_artists.append(mpl.lines.Line2D([0], [0]))
            labs.append('\u2b24 ' + moa if truncate else moa)
        legend = fig.legend(proxy_artists, labs, handletextpad=0, handlelength=0, loc='upper left')
        for n, text in enumerate(legend.texts):
            text.set_bbox(dict(facecolor=cm(n), edgecolor='none', boxstyle='square,pad=0.1'))
            #text.set_color(cm(n))
    if add_clusters:
        clusters = scipy.cluster.hierarchy.fcluster(Z, t=k, criterion='maxclust')
        df['cluster'] = pd.Categorical(clusters)
    
fn = 'clustering_comparisons.csv'
def evaluate_clustering(df, dr_method, k_min=10, k_max=30, metric='euclidean'):
    if os.path.isfile(fn):
        ec = pd.read_csv(fn)
    else:
        ec = pd.DataFrame()
    X = extract_data(df)
    Z = scipy.cluster.hierarchy.linkage(X, 'ward', metric=metric)
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

def remove_evaluation(method):
    ec = pd.read_csv(fn)
    ec = ec.query('dr != @method')
    ec.to_csv(fn, index=False)
    
def plot_comparisons():
    ec = pd.read_csv(fn)
    (ggplot(ec)
     + aes('k', 'value', color='index')
     + geom_line()
     + facet_grid('~dr')
     + theme(figure_size=(8, 2))
    ).draw()
    
def plot_comparisons2():
    ec = pd.read_csv(fn)
    labels = {}
    fig, ax = plt.subplots(1, 2, figsize=(8, 2), sharex=True, sharey=False)
    for ii, index in enumerate(['AMI', 'ARI']):
        df = ec.query('index == @index').set_index(['dr', 'k', 'index']).unstack('dr')
        dr = df.columns.get_level_values('dr')
        k = df.index.get_level_values('k').values
        X = df.values.T
        for jj, method in enumerate(dr):
            if method.startswith('PCA'):
                kwa = dict(label='PCA', color='red', lw=3, zorder=2)
            elif method.startswith('CTRS'):
                kwa = dict(label='CTRS', color='green', lw=3, zorder=3)
            elif method.startswith('UMAP'):
                kwa = dict(label='UMAP', color='silver', lw=3, zorder=1)

            labels[kwa['label']] = ax[ii].plot(k, X[jj], alpha=.5, **kwa)[0]
        ax[ii].set_title(index)
        ax[ii].set_xlabel('# clusters')
        [ax[ii].spines[key].set_visible(False) for key in ['left', 'right', 'top', 'bottom']]
    fig.legend(labels.values(), labels.keys(), loc=7) #ncol=3
    # fig.tight_layout()
    fig.subplots_adjust(right=0.85)
#     fig.set_facecolor('pink')

def plot_comparisons3():
    ec = pd.read_csv(fn)
    ec['method'] = ec['dr'].str.extract(r'(\D+)')
    ec['method'] = pd.Categorical(ec['method'], categories=['PCA', 'UMAP', 'CTRS'], ordered=True)
    ec['dr'] = pd.Categorical(ec['dr'], ordered=True)
#     ec['dr'].cat.categories = ec['dr'].cat.categories[::-1]
    (ggplot(ec)
     + aes('k', 'value', color='method', group='dr')
     + geom_line(alpha=0.5, size=1)
     + facet_grid('~index')
     + theme(figure_size=(6, 2))
     + labs(x='number of clusters', y=None)
     + scale_color_manual(['red', 'silver', 'green'])
    ).draw()

def plot_pca_explained_variance(model):
    evr = model.explained_variance_ratio_
    residual = 1 - evr.sum()
    n = model.n_components
    index = np.arange(1, n+2)
    color = ['g' for i in range(n)] + ['r']
    variance = np.concatenate([evr, np.array([residual])])
    fig, ax = plt.subplots(figsize=(8,1))
    ax.barh(index, variance, color=color)
    ax.set_yticks(range(1, n, n//3))
    ax.invert_yaxis()
    ax.set_xlim(0,1)
    ax.set_xlabel('explained variance')
    ax.set_ylabel('components')
    ax.text(.99, 0.1, 'residual = %.3f' % residual, color='r', transform=ax.transAxes, ha='right')
    

# CTRS ####################################################

inflectionShift = 6

cc = np.array([-8.7, -8.4, -8.1, -7.8, -7.5, -7.2, -6.9, -6.6, -6.3, -6. , -5.7, -5.4, -5.1, -4.8, -4.5, -4.2])
cc1 = np.expand_dims(cc, 0)

x = np.linspace(-1,1,44)
T = np.zeros([44,10])
for n, r in enumerate(np.identity(10)):
    T[:,n] = -np.polynomial.chebyshev.chebval(-x, r)

def f_ctrs(p):
    Q = T @ p.reshape([3,10]).T
    m = Q[:,0:1]
    i = Q[:,1:2] - inflectionShift
    s = Q[:,2:3]
    s = s * (s > 0)
    Yhat = m / (1 + np.exp(-s*(cc1 - i)))
    return Yhat

class CTRS():
    """CTRS object"""
    def __init__(self, cost='potency_invariant', **kwargs):
        self.cost = cost
        if cost == 'residuals':
            self.costf = self.costf_residuals
        elif cost == 'regularized':
            self.costf = self.costf_regularized
        elif cost == 'potency_invariant':
            self.costf = self.costf_potency_invariant
        self.verbose = kwargs.pop('verbose', False)
        self.kwargs = kwargs

    def costf_residuals(self, Y):
        return lambda p: (f_ctrs(p) - Y).flatten()

    def costf_regularized(self, Y):
        return lambda p: np.concatenate([(f_ctrs(p) - Y).flatten(), p])

    def costf_potency_invariant(self, Y):
        return lambda p: np.concatenate([(f_ctrs(p) - Y).flatten(), p[:10], p[11:]])

    def fit(self, *args, **kwargs):
        return self
    
    def transform(self, X):
        Y  = np.empty([X.shape[0], 30])
        for m, row in enumerate(X):
            Y[m, :] = self.transform_single(row)
        return Y
        
    def transform_single(self, row):
        Y = row.reshape([16,44]).T
        t0 = time()
        res = least_squares(self.costf(Y), np.zeros(30), jac='2-point', **self.kwargs)
        success = 'success' if res['success'] else 'fail'
        if self.verbose:
            print(f'{time() - t0:.2f}s\t{success}\t cost={res.cost:.2f}\t nfev={res.nfev}')
        return res.x
    
    def inverse_transform(self, Y):
        X = np.empty([Y.shape[0], 16*44])
        for m, p in enumerate(Y):
            X[m, :] = self.inverse_transform_single(p).T.flatten()
        return X
    
    def inverse_transform_single(self, p):
        return f_ctrs(p)
    
    def fit_transform(self, X):
        self.transform(X)
        
    def __repr__(self):
        params = {'cost' : self.cost, **self.kwargs}
        if self.verbose:
            params['verbose'] = True
        params_str = ', '.join(['{}={}'.format(key, val) for key, val in params.items()])
        return f'CTRS({params_str})'

def plot_ctrs(Y, ax=None):
    jet = plt.get_cmap('jet')
    if ax is None: ax = plt.gca()
    l = Y.shape[0]
    for i in range(l):
        ax.plot(cc, Y[i,:], color=jet(i/l), linewidth=1)

def dg_ctrs(row, ylim=False, **kwargs):
    model = CTRS(**kwargs)
    p = model.transform_single(row.values)
    Y = row.values.reshape([16,44]).T
    
    fig = plt.figure(tight_layout=True, figsize=(10, 5))
    ax1 = plt.subplot(231)
    plot_ctrs(Y, ax1)
    ax1.set_title('{1}, {2}'.format(*row.name))

    ax2 = plt.subplot(232, sharey=ax1)
    plot_ctrs(f_ctrs(p), ax2)
    
    ax3 = plt.subplot(233)
    q = p.reshape([3,10]).T
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
    
    if ylim:
        ax1.set_ylim((-4, 1))
        ax3.set_ylim((-1, 1))
        ax5.set_ylim((-9, -4))
        ax6.set_ylim((0, 40))
        
    return None

def compare_ctrs_group(original, ctrs, compounds, title=None, concentration_independent=True, **kwargs):
    fig0, ax0 = plt.subplots()
    plt.scatter('tSNE1', 'tSNE2', data=compounds)
    for r, row in enumerate(compounds.iloc):
        plt.annotate(row['compound'], (row['tSNE1'], row['tSNE2']), color='C' + str(r))
    if title:
        plt.title(title)
    
    cls = original.index.unique(level='cl')
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
                
            row = original.query('cl == @cl & compound == @compound').iloc[0]
            Y = row.values.reshape([16,44]).T
            plot_ctrs(Y, ax1[r+1, c * 2])

            p = ctrs.query('cl == @cl & compound == @compound').copy().values.flatten()
            if concentration_independent:
                p[10] = 0
            plot_ctrs(f_ctrs(p), ax1[r+1, c * 2 + 1])

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
