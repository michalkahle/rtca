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
    elif ('cp' in cols or 'plate' in cols or 's_plate' in cols) and ('welln' in cols or 'well' in cols or 's_well' in cols):
        if 'plate' in cols:
            pass
        elif 'cp' in cols:
            df['plate'] = df['cp'].apply(lambda cp: 'CP-0%s-00' % cp)
            to_drop.append('plate')
        elif 's_plate' in cols:
            df['plate'] = df['s_plate']
            to_drop.append('plate')

        if 'well' in cols:
            pass
        elif 'welln' in cols:
            df['well'] = np.array(list('ABCDEFGHIJKLMNOP'))[df['welln']//24] + (df['welln']%24 + 1).astype(str)
            to_drop.append('well')
        elif 's_well' in cols:
            df['well'] = df['s_well']
            to_drop.append('well')
        dq = df[['plate', 'well']]
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
    res_df = pd.concat(ll)
    res_df = df.merge(res_df, how='left').drop(to_drop, axis=1) # , errors='ignore'
    return res_df

def plate_96():
    ll = pd.DataFrame(dict(
        row = np.repeat(np.arange(8),12),
        col = np.tile(np.arange(12),8),
        welln = np.arange(96),
        edge=0))
    ll.edge[ll.row < 1] += 1
    ll.edge[ll.col < 1] += 1
    ll.edge[ll.row > ll.row.max() - 1] += 1
    ll.edge[ll.col > ll.col.max() - 1] += 1
    return ll

def plate_384():
    ll = pd.DataFrame(dict(
        row = np.repeat(np.arange(16),24),
        col = np.tile(np.arange(24),16),
        welln = np.arange(384),
        edge=0))
    for m in [1,2]:
        ll.edge[ll.row < m] += 1
        ll.edge[ll.col < m] += 1
        ll.edge[ll.row > ll.row.max() - m] += 1
        ll.edge[ll.col > ll.col.max() - m] += 1
    return ll

