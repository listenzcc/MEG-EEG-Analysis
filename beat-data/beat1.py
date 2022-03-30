# %%
import scipy.io
import numpy as np

import plotly.express as px

from pathlib import Path

# %%
raw_data1 = scipy.io.loadmat(Path.cwd().joinpath(
    '../data/S01_G33IA_20220119_01.ds.mat'))

raw_data2 = scipy.io.loadmat(Path.cwd().joinpath(
    '../data/S01_G33IA_20220119_02.ds.mat'))

# %%
raw_data1['eeg_data'].shape, raw_data1['events'].shape

# %%
np.unique(raw_data1['events'][:, -1])

# %%
averages = dict()
for e in np.unique(raw_data1['events'][:, -1]):
    averages[e] = np.mean(raw_data1['eeg_data']
                          [raw_data1['events'][:, -1] == e], axis=0)
    print(e, averages[e].shape)

# %%
pred_coefs = []
for e in np.unique(raw_data2['events'][:, -1]):
    dd = np.mean(raw_data2['meg_data']
                 [raw_data2['events'][:, -1] != e], axis=0)
    a = np.corrcoef(dd, averages[e])
    pred_coefs.append((e, a))

pred_coefs[0][1].shape

# %%
for e, coef in pred_coefs:
    _coef = coef.copy()
    np.fill_diagonal(_coef, 0)
    fig = px.imshow(_coef, title='Event: {} -> Others'.format(e), width=600)
    fig.show()

# %%
