# %%
import mne
import numpy as np
import plotly.express as px

from tqdm.auto import tqdm
from onstart import LOGGER
from subjects_on_disk import mk_file_table
from known_folders import known_folders
from toolbox import Drawer, _match_eeg_montage


# %%
folder = known_folders[0]
subject_table = mk_file_table(folder, select_mode='MEG')
subject_table

# %%
raw_idx = 0
raw = mne.io.read_raw_ctf(subject_table.loc[raw_idx, 'path'], preload=False)


def _mk_epochs_kwargs(raw, picks='eeg'):
    sfreq = raw.info['sfreq']
    assert sfreq % 200 == 0, 'Invalid sfreq {}'.format(sfreq)
    decim = int(sfreq / 200)

    return dict(
        decim=decim,
        tmin=-2.0,
        tmax=8.0,
        picks=picks,
        detrend=1,
        preload=True,
    )


filter_kwargs = dict(
    l_freq=1.0,
    h_freq=40.0
)

events, events_id = mne.events_from_annotations(raw)
epochs = mne.Epochs(raw, events, **_mk_epochs_kwargs(raw))
epochs.filter(**filter_kwargs)

# If picks is 'eeg'
epochs = _match_eeg_montage(epochs)

times = epochs.times

print(epochs.get_data().shape, epochs.info)
epochs

# %%
evokeds = dict()
for event in tqdm(events_id, 'Average epochs'):
    evoked = epochs[event].average()
    evokeds[event] = evoked

evokeds

# %%
freqs = np.linspace(1, 40, 40)

morlet_kwargs = dict(
    freqs=freqs,
    n_cycles=5,
    return_itc=False
)

tfs = dict()
for event in tqdm(events_id, 'Compute time-frequency'):
    tfs[event] = mne.time_frequency.tfr_morlet(evokeds[event], **morlet_kwargs)

# Data shape is (273 x 40 x 2001), (channels x freqs x times)
print(tfs['1'].data.shape)
tfs

# %%
times.shape, freqs.shape

# %%
freq_bands = dict(
    delta=(0, 3),
    theta=(3.5, 7.5),
    alpha=(7.5, 13),
    beta=(14, 1e4),
    custom=(8.5, 10.5)
)

info = epochs.info
evoked = evokeds['1'].copy()

drawer = Drawer()
for event in tqdm(events_id, 'Display on plot_joint'):
    for band in freq_bands:
        a, b = freq_bands[band]
        freq_select = [e > a and e < b for e in freqs]
        evoked.data = np.mean(tfs[event].data[:, freq_select], axis=1)
        drawer.fig = evoked.plot_joint(
            title='Joint plot {}, {}'.format(event, band))
drawer.save('time-frequency-eeg.pdf')

print('Done')

# %%
