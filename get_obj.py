# %%
import mne
from get_folders import table as folder_table

from onstart import CONFIG, logger

# %%
folder_table

# %%
meg_folder = folder_table.query('mode=="MEG"').iloc[0]['path']
eeg_folder = folder_table.query('mode=="EEG"').iloc[0]['path']

# %%


def _read_folder(folder):
    raw = mne.io.read_raw_ctf(folder)
    logger.debug('Read raw ctf from {}'.format(folder))
    return raw


eeg = _read_folder(eeg_folder)
meg = _read_folder(meg_folder)

eeg, meg

# %%
events, events_id = mne.events_from_annotations(meg)
fig = mne.viz.plot_events(events, show=True)

# %%
epochs = mne.Epochs(meg, events, tmin=-0.2, tmax=4.0, picks='mag')
epochs

# %%
fig = mne.viz.plot_sensors(epochs.info, show_names=True)
for eid in events_id:
    evoked = epochs[eid].average()
    fig = mne.viz.plot_evoked_joint(
        evoked, title='MEG Joint Plot Event ID:{}'.format(eid))

# %%
epochs = mne.Epochs(meg, events, tmin=-0.2, tmax=4.0, picks='eeg')
epochs

# %%
mon = mne.channels.make_standard_montage('biosemi64')
mon.ch_names = epochs.ch_names
mon

# %%
epochs.set_montage(mon)

# %%
fig = mne.viz.plot_sensors(epochs.info, show_names=True)
for eid in events_id:
    evoked = epochs[eid].average()
    fig = mne.viz.plot_evoked_joint(
        evoked, title='EEG Joint Plot Event ID:{}'.format(eid))

# %%
