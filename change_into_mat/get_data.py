
# %%
import numpy as np
import scipy.io
import mne

from get_folders import table as folder_table

from pathlib import Path
from toolbox import _match_montage, _append_SSP
from onstart import CONFIG, logger
from tqdm.auto import tqdm

# %%
folder_table

# %%
save_to_folder = Path.cwd().joinpath('../data')

# %%


def _read_folder(folder):
    raw = mne.io.read_raw_ctf(folder)
    logger.debug('Read raw ctf from {}'.format(folder))
    return raw


# %%
for meg_folder in tqdm(folder_table.query('mode=="MEG"')['path'], 'Read MEG Folder'):
    name = Path(meg_folder).name
    print(name)

    # -- %%
    # Read MEG Obj
    meg = _read_folder(meg_folder)

    # -- %%
    # Get Events
    events, events_id = mne.events_from_annotations(meg)

    # -- %%
    # Parse MEG Epochs
    meg_epochs = mne.Epochs(meg, events, tmin=-2.0, tmax=8.0, picks='mag')
    meg_data = meg_epochs.get_data()

    # -- %%
    # Parse EEG Epochs
    eeg_epochs = mne.Epochs(meg, events, tmin=-2.0, tmax=8.0, picks='eeg')
    eeg_epochs = _match_montage(eeg_epochs)
    eeg_data = eeg_epochs.get_data()

    print(meg_data.shape, eeg_data.shape)

    file_path = save_to_folder.joinpath('{}.mat'.format(name))
    mat = dict(
        eeg_data=eeg_data,
        meg_data=meg_data,
        events=events
    )
    scipy.io.savemat(file_path, mat)

# %%

