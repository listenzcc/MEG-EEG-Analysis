# %%
import mne
from pathlib import Path
from joblib import dump

from tqdm.auto import tqdm
from onstart import LOGGER
from subjects_on_disk import mk_subject_table
from known_folders import known_folders

# %%


def _mk_epochs_kwargs(raw, picks='mag', target_sfreq=200):
    sfreq = raw.info['sfreq']
    assert sfreq % target_sfreq == 0, 'Invalid sfreq {}'.format(sfreq)
    decim = int(sfreq / target_sfreq)

    return dict(
        decim=decim,
        tmin=-1.0,
        tmax=4.0,
        picks=picks,
        detrend=1,
        preload=True,
    )


filter_kwargs = dict(
    l_freq=7.0,
    h_freq=35.0
)

# %%
folder = known_folders[0]

# %%
allow_override = True

for folder in tqdm(known_folders, 'Reading Subjects'):
    subject_name = Path(folder).name
    subject_name

    subject_table = mk_subject_table(folder, select_mode='MEG')
    subject_table

    for raw_idx in tqdm(subject_table.index, 'Reading raw ctf'):
        output_name = 'data/{}.{}.dump'.format(subject_name, raw_idx)

        if not allow_override:
            if Path(output_name).is_file():
                LOGGER.warning(
                    'Ignore {} since it exists'.format(output_name))
                continue

        raw = mne.io.read_raw_ctf(
            subject_table.loc[raw_idx, 'path'], preload=False)

        events, events_id = mne.events_from_annotations(raw)
        epochs = mne.Epochs(
            raw, events, **_mk_epochs_kwargs(raw, picks=['mag', 'eeg']))
        epochs.filter(**filter_kwargs)

        LOGGER.debug('Read data from {}'.format(epochs))

        data = dict(
            data=epochs.get_data(),
            times=epochs.times,
            ch_names=epochs.ch_names,
            info=epochs.info,
            events=epochs.events,
            events_id=events_id,
        )

        epochs.save(output_name + '.epoch')

        dump(data, output_name)
        LOGGER.debug('Saved data into {}'.format(output_name))
    print('Done')

print('All done')

# %%
