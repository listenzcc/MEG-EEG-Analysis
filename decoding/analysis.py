# %%
import mne
import torch
import numpy as np
import pandas as pd

from sklearn import metrics

from braindecode.models import ShallowFBCSPNet
from braindecode.models import to_dense_prediction_model
from braindecode.models import get_output_shape

from braindecode.datasets import create_from_mne_epochs

from braindecode.preprocessing import exponential_moving_standardize
from braindecode.preprocessing import preprocess
from braindecode.preprocessing import Preprocessor
from braindecode.preprocessing import scale

from braindecode.augmentation import AugmentedDataLoader, SignFlip
from braindecode.augmentation import FrequencyShift

from skorch.callbacks import LRScheduler
from skorch.helper import predefined_split

from braindecode import EEGClassifier
from braindecode.training import CroppedLoss

from pathlib import Path
from tqdm.auto import tqdm

from subjects_on_disk import mk_subject_table
from known_folders import known_folders


# %%
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

# %%
input_window_samples = 500
n_chans = 35
n_classes = 5

model = ShallowFBCSPNet(
    n_chans,
    n_classes,
    input_window_samples=input_window_samples,
    final_conv_length=10
)

model2 = ShallowFBCSPNet(
    n_chans,
    n_classes,
)

model, model2

# %%

to_dense_prediction_model(model)

n_preds_per_input = get_output_shape(model, n_chans, input_window_samples)[2]
n_preds_per_input

# %%
subject_name_table = dict(
    S01='meg-eeg-20220119',
    S02='MEG20220315',
    S03='MEG-EEG20220322',
    S04='MEG20220329'
)

# %%
subject = 'S01'

filenames = [e for e in Path('data').iterdir()
             if e.name.endswith('.dump.epoch')
             and e.name.startswith(subject_name_table[subject])
             and e.name.split('.')[-2] != '10']

filenames = sorted(filenames)[2:]
filenames

# %%


def get_epochs_list(filenames):
    mag_epochs_list = []
    for e in filenames:
        epochs = mne.read_epochs(e).pick_types(meg=True)
        epochs.events[:, -1] %= 5
        mag_epochs_list.append(epochs)

    eeg_epochs_list = []
    for e in filenames:
        epochs = mne.read_epochs(e).pick_types(eeg=True)
        drop_ch_names = epochs.ch_names[35:]
        epochs.drop_channels(drop_ch_names)
        epochs.events[:, -1] %= 5
        eeg_epochs_list.append(epochs)

    print(len(mag_epochs_list), mag_epochs_list[0].get_data().shape)
    print(len(eeg_epochs_list), eeg_epochs_list[0].get_data().shape)

    return mag_epochs_list, eeg_epochs_list


# %%
# The shape of mag_epochs_list[0] is like
# 40 x 273 x 1001

# The shape of eeg_epochs_list[0] is like
# 40 x 35 x 1001

mag_epochs_list, eeg_epochs_list = get_epochs_list(filenames)

# %%
kwargs = dict(
    window_size_samples=input_window_samples,
    window_stride_samples=n_preds_per_input,
    drop_last_window=False


)
mag_dataset = create_from_mne_epochs(mag_epochs_list, **kwargs)
eeg_dataset = create_from_mne_epochs(eeg_epochs_list, **kwargs)

# %%
group = []
event = []
for grp, epochs in enumerate(eeg_epochs_list):
    for evt in epochs.events:
        group.append(grp)
        event.append(evt[-1])

description = dict(
    event=event,
    group=group
)

mag_dataset.set_description(description, overwrite=True)
eeg_dataset.set_description(description, overwrite=True)
mag_dataset.description, eeg_dataset.description

# %%
factor_new = 1e-3

init_block_size = 200

mag_preprocessors = [
    Preprocessor(scale, factor=1e14, apply_on_array=True),
    # Preprocessor(exponential_moving_standardize,
    #              factor_new=factor_new, init_block_size=init_block_size)
]

eeg_preprocessors = [
    Preprocessor(scale, factor=1e6, apply_on_array=True),
    # Preprocessor(exponential_moving_standardize,
    #              factor_new=factor_new, init_block_size=init_block_size)
]


preprocess(mag_dataset, mag_preprocessors)
preprocess(eeg_dataset, eeg_preprocessors)

# %%
valid_group = 0


def _map(e):
    if e == valid_group:
        return 'valid'
    return 'train'


dct = dict(
    split=eeg_dataset.description['group'].map(_map)
)

eeg_dataset.set_description(dct, overwrite=True)

splitted = eeg_dataset.split('split')

train_set = splitted['train']
valid_set = splitted['valid']

len(train_set), len(valid_set)

# %%
n_classes = 5
sfreq = 200

freq_shift = FrequencyShift(
    probability=.5,
    sfreq=sfreq,
    max_delta_freq=2.  # the frequency shifts are sampled now between -2 and 2 Hz
)

sign_flip = SignFlip(probability=.3)

transforms = [
    freq_shift,
    sign_flip
]

# %%
input_window_samples = 500
n_chans = 35
n_classes = 5

model = ShallowFBCSPNet(
    n_chans,
    n_classes,
    input_window_samples=input_window_samples,
    final_conv_length=10
)

# These values we found good for shallow network:
lr = 0.0625 * 0.01
weight_decay = 0

# For deep4 they should be:
# lr = 1 * 0.01
# weight_decay = 0.5 * 0.001

batch_size = 64
n_epochs = 200

clf = EEGClassifier(
    model,
    cropped=True,
    iterator_train=AugmentedDataLoader,
    iterator_train__transforms=transforms,
    criterion=CroppedLoss,
    criterion__loss_function=torch.nn.functional.nll_loss,
    optimizer=torch.optim.AdamW,
    train_split=predefined_split(valid_set),
    optimizer__lr=lr,
    optimizer__weight_decay=weight_decay,
    iterator_train__shuffle=True,
    batch_size=batch_size,
    callbacks=[
        "accuracy", ("lr_scheduler", LRScheduler(
            'CosineAnnealingLR', T_max=n_epochs - 1)),
    ],
    device=device,
)
# Model training for a specified number of epochs. `y` is None as it is already supplied
# in the dataset.
clf.fit(train_set, y=None, epochs=n_epochs)

# %%
y = clf.predict(valid_set)
# %%
y
# %%
y_true = np.array([e[1] for e in valid_set])
y_pred = clf.predict(valid_set)

report = metrics.classification_report(y_true=y_true, y_pred=y_pred)
print(report)
# %%
