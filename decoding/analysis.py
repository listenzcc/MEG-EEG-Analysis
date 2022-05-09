# %%
import mne
import torch
import numpy as np

from pathlib import Path
from tqdm.auto import tqdm
from sklearn import metrics

from braindecode import EEGClassifier

from braindecode.models import ShallowFBCSPNet, Deep4Net
from braindecode.models import to_dense_prediction_model
from braindecode.models import get_output_shape

from braindecode.training import CroppedLoss

from braindecode.datasets import create_from_mne_epochs

from braindecode.augmentation import AugmentedDataLoader, SignFlip
from braindecode.augmentation import FrequencyShift

from braindecode.preprocessing import exponential_moving_standardize
from braindecode.preprocessing import preprocess
from braindecode.preprocessing import Preprocessor
from braindecode.preprocessing import scale

from skorch.callbacks import LRScheduler
from skorch.helper import predefined_split


# %% --------------------------------------------------------------------------------
# System setup

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using the {}'.format(device))

# %% --------------------------------------------------------------------------------
# User setup

subject_name_table = dict(
    S01='meg-eeg-20220119',
    S02='MEG20220315',
    S03='MEG-EEG20220322',
    S04='MEG20220329'
)

input_window_samples = 601
final_conv_length = 30
sfreq = 200

eeg_model_kwargs = dict(
    n_chans=35,
    n_classes=5,
    input_window_samples=input_window_samples,
    final_conv_length=final_conv_length,
)

mag_model_kwargs = dict(
    n_chans=273,
    n_classes=5,
    input_window_samples=input_window_samples,
    final_conv_length=final_conv_length,
)

# %% --------------------------------------------------------------------------------


def mk_model(model_kwargs, return_n_preds=False):
    '''
    Make a brand new model using the model_kwargs.
    If return_n_preds is True, the aim is to get the n_preds_per_input.
    The meaning is in the document of
    https://braindecode.org/auto_examples/plot_bcic_iv_2a_moabb_cropped.html#
    '''
    kwargs = model_kwargs.copy()

    n_chans = kwargs.pop('n_chans')
    n_classes = kwargs.pop('n_classes')

    model = ShallowFBCSPNet(
        n_chans,
        n_classes,
        **kwargs,
    )

    if return_n_preds:
        to_dense_prediction_model(model)
        n_preds_per_input = get_output_shape(
            model, n_chans, input_window_samples)[2]
        return model, n_preds_per_input

    return model


# It requires dry-run to compute the n_preds_per_input as a startup
model, n_preds_per_input = mk_model(eeg_model_kwargs, return_n_preds=True)
n_preds_per_input


def mk_transforms(sfreq=sfreq):
    '''
    Make the transforms for the model training.
    '''
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

    return transforms


def mk_clf(model, transforms, n_epochs, train_split, batch_size=40):
    '''
    Make the clf from the model and transforms.

    !!! The classifier is highly customized for model setup and play method,
    !!! So, make sure you check and change this before use.
    '''

    # These values we found good for shallow network:
    lr = 0.0625 * 0.01
    weight_decay = 0

    # For deep4 they should be:
    # lr = 1 * 0.01
    # weight_decay = 0.5 * 0.001

    clf = EEGClassifier(
        model,
        cropped=True,
        iterator_train=AugmentedDataLoader,
        iterator_train__transforms=transforms,
        criterion=CroppedLoss,
        criterion__loss_function=torch.nn.functional.nll_loss,
        optimizer=torch.optim.AdamW,
        train_split=train_split,
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

    return clf


def read_subject(subject, input_window_samples=input_window_samples, window_stride_samples=n_preds_per_input):
    '''
    Read the subject's file.

    And convert them into the dataset.

    Returns:
        - mag_dataset, eeg_dataset
    '''

    # ----
    # File names
    filenames = [e for e in Path('data').iterdir()
                 if e.name.endswith('.dump.epoch')
                 and e.name.startswith(subject_name_table[subject])
                 and e.name.split('.')[-2] != '10']

    filenames = sorted(filenames)[2:]

    # ----
    # Make two lists of epochs
    # - One for MAG
    # - Another for EEG

    def get_epochs_list(filenames):
        mag_epochs_list = []
        for e in filenames:
            epochs = mne.read_epochs(e).pick_types(meg=True)
            epochs.events[:, -1] %= 5
            epochs = epochs.crop(tmin=1)
            mag_epochs_list.append(epochs)

        eeg_epochs_list = []
        for e in filenames:
            epochs = mne.read_epochs(e).pick_types(eeg=True)
            drop_ch_names = epochs.ch_names[35:]
            epochs.drop_channels(drop_ch_names)
            epochs.events[:, -1] %= 5
            epochs = epochs.crop(tmin=1)
            eeg_epochs_list.append(epochs)

        print(len(mag_epochs_list), mag_epochs_list[0].get_data().shape)
        print(len(eeg_epochs_list), eeg_epochs_list[0].get_data().shape)

        return mag_epochs_list, eeg_epochs_list

    # The shape of mag_epochs_list[0] is like
    # 40 x 273 x 1001

    # The shape of eeg_epochs_list[0] is like
    # 40 x 35 x 1001

    # mag_epochs_list, eeg_epochs_list = get_epochs_list(filenames[:2])
    mag_epochs_list, eeg_epochs_list = get_epochs_list(filenames)

    # ----
    # Make dataset for MAG and EEG data
    kwargs = dict(
        window_size_samples=input_window_samples,
        window_stride_samples=window_stride_samples,
        drop_last_window=False
    )
    mag_dataset = create_from_mne_epochs(mag_epochs_list, **kwargs)
    eeg_dataset = create_from_mne_epochs(eeg_epochs_list, **kwargs)

    # ----
    # Update the description
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

    # ! The set_description raw code is changed to make it work
    mag_dataset.set_description(description, overwrite=True)
    eeg_dataset.set_description(description, overwrite=True)
    mag_dataset.description, eeg_dataset.description

    return mag_dataset, eeg_dataset, mag_epochs_list, eeg_epochs_list


def preprocess_dataset(dataset, factor):
    '''
    Preprocess the dataset **IN-PLACE**.

    The default parameter is

    - MAG: factor = 1e14
    - EEG: factor = 1e6
    '''

    preprocessors = [
        Preprocessor(scale, factor=factor, apply_on_array=True),
    ]

    preprocess(dataset, preprocessors)

    return


def split_dataset(dataset, valid_group=0):
    '''
    Split the dataset into 'train' and 'valid' set.

    The description of the dataset will be changed accordingly.
    The set label has the column of 'split'.
    '''

    def _map(e):
        if e == valid_group:
            return 'valid'
        return 'train'

    dct = dict(
        split=dataset.description['group'].map(_map)
    )

    dataset.set_description(dct, overwrite=True)

    splitted = dataset.split('split')

    train_set = splitted['train']
    valid_set = splitted['valid']

    print('Split the dataset into train ({}) and valid ({}) set'.format(
        len(train_set), len(valid_set)))

    return train_set, valid_set


# %% --------------------------------------------------------------------------------
# Setup the subject name
subject = 'S01'

n_epochs = 500

# Read files and make dataset
mag_dataset, eeg_dataset, mag_epochs_list, eeg_epochs_list = read_subject(subject)

# %%

# Preprocess dataset
preprocess_dataset(mag_dataset, 1e14)
preprocess_dataset(eeg_dataset, 1e6)

# %% --------------------------------------------------------------------------------
# EEG dataset
# Split the dataset
train_set, valid_set = split_dataset(eeg_dataset, valid_group=3)

# Make the clf
transforms = mk_transforms()
model = mk_model(eeg_model_kwargs)
clf = mk_clf(model, transforms, n_epochs, predefined_split(valid_set))

# Model training for a specified number of epochs.
# `y` is None as it is already supplied in the dataset.
clf.fit(train_set, y=None, epochs=n_epochs)

# %% --------------------------------------------------------------------------------
y_true = np.array([e[1] for e in valid_set])
y_pred = clf.predict(valid_set)

report = metrics.classification_report(y_true=y_true, y_pred=y_pred)
print(report)

confusion_matrix = metrics.confusion_matrix(y_true=y_true, y_pred=y_pred)
print(confusion_matrix)

# %%
