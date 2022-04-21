
# %%
import plotly.express as px
import numpy as np
from joblib import load
from pathlib import Path
from tqdm.auto import tqdm

from sklearn import metrics

import torch
import skorch
from skorch.callbacks import LRScheduler
from skorch.helper import predefined_split
from braindecode.models import ShallowFBCSPNet, EEGNetv4
from braindecode import EEGClassifier

from braindecode.augmentation import AugmentedDataLoader, SignFlip
from braindecode.augmentation import FrequencyShift

from onstart import LOGGER

# %%
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

# %%
subject_name_table = dict(
    S01='meg-eeg-20220119',
    S02='MEG20220315',
    S03='MEG-EEG20220322',
    S04='MEG20220329'
)


# %%
def _get_x_y_g(filenames):
    X, Y, G = [], [], []

    for name in tqdm(filenames):
        d = load(name)
        run_idx = int(name.as_posix().split('.')[1])
        x = d['data'][:, :, 200:]
        y = d['events'][:, -1]
        y[y == 5] = 0
        g = run_idx + y * 0

        X.append(x)
        Y.append(y)
        G.append(g)

        ch_names = d['ch_names']

    return np.concatenate(X).astype(np.float64), np.concatenate(Y).astype(np.int64), np.concatenate(G).astype(np.int64), ch_names


def _select_mode(X, ch_names, mode='MEG'):
    select = [True for e in ch_names]

    if mode == 'MEG':
        select = [not e.startswith('EEG') for e in ch_names]

    if mode == 'EEG':
        select = [e.startswith('EEG') for e in ch_names]

    x = X[:, select]

    if mode == 'EEG':
        x = x[:, :35]

    LOGGER.debug('Selected mode {} for {}'.format(mode, x.shape))

    return x


def _preprocess(X):
    for j in range(len(X)):
        m1 = np.min(X[j])
        m2 = np.max(X[j])
        X[j] /= m2
    return X


def decoding(X, Y, G, valid_group):
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

    # These values we found good for shallow network:
    lr = 0.0625 * 0.1
    weight_decay = 1.0

    batch_size = 50
    n_epochs = 500

    n_chans = X.shape[1]
    input_window_samples = X.shape[2]

    model = ShallowFBCSPNet(
        n_chans,
        n_classes,
        pool_mode='mean',
        input_window_samples=input_window_samples,
        final_conv_length='auto',
    )

    def train_split(dataset, G=G, valid_group=valid_group):
        X = dataset.X
        Y = dataset.y
        train = skorch.dataset.Dataset(
            X[G != valid_group], Y[G != valid_group])
        valid = skorch.dataset.Dataset(
            X[G == valid_group], Y[G == valid_group])
        return tuple((train, valid))

    clf = EEGClassifier(
        model,
        iterator_train=AugmentedDataLoader,
        # iterator_train__transforms=transforms,
        criterion=torch.nn.NLLLoss,
        optimizer=torch.optim.AdamW,
        optimizer__lr=lr,
        optimizer__weight_decay=weight_decay,
        train_split=train_split,
        batch_size=batch_size,
        callbacks=[
            "accuracy",
            ("lr_scheduler", LRScheduler('CosineAnnealingLR', T_max=n_epochs - 1)),
        ],
        device=device,
    )

    dataset = skorch.dataset.Dataset(X, Y)

    for repeat in range(1):
        clf.fit(dataset, y=None, epochs=n_epochs)

    _, test_dataset = train_split(dataset)

    p = clf.predict(test_dataset.X)

    report = metrics.classification_report(y_true=test_dataset.y,
                                           y_pred=p)

    print(report)
    return report

# %%


allow_override = False


def decode_subject(subject):
    filenames = [e for e in Path('data').iterdir()
                 if e.name.endswith('.dump')
                 and e.name.startswith(subject_name_table[subject])]

    X_raw, Y, G, ch_names = _get_x_y_g(filenames)
    for mode in ['EEG', 'MEG']:
        X = _select_mode(X_raw, ch_names, mode=mode)

        X = _preprocess(X)

        for valid_group in np.unique(G):
            txt_path = Path(
                'reports/{}-{}-{}.txt'.format(subject, mode, valid_group))

            if not allow_override and txt_path.is_file():
                LOGGER.debug('File exists {}, ignoring it.'.format(txt_path))
                continue

            print(subject, mode, valid_group, file=open(txt_path, 'w'))
            print('\n\n', file=open(txt_path, 'a'))

            report = decoding(X, Y, G, valid_group)

            print(report)
            print(report, file=open(txt_path, 'a'))
            LOGGER.debug('Done with {}, {}'.format(txt_path, report))


# %%
for subject in subject_name_table:
    decode_subject(subject)

# %%
