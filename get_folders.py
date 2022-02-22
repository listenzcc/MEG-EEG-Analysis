# %%
import pandas as pd

from onstart import CONFIG, logger

# %%
folder = CONFIG['data_folder']

folders = [e for e in folder.iterdir() if e.as_posix().endswith('.ds')]

dcts = []
for folder in folders:
    dcts.append(dict(
        name=folder.name,
        children=len([e for e in folder.iterdir()]),
        path=folder.as_posix(),
    ))

table = pd.DataFrame(dcts)


def _mode(e, default_mode='MEG'):
    if e.startswith('EEG'):
        return 'EEG'

    if e.startswith('MEG'):
        return 'MEG'

    if e.startswith('Noise'):
        return 'Noise'

    return default_mode


table['mode'] = table['name'].map(_mode)

table

# %%
logger.debug('Found subfolders in {} for {} entries'.format(
    folder,
    len(table)
))

# %%
