# %%
import re
import pandas as pd
from pathlib import Path

from onstart import LOGGER

# %%


def _mode_by_name(name, default_mode='N.A.'):
    pat = re.compile('^S\d{2}_G33IA_\d{8}_\d{2}\.ds$')
    if pat.match(name):
        LOGGER.debug('Found MEG for {}'.format(name))
        return 'MEG'

    if name.startswith('Noise'):
        return 'Noise'

    return default_mode

# %%


def mk_file_table(raw_folder, select_mode='MEG'):
    if isinstance(raw_folder, str):
        raw_folder = Path(raw_folder)

    folders = [e for e in raw_folder.iterdir()
               if e.as_posix().endswith('.ds')]

    dct_list = []
    for folder in folders:
        dct_list.append(dict(
            subject=raw_folder.name,
            name=folder.name,
            mode='N.A.',
            children=len([e for e in folder.iterdir()]),
            path=folder.as_posix(),
        ))

    table = pd.DataFrame(dct_list)

    table['mode'] = table['name'].map(_mode_by_name)

    if select_mode is not None:
        table = table.query('mode=="{}"'.format(select_mode))
        table.index = range(len(table))
    count = len(table)

    LOGGER.debug('Made table for subject folder {}'.format(raw_folder))
    LOGGER.debug('Found {} entries for {}'.format(count, select_mode))

    return table

# %%
