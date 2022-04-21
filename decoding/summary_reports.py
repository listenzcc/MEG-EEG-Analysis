# %%
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm

import plotly.express as px

# %%
folder = Path('./reports')

files = [e for e in folder.iterdir() if e.is_file()
         and e.name.endswith('.txt')]
files

# %%
chunk = []
for file in tqdm(files, 'Read files'):
    table = pd.read_table(file)

    # split has the format of
    # ['S04', 'MEG', '9', 'accuracy', '0.42', '...']
    split = table.loc[6].to_string().split()
    subject, mode, run, metric, value, _ = split

    value = float(value)

    chunk.append((subject, mode, run, metric, value, file.as_posix()))

table = pd.DataFrame(chunk,
                     columns=['subject', 'mode', 'run', 'metric', 'value', 'file'])

table

# %%
table

# %%
group = table.groupby(['subject', 'mode'])
group
# %%
m = group['value'].mean().to_frame(name='mean')
s = group['value'].std().to_frame()
m.loc[:, 'std'] = s['value']
m

# %%
