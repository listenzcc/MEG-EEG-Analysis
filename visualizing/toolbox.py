# %%
import os
import mne
import traceback
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from onstart import LOGGER

# %%


class Drawer(object):
    ''' Functional Drawer using Matplotlib '''

    def __init__(self):
        '''
        Method: __init__

        Initialization

        Args:
        - @self
        '''

        # The Latest Added Fig
        self._fig = None
        # The Collection of Figures
        self.figures = []
        pass

    @property
    def fig(self):
        '''
        Method: fig

        Link Method of <fig> to Member <self._fig>

        Args:
        - @self

        Outputs:
        - self._fig

        '''

        return self._fig

    @fig.setter
    def fig(self, f):
        '''
        Method: fig

        Setter of <fig> Method,
        it will set the latest figure as [f] and append it to the collection

        Args:
        - @self,
        - @f: The incoming figure

        '''
        self._fig = f
        self.figures.append(f)
        LOGGER.debug(
            'Added new Figure, Collection size is {}'.format(len(self.figures)))

        pass

    def clear(self):
        '''
        Method: clear

        Clear the Collection of Figures,
        the figures will be closed before discarded from the collection.

        '''

        success, fail = 0, 0
        for fig in self.figures:
            try:
                plt.close(fig)
                success += 1
            except:
                fail += 1

        if fail > 0:
            LOGGER.warn('Failed Close the figures for {} times'.format(fail))

        self.figures = []
        LOGGER.debug(
            'Cleared Collection, SuccessRate is {} | {}'.format(success, fail))

        pass

    def save(self, pdfPath, override=True):
        '''
        Method: save

        Save the Collection into [pdfPath],
        it is allowed to override existing files according to [override] option.

        Args:
        - @self, pdfPath, override=True

        Returns:
        - Successful Code of
          - 0: Success;
          - 1: Incorrect pdfPath;
          - 2: Block by File Existing Situation;
          - 3: Others.

        '''

        if not pdfPath.endswith('.pdf'):
            LOGGER.error('Failed to Save Pdf File of {}'.format(pdfPath))
            return 1

        if os.path.isfile(pdfPath) and not override:
            LOGGER.error(
                'Failed to Save Pdf File to {}, since Override is not Allowed'.format(pdfPath))
            return 2

        try:
            with PdfPages(pdfPath, 'w') as pp:
                for fig in self.figures:
                    pp.savefig(fig)

            LOGGER.debug('Saved {} Figures into {}'.format(
                len(self.figures), pdfPath))
            return 0
        except Exception as err:
            LOGGER.error(
                'Failed to Save Pdf File to {}, since {}, see the Debug Log for the Details'.format(pdfPath, err))
            LOGGER.debug(traceback.format_exc())
            return 3

        pass


# %%
def _match_eeg_montage(epochs):
    '''
     Match the montage between standard montage and experiment setup
    '''

    # The experiment setup
    txt = '''
    f5, f3, f1, fz, f2, f4, f6,
    fc5, fc3, fc1, fcz, fc2, fc4, fc6,
    c5, c3, c1, cz, c2, c4, c6,
    cp5, cp3, cp1, cpz, cp2, cp4, cp6,
    p5, p3, p1, pz, p2, p4, p6
    '''

    # Parse the txt
    def _str(e):
        return e[:-1].upper() + e[-1]

    eeg_custom_montage_table = [_str(e)
                                for e in txt.replace('\n', '').replace(' ', '').split(',')]

    # Make the standard montage with biosemi64 setup
    montage64 = mne.channels.make_standard_montage('biosemi64')

    # Separate the channels
    # inside: The channels are used
    # outside: The channels are not used
    channels_inside = eeg_custom_montage_table

    channels_outside = [e
                        for e in montage64.ch_names
                        if e not in eeg_custom_montage_table]

    # Concat the inside and outside channels by their order
    new_channels = channels_inside + channels_outside

    # Rename the standard montage with the names in the experiment setup
    new_montage64_ch_names = []
    for name in montage64.ch_names:
        idx = new_channels.index(name)
        new_montage64_ch_names.append('EEG{:03d}-4504'.format(idx+1))

    # Make sure the renamed channels are correct
    # The check is very RESTRICTIVE
    a = len(set(epochs.ch_names))
    b = len(set(new_montage64_ch_names))
    c = len(set(epochs.ch_names + new_montage64_ch_names))
    assert(all([a == 64, b == 64, c == 64]))

    # Perform the renaming and setup the new montage
    montage64.ch_names = new_montage64_ch_names
    epochs.set_montage(montage64)

    # Pick the channels in use
    epochs.load_data()
    epochs.pick_channels(epochs.ch_names[:len(channels_inside)])

    return epochs


def _append_SSP(epochs):
    '''
    Estimate the SSP projs for the epochs,
    and add them
    '''

    projs = mne.compute_proj_epochs(epochs, n_jobs=32)
    epochs.add_proj(projs)
    return epochs
