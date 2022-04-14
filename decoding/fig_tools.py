'''
FileName: fig_tools.py
Author: Chuncheng
Version: V0.0
Purpose: Provide One-Sentence Figure Tool
'''

import os
import traceback
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from onstart import LOGGER


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
