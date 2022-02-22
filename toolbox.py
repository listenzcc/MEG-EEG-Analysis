import mne


def _match_montage(epochs):
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
