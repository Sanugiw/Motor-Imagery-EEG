"""Traditional WPD-CSP: db4 wavelet packet channels before CSP."""

import numpy as np
import pywt

from .csp_features import CSPFeatures
from .csp_wpd import selected_packet_indices


def reconstruct_selected_packet_channels(X, fs: float):
    level = 3 if fs <= 100 else 4
    selected = set(selected_packet_indices(fs, level))
    output = []
    for trial in np.asarray(X, dtype=float):
        channels = []
        for signal in trial:
            packet = pywt.WaveletPacket(signal, "db4", mode="symmetric", maxlevel=level)
            leaves = packet.get_level(level, order="freq")
            for idx, node in enumerate(leaves):
                if idx not in selected:
                    continue
                isolated = pywt.WaveletPacket(None, "db4", mode="symmetric", maxlevel=level)
                isolated[node.path] = node.data
                reconstructed = isolated.reconstruct(update=False)[:signal.size]
                channels.append(reconstructed)
        output.append(channels)
    return np.asarray(output)


def extract_train_test(X_train, y_train, X_test, fs: float, n_pairs: int = 3):
    train_channels = reconstruct_selected_packet_channels(X_train, fs)
    test_channels = reconstruct_selected_packet_channels(X_test, fs)
    model = CSPFeatures(n_pairs).fit(train_channels, y_train)
    return model.transform(train_channels), model.transform(test_channels), model
