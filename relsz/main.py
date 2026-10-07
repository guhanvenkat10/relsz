import numpy as np
from epilepsy2bids.annotations import Annotations
from epilepsy2bids.eeg import Eeg

from relsz.model import detect


def main(edf_file, out_file):
    eeg = Eeg.loadEdfAutoDetectMontage(edfFile=edf_file)
    if eeg.fs != 256:
        eeg.resample(256)
    mask = detect(eeg.data, eeg.montage is Eeg.Montage.BIPOLAR)
    n = int(round(eeg.data.shape[1] / 256))
    full = np.zeros(max(n, len(mask)), dtype=int)
    full[: len(mask)] = mask
    Annotations.loadMask(full, 1).saveTsv(out_file)
