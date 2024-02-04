import numpy as np
from scipy.signal import find_peaks
from scipy.stats import zscore
import pywt

def feature_extraction(xbr, Fs, ideltasr, paramVBDS, verb):
    wname = 'sym6'
    hL, hD = pywt.Wavelet(wname).filter_bank[:2]
    dwtmode('per')

    L = 6
    c, l = pywt.wavedec(xbr, wname, level=L)
    cs = np.zeros_like(c)

    Ns = len(ideltasr)
    nsw = np.arange(-1.5 * Fs / 1000, 3 * Fs / 1000)
    indcs = indwavsupp(ideltasr[0] + min(nsw), ideltasr[0] + max(nsw) - 1, l, wname, 12)
    tabcs = np.zeros((Ns, len(indcs)))
    tabcs[0, :] = indcs

    for i in range(1, Ns):
        tabcs[i, :] = indwavsupp(ideltasr[i] + min(nsw), ideltasr[i] + max(nsw) - 1, l, wname, 12)
        if verb and i % 1000 == 0:
            print(f"{i}/{Ns} spikes processed")

    Cs = c[np.array(tabcs).flatten()]

    indselec = indwavsupp(ideltasr[0] - Fs / 2000, ideltasr[0] + Fs / 2000 - 1, l, wname, 16)
    indselec = np.mod(np.where(indselec == indcs)[0], len(indcs))
    vcs = np.var(Cs[:, indselec], axis=1)
    Is = np.argsort(vcs)[::-1]
    plage = indselec[Is[:paramVBDS['nbfeat']]]
    wfeat = Cs[:, plage]

    mind = np.zeros(Ns)
    for ii in range(Ns):
        dist = np.sort(np.sqrt(np.sum((wfeat - wfeat[ii, :])**2, axis=1)))
        mind[ii] = dist[np.ceil(Ns / 100).astype(int)]
    th = np.mean(mind) + 2.64 * np.std(mind)
    Ir = np.where(mind > th)[0]

    wfeat = np.delete(wfeat, Ir, axis=0)
    Cs = np.delete(Cs, Ir, axis=0)
    tabcs = np.delete(tabcs, Ir, axis=0)
    ideltasr = np.delete(ideltasr, Ir)

    paramVBDS['c'] = c
    paramVBDS['l'] = l
    paramVBDS['Cs'] = Cs
    paramVBDS['tabcs'] = tabcs
    paramVBDS['plage'] = plage
    paramVBDS['ideltas'] = np.delete(paramVBDS['ideltas'], Ir)

    return wfeat, paramVBDS, ideltasr

def indwavsupp(start, stop, l, wname, extent):
    h, _ = pywt.Wavelet(wname).filter_bank[:2]
    Nf = len(h)
    L = len(l) - 2
    interval = stop - start + 1
    ind = []

    for ll in range(1, L + 1):
        cinf = max(round((start - Nf / extent) / 2), 1)
        csup = min(round(cinf + interval / 2**ll + Nf / 12), l[-ll])
        ind.extend(list(range(sum(l[:-ll+1]) + cinf, sum(l[:-ll+1]) + csup + 1)))
        start = cinf

    ind = [i for i in ind if i <= sum(l[:-1])]  # remove highest details level for features

    return ind
