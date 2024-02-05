import numpy as np
from scipy.interpolate import interp1d

def spike_registering(xb, xb_spkband, Fs, spkband, spk, ideltas, verb):
    Ns = len(ideltas)
    T = len(xb)
    vecun = np.ones(20)
    ns = max(Fs / 1e3 * 3, 2 ** 5)
    ups = 10

    for ii in range(Ns):
        y = interp1d(np.arange(len(spkband[ii])), spkband[ii], kind='linear', fill_value='extrapolate')(np.linspace(0, len(spkband[ii]) - 1, len(spkband[ii]) * ups))
        im = np.argmax(np.abs(y[int(ns * ups / 2 - 2 * ups):int(ns * ups / 2 + 2 * ups)]))
        difm = 2 * ups - im + 1
        yr = np.concatenate((y[0] * vecun[0:difm], y[max(0, -difm + 1):min(ns * ups, ns * ups - difm)], y[-1] * vecun[0:-difm]))
        spkband[ii, :] = yr[ups:ups:len(yr) - 1]

        y = interp1d(np.arange(len(spk[ii])), spk[ii], kind='linear', fill_value='extrapolate')(np.linspace(0, len(spk[ii]) - 1, len(spk[ii]) * ups))
        yr = np.concatenate((y[0] * vecun[0:difm], y[max(0, -difm + 1):min(ns * ups, ns * ups - difm)], y[-1] * vecun[0:-difm]))
        spk[ii, :] = yr[ups:ups:len(yr) - 1]

    xb_spkbandr = xb_spkband.copy()
    xbr = xb.copy()
    ideltass = np.concatenate(([0], ideltas))
    ideltasr = np.concatenate(([0], ideltas))
    lsupp = np.zeros(Ns)
    indsupp = []
    tm = np.zeros(Ns)

    for i in range(1, Ns + 1):
        rdec = ideltasr[i] % 2 ** 5
        lsupp[i - 1] = rdec
        tm[i - 1] = int((ideltass[i] + ideltass[i - 1]) / 2 - rdec / 2)
        ideltasr[i:] -= rdec
        indsupp.extend(list(range(tm[i - 1] + 1, tm[i - 1] + rdec)))
        xbr[ideltass[i] - int(ns / 2) + 1:ideltass[i] + int(ns / 2)] = spk[i - 1, :]
        xb_spkbandr[ideltass[i] - int(ns / 2) + 1:ideltass[i] + int(ns / 2)] = spkband[i - 1, :]

    if len(indsupp) % 2 == 1:
        indsupp.append(T)

    xb_spkbandr = np.delete(xb_spkbandr, indsupp)
    xbr = np.delete(xbr, indsupp)
    ideltasr = ideltasr[1:]

    if verb:
        print(' ')
        print('####################')
        print('DATA READY FOR FEATURE EXTRACTION')
        print('####################')
        print(' ')

    return xbr, xb_spkbandr, ideltasr

# Example usage:

