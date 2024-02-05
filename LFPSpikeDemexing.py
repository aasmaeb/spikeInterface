import numpy as np
import scipy.fftpack
from scipy.io import wavfile
import pywt

def LFPSpikeDemixing(xbr, xb, Fs, ideltasr, gB, wfeat, paramVBDS, indsupp, verbose):
    T = len(xb)
    Tr = len(xbr)
    gBr = np.zeros(Tr)
    gBr[:Tr//2+1] = gB[np.round(np.linspace(0, T/2, Tr//2+1)).astype(int)]
    gBr[Tr//2+1:Tr] = np.flipud(gBr[1:Tr//2])

    ns = paramVBDS['ds'] * Fs / 1000
    wname = 'sym6'
    L = 6
    K_init = paramVBDS['K']
    Lc = 1 + paramVBDS['nbfeat'] + paramVBDS['nbfeat'] * (paramVBDS['nbfeat'] - 1) / 2

    Klow = max(K_init - paramVBDS['rangeK'], 1)
    Khigh = K_init + paramVBDS['rangeK']

    risstrial = np.full((Khigh, paramVBDS['nbiter']), np.nan)

    # Classif/despiking iterations
    for trial in range(paramVBDS['nbiter']):
        for K in range(Klow, Khigh):
            _, optmixture = GaussianMixture(wfeat, K + 10, K, False)

            # VBDS
            optmixture = RStep(optmixture, paramVBDS['Cs'])
            wr, param = VBClassifDespike7w(xbr, optmixture, ideltasr, paramVBDS, gBr, Fs, wname, L)

            risstrial[K, trial] = (-param['ll'] + 0.5 * (K * Lc - 1) * np.log(len(ideltasr) * paramVBDS['nbfeat']))

            if verbose:
                print('fin K={}, trial={}, riss: {}, nbr: {}'.format(K, trial, risstrial[K, trial], np.sum(param['r'])))

            paramK[K, trial] = {'param': param}

    # Best result selection (min rissanen) and shaping
    imin = np.argmin(risstrial)
    tbest = np.ceil(imin / Khigh)
    Kbest = (imin - 1) % Khigh + 1
    param = paramK[Kbest, tbest]['param']

    # Final w building
    w = scipy.fftpack.ifft(scipy.fftpack.fft(xb - param['mb'] / 2**(L/2)) / np.fft.fftshift((param['gam'] * gB * T + param['varb'])))
    w = param['gam'] * scipy.fftpack.ifft(scipy.fftpack.fft(w) * np.fft.fftshift(T * gB))
    indSpike = np.setdiff1d(np.arange(1, T + 1), indsupp)
    w[indSpike - 1] = wr
    w = scipy.fftpack.ifft(scipy.fftpack.fft(w - param['mb'] / 2**(L/2)) / np.fft.fftshift((param['gam'] * gB * T + param['varb'])))
    w = param['gam'] * scipy.fftpack.ifft(scipy.fftpack.fft(w) * np.fft.fftshift(T * gB))

    # Posterior spike temporal waveforms
    csr = np.zeros(paramVBDS['c'].shape)
    csr[paramVBDS['tabcs']] = param['wSpike']

    yr = pywt.waverec(csr, wname)
    idx = np.transpose(np.tile(ideltasr, (ns, 1)) - np.tile(ns/2 + 1, (len(ideltasr), 1)) + np.arange(1, ns + 1))
    Spike = yr[idx]

    # Mean and std of spike clusters
    S0 = np.dot(param['piapost'].T, Spike) / np.tile(np.sum(param['piapost'], axis=0), (ns, 1))
    varskt = np.zeros(Kbest)

    for k in range(Kbest):
        varskt[k] = np.sum(np.dot(param['piapost'][:, k].T, (Spike - S0[k, :])**2)) / (ns * np.sum(param['piapost'][:, k]))

    # Output structure
    paramVBDS['K'] = Kbest
    paramVBDS['S0'] = S0
    paramVBDS['varskt'] = varskt
    paramVBDS['param'] = param

    return paramVBDS, Spike, w
