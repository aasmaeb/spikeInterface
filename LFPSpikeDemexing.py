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

   
    imin = np.argmin(risstrial)
    tbest = np.ceil(imin / Khigh)
    Kbest = (imin - 1) % Khigh + 1
    param = paramK[Kbest, tbest]['param']

  
    w = scipy.fftpack.ifft(scipy.fftpack.fft(xb - param['mb'] / 2**(L/2)) / np.fft.fftshift((param['gam'] * gB * T + param['varb'])))
    w = param['gam'] * scipy.fftpack.ifft(scipy.fftpack.fft(w) * np.fft.fftshift(T * gB))
    indSpike = np.setdiff1d(np.arange(1, T + 1), indsupp)
    w[indSpike - 1] = wr
    w = scipy.fftpack.ifft(scipy.fftpack.fft(w - param['mb'] / 2**(L/2)) / np.fft.fftshift((param['gam'] * gB * T + param['varb'])))
    w = param['gam'] * scipy.fftpack.ifft(scipy.fftpack.fft(w) * np.fft.fftshift(T * gB))

    csr = np.zeros(paramVBDS['c'].shape)
    csr[paramVBDS['tabcs']] = param['wSpike']

    yr = pywt.waverec(csr, wname)
    idx = np.transpose(np.tile(ideltasr, (ns, 1)) - np.tile(ns/2 + 1, (len(ideltasr), 1)) + np.arange(1, ns + 1))
    Spike = yr[idx]

  
    S0 = np.dot(param['piapost'].T, Spike) / np.tile(np.sum(param['piapost'], axis=0), (ns, 1))
    varskt = np.zeros(Kbest)

    for k in range(Kbest):
        varskt[k] = np.sum(np.dot(param['piapost'][:, k].T, (Spike - S0[k, :])**2)) / (ns * np.sum(param['piapost'][:, k]))

 
    paramVBDS['K'] = Kbest
    paramVBDS['S0'] = S0
    paramVBDS['varskt'] = varskt
    paramVBDS['param'] = param

    return paramVBDS, Spike, w


##VBClassifDespike7w
import numpy as np
import scipy.signal
import scipy.fftpack
import pywt

def VBClassifDespike7w(y, minit, ideltas, paramFeat, g, Sr, wname, level):
    nbiter = 5
    T = len(y)
    K = minit['K']
    Ns = len(ideltas)
    ns = paramFeat['tabcs'].shape[1]
    gam = 1
    r = np.zeros(Ns)
    covsmem = np.zeros(nbiter)

  
    cy, l = pywt.wavedec(y, wname, level=level)


    b, a = scipy.signal.butter(1, 50 / Sr * 2)
    w = scipy.signal.filtfilt(b, a, y)

  
    pik = np.ones(K) / K
    mb = np.mean(y)
    cmb = np.concatenate([2**(level/2) * mb * np.ones(l[0]), np.zeros(np.sum(l[1:]))])
    cw = pywt.wavedec(w, wname, level=level)
    varb = np.var(y) / 1000

   
    Cs = cy[paramFeat['tabcs']] - cw[paramFeat['tabcs']] - cmb[paramFeat['tabcs']]

    piapost = minit['pnk']
    Sw0 = np.sum((piapost.T * Cs).T, axis=0) / np.sum(piapost, axis=0)
    CovSw = np.zeros((K, ns, ns))
    invSw = np.zeros((K, ns, ns))
    reg = np.max(np.abs(minit['cluster'][0]['R']))
    wSpike = np.zeros((Ns, ns))
    covSwapost = np.zeros((Ns, ns, ns))

   
    circone = np.zeros((ns * ns, ns))
    tocirc = np.concatenate([np.ones(ns), np.zeros((ns - 1) * ns)])
    tocirc2 = np.concatenate([np.ones(ns), np.zeros((K - 1) * ns)])
    for i in range(ns):
        circone[:, i] = np.roll(tocirc, -i * ns)
    circoneK = np.tile(circone, (K, 1))
    indtrace = np.arange(ns, ns * ns, ns) + np.arange(ns)

    it = 0
    while it < nbiter:
        it += 1
        indr = np.where(r == 1)[0]
        indk = np.where(r == 0)[0]

     
        Cs = cy[paramFeat['tabcs']] - cw[paramFeat['tabcs']] - cmb[paramFeat['tabcs']]
        invSwapost = 1 / varb * np.eye(ns) + np.sum(piapost[:, :, np.newaxis] * invSw[:, np.newaxis, :, :], axis=0)
        covSW0 = np.sum((np.dot(invSw.reshape(K, ns * ns).T, circoneK * np.tile(Sw0, (ns, K)).T) * circoneK2).reshape(ns, K * ns, K), axis=0).T
        covSW0 = piapost * covSW0.T

        for n in range(Ns):
            covSwapost[n, :, :] = np.linalg.inv(invSwapost[n, :, :])
            wSpike[n, :] = np.dot(covSwapost[n, :, :], 1 / varb * Cs[n, :] + covSW0[n, :])

        
        wSpikez = wSpike[:, paramFeat['plage']]
        CovSwz = CovSw[:, paramFeat['plage'], paramFeat['plage']]
        SW0z = Sw0[:, paramFeat['plage']]

        piapost = np.zeros((Ns, K))
        for k in range(K):
            Covsk = CovSwz[k, :, :]
            piapost[:, k] = 1 / np.sqrt(np.linalg.det(Covsk)) * np.exp(
                -0.5 * np.diag(np.dot((wSpikez - SW0z[k, :]).T, np.linalg.inv(Covsk)).dot(wSpikez - SW0z[k, :])))

        piapostnn = np.zeros_like(piapost)
        piapostnn[indk, :] = np.tile(pik, (Ns - np.sum(r), 1)) * piapost[indk, :]
        piapost[indk, :] = piapostnn[indk, :] / np.tile(np.sum(piapostnn[indk, :], axis=1)[:, np.newaxis], (1, K))
        piapost[indr, :] = 0
        r[np.where(np.isnan(np.sum(piapost, axis=1)))] = 1
        indr = np.where(r == 1)[0]
        indk = np.where(r == 0)[0]
        piapost[indr, :] = 0

      
        cw = cy - cmb
        cw[paramFeat['tabcs']] = cw[paramFeat['tabcs']] - wSpike
        w = pywt.waverec(cw, wname)
        w = np.fft.ifft(np.fft.fft(w) / np.fft.fftshift((gam * g * T + varb)))
        w = gam * np.fft.ifft(np.fft.fft(w) * np.fft.fftshift(T * g))
        cw = pywt.wavedec(w, wname, level=level)

      
        for k in range(K):
            if np.sum(piapost[indk, k]) > 1 and pik[k] > 0.001:
                Sw0[k, :] = 1 / np.sum(piapost[indk, k]) * np.sum(piapost[indk, k]) * wSpike[indk, :]
                CovSw[k, :, :] = np.dot((wSpike[indk, :] - Sw0[k, :]).T * np.tile(piapost[indk, k], (ns, 1)),
                                        wSpike[indk, :] - Sw0[k, :]) / np.sum(piapost[indk, k])
                invSw[k, :, :] = np.linalg.inv(CovSw[k, :, :] + reg * 1e-5 * np.eye(ns))
            else:
                CovSw[k, :, :] = 0
        covsmem[it - 1] = np.sum(np.sum(CovSw**2))

       
        cb = cy - cw
        cb[paramFeat['tabcs']] = cb[paramFeat['tabcs']] - wSpike
        mb = np.mean(cb[:l[0]])
        cmb = np.concatenate([mb * np.ones(l[0]), np.zeros(np.sum(l[1:]))])
        eigbm = np.sum(1 / (1 / (gam * g * T) + 1 / varb))
        varb = np.var(cb[:l[0]] - cmb[:l[0]]) + eigbm / T + np.sum(np.sum(covSwapost[:, indtrace], axis=2), axis=0) / T

       
        eigb = np.sum(1 / (gam * g * T + varb))
        gam = varb * gam * eigb / T + np.sum(w * np.fft.ifft(np.fft.fft(w) / np.fft.fftshift((gam * g * T)))) / T

        # les proporgtions
        pik = np.sum(piapost[indk, :], axis=0) / (Ns - np.sum(r))

    param = {'it': it, 'varb': varb, 'mb': mb, 'vars': covsmem, 'r': r, 'Sw0': Sw0, 'wSpike': wSpike,
             'gam': gam, 'piapost': piapost, 'pik': pik, 'll': np.sum(np.log(np.sum(piapostnn[indk, :], axis=1)))}

    return w, param

