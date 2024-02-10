import numpy as np
from scipy.interpolate import interp1d

def spike_registering(xb, xb_spkband, Fs, spkband, spk, ideltas, verb):
    Ns = len(ideltas)
    T = len(xb)
    vecun = np.ones(20)
    ns = max(int(Fs / 1e3 * 3), 2 ** 5)  # Correction du calcul de ns
    ups = 10

    for ii in range(Ns):
        y = interp1d(np.arange(len(spkband[ii])), spkband[ii], kind='linear', fill_value='extrapolate')(np.linspace(0, len(spkband[ii]) - 1, len(spkband[ii]) * ups))
        im = np.argmax(np.abs(y[int(ns * ups / 2 - 2 * ups):int(ns * ups / 2 + 2 * ups)]))
        difm = 2 * ups - im + 1
        # Calculer la longueur attendue de yr
        expected_length = len(spkband[ii, :])

        # Ajuster le dimensionnement de yr en fonction de la longueur attendue
        yr = np.concatenate((y[0] * vecun[0:difm], y[max(0, -difm + 1):min(ns * ups, ns * ups - difm)], y[-1] * vecun[0:-difm]))
        yr = np.resize(yr, expected_length)
        spkband[ii, :] = yr[:T]  


        y = interp1d(np.arange(len(spk[ii])), spk[ii], kind='linear', fill_value='extrapolate')(np.linspace(0, len(spk[ii]) - 1, len(spk[ii]) * ups))
        yr = np.concatenate((y[0] * vecun[0:difm], y[max(0, -difm + 1):min(ns * ups, ns * ups - difm)], y[-1] * vecun[0:-difm]))
        expected_length1= len(spk[ii,:])
        yr= np.resize(yr,expected_length1)
        spk[ii, :] = yr[:T] 

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
        indsupp.extend(list(range(int(tm[i - 1] + 1), int(tm[i - 1] + rdec))))
        spk_slice = spk[i - 1, :ns]
        spk_slice = np.resize(spk_slice, ideltass[i] + int(ns / 2) - (ideltass[i] - int(ns / 2) + 1))
        xbr[ideltass[i] - int(ns / 2) + 1:ideltass[i] + int(ns / 2)] = spk_slice

        spkband_slice = spkband[i - 1, :ideltass[i] + int(ns / 2) - (ideltass[i] - int(ns / 2) + 1)]
        xb_spkbandr[ideltass[i] - int(ns / 2) + 1:ideltass[i] + int(ns / 2)] = spkband_slice

        spk_slice = spk[i - 1, :ideltass[i] + int(ns / 2) - (ideltass[i] - int(ns / 2) + 1)]
        xbr[ideltass[i] - int(ns / 2) + 1:ideltass[i] + int(ns / 2)] = spk_slice

        #xb_spkbandr[ideltass[i] - int(ns / 2) + 1:ideltass[i] + int(ns / 2)] = spkband[i - 1, :]

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


# Génération de données factices pour le test
T = 1000
Fs = 1000
Ns = 10
xb = np.random.randn(T)
xb_spkband = np.random.randn(T)
spkband = np.random.randn(Ns, T)
spk = np.random.randn(Ns, T)
ideltas = np.sort(np.random.choice(T, Ns, replace=False))

# Appel de la fonction spike_registering
xbr, xb_spkbandr, ideltasr = spike_registering(xb, xb_spkband, Fs, spkband, spk, ideltas, verb=True)

# Visualisation des résultats
plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.plot(xb, label='Données brutes')
plt.plot(ideltas, xb[ideltas], 'r*', markersize=10, label='Pics détectés')
plt.title('Données brutes avec les pics détectés')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(xbr, label='Données brutes après enregistrement de pics')
plt.plot(ideltasr, xbr[ideltasr], 'r*', markersize=10, label='Pics enregistrés')
plt.title('Données brutes après enregistrement de pics')
plt.legend()

plt.tight_layout()
plt.show()

# Example usage:

