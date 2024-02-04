from operator import length_hint
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
import matplotlib.pyplot as plt
from scipy.signal import convolve
import scipy.io as sio

def spikeExtractor(xb, Fs, paramVBDS, display=True, verb=True):
    # Filtrage


    low = paramVBDS['spike_band'][0] / (Fs/2)
    high = paramVBDS['spike_band'][1] / (Fs/2)
    b, a = butter(2, [low, high], btype='band')
    xb_spkband = filtfilt(b, a, xb)


    # Thresholds
    thrQ = paramVBDS['coef'] * np.median(np.abs(xb_spkband)) / 0.6745
    thrQH = paramVBDS['coefH'] * thrQ

    # Détection des pics
    if xb_spkband.ndim != 1:
        xb_spkband = xb_spkband.flatten()

    plt.figure()
    plt.plot(xb_spkband)
    plt.title('Données filtrées')
    plt.show()

    peaks, _ = find_peaks(np.abs(xb_spkband), distance=50)

    ideltas = peaks[np.abs(xb_spkband[peaks]) > thrQ]

    # Eliminate overlapping events
    mini = np.min(np.diff(ideltas))
    while mini < Fs * paramVBDS['ds'] / 1000:
        isuppr = np.argmin(np.abs(xb_spkband[ideltas[imini:imini+2]]))
        ideltas= np.delete(ideltas, imini + isuppr - 1)
        mini = np.min(np.diff(ideltas))


    ns = max(int(Fs * paramVBDS['ds'] / 1000), 32)

    # Continuez avec le reste de votre code...


    # Avoid border effects
    ideltas[(ideltas <= ns)] =[]
    ideltas[ (ideltas >= len(xb[0]) - ns)]=[]


    # Eliminate positive or negative spikes if very low
    if np.sum(xb_spkband[ideltas] < 0) < 50:
        ideltas = ideltas[xb_spkband[ideltas] > 0]
    if np.sum(xb_spkband[ideltas] > 0) < 50:
        ideltas = ideltas[xb_spkband[ideltas] < 0]


    if display:
        plt.figure()
        plt.plot(xb_spkband)
        plt.plot(ideltas, xb_spkband[ideltas], 'r*')
        plt.title('Filtered data with detected spikes as *')
        plt.show()


    # Extract spike waveforms

    ns = max(int(Fs /paramVBDS['ds'] * 1000), 2**5)

    deltas = np.zeros(len(xb[0]))
    deltas[ideltas] = 1

    deltafen = convolve(deltas, np.ones(int(ns)), mode='full')[:len(xb_spkband)]
    spkband = deltafen * xb_spkband
    spk = deltafen * xb
    
    # Outliers filtering

    mspkband = np.max(np.abs(spkband), axis=0)

    print("Valeurs maximales (mspkband) :", mspkband)

    mask = mspkband >= thrQH
    spkband = spkband[:, mask]  # Assurez-vous que les colonnes correspondant au masque sont sélectionnées
    spk = spk[:, mask]  # Assurez-vous de faire la même opération pour spk
    ideltas = ideltas[mask]

    transposed_data = list(map(list, zip(*spkband)))

    # Créer un graphique à lignes
    for series in transposed_data:
        plt.plot(series)

    # Afficher le graphe
    plt.show()

    print('mask', mask)
    print('spk',spkband)


    #if display:
       # plt.figure()
       # plt.plot(spkband.T)
      # plt.title('Superimposed extracted spike waveforms (from filtered data)')
        #plt.show()

    if verb:
        print('####################')
        print('END OF SPIKE EXTRACTION')
        print('Number of detected events:', len(ideltas))
        print('####################')
        print(' ')

    paramVBDS['ideltas'] = ideltas

    return xb_spkband, spkband, spk, paramVBDS


# données de test

if __name__=='__main__':

    # les données

    # Load the .mat file into a dictionary
    mat_contents = sio.loadmat('test1.mat')
    # Access the variables you need from the dictionary
    xb = mat_contents['xb']
    Fs= mat_contents['Fs']
    paramVBDS = {
        'spike_band': [300, 3000],
        'coef': 4,
        'coefH': 5,
        'ds': 3,
        'nbfeat': 6,
        'mode': 1,
        'K': 2,
        'rangeK': 1,
        'nbiter': 10
    }

    display= True
    verb=True
    mode=True
    spikeExtractor(xb,Fs,paramVBDS,display,verb)