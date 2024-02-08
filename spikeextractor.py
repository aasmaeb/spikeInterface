import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, convolve, find_peaks
import scipy.io as sio

def spikeExtractor(xb, Fs, paramVBDS, display, verb):
    l, c = xb.shape
    T = max(l, c)

    # band pass filtering
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
    Ns = len(ideltas)
    print("Ns",Ns)
    deltas = np.zeros((max(l, c), 1)) 
    deltas[ideltas, 0] = 1

    deltafen = convolve(deltas[:, 0], np.ones(ns), mode='full')
    deltafen = deltafen[int(ns / 2):len(deltafen) - int(ns / 2)]
    deltafen = np.append(deltafen, 0)
    xb_spkband = xb_spkband[:len(deltafen)]


    print("xb_",len(xb_spkband))
    print("delta",len(deltafen))
    
    spkband = np.array([])

    spkband = deltafen * xb_spkband
    spk = deltafen * xb
  
    # Trouvez les indices non nuls
    iNZ = np.where(spkband != 0)[0]
    print("iNZ",len(iNZ))
    spk1=spk[0]

    # Filtrez les valeurs non nulles
    spkband = spkband[iNZ]
    spk1 = spk1[iNZ]

    # Redimensionnez les tableaux correctement
    spkband = np.reshape(spkband, (Ns, ns)).T
    print("len spkband apres reshape",len(spkband))
    spk1 = np.reshape(spk1, (Ns, ns)).T

    # Filtrez les outliers
    mspkband = np.max(np.abs(spkband), axis=0)

    print("mspkband",len(mspkband))
    spkband1= np.transpose(spkband)
    spk1= np.transpose(spk)

    spkband1= spkband1[mspkband<= thrQH,:]
    #spk1= spk1[mspkband <= thrQH,:] 

    ideltas = ideltas[mspkband<= thrQH] 

 
    Ns = len(ideltas)
    deltas = np.zeros((max(l, c), 1))
    deltas[ideltas] = 1


    # paramVBDS update with ideltas
    paramVBDS['ideltas'] = ideltas

    if display:
        plt.figure()
        plt.plot(spkband1.T)
        plt.title('Superimposed extracted spike waveforms (from filtered data)')
        plt.show()

    if verb:
        print('####################')
        print('END OF SPIKE EXTRACTION')
        print('Number of detected events:', Ns)
        print('####################\n')

    return xb_spkband, spkband, spk, paramVBDS

    # données de test

if __name__=='__main__':

    # les données

    # Load the .mat file into a dictionary
    mat_contents = sio.loadmat('test1.mat')
    # Access the variables you need from the dictionary
    xb = mat_contents['xb']
    Fs= mat_contents['Fs']
    #Fs = mat_contents['Fs'][0, 0]  # Extrait la valeur unique de la matrice 1x1

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

    