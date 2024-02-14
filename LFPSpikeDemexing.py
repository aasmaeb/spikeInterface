import numpy as np
from sklearn.mixture import GaussianMixture

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

num_waveforms = 1000
waveform_length = 50

# Generate random data for wfeat
wfeat = np.random.rand(num_waveforms, waveform_length)

K_init = paramVBDS['K']
Klow = max(K_init - paramVBDS['rangeK'], 1)
Khigh = K_init + paramVBDS['rangeK']

for K in range(Klow, Khigh):
    gmm = GaussianMixture(n_components=K + 10, covariance_type='full')
    gmm.fit(wfeat)

    # Predict cluster labels for each data point
    labels = gmm.predict(wfeat)
