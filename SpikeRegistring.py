import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

def spike_registering(xb, xb_spkband, Fs, spkband, spk, ideltas, verb):
    Ns = len(ideltas)
    T = len(xb)
    vecun = np.ones(20)
    ns = max(Fs / 1e3 * 3, 2 ** 5)
    ups = 10

    for ii in range(Ns):
        y = interp1d(np.arange(len(spkband[ii])), spkband[ii], kind='linear', fill_value='extrapolate')(np.linspace(0, len(spkband[ii]) - 1, len(spkband[ii]) * ups))

        if len(y) == 0:
            continue  # Skip if the length is zero

        im = np.argmax(np.abs(y[int(ns * ups / 2 - 2 * ups):int(ns * ups / 2 + 2 * ups)]))
        difm = 2 * ups - im + 1

        yr = np.concatenate((y[0] * vecun[0:difm], y[max(0, -difm + 1):min(ns * ups, ns * ups - difm)], y[-1] * vecun[0:-difm]))
        spkband[ii, :] = yr[::ups][:len(spkband[ii])]  # Fix the indexing here

        y = interp1d(np.arange(len(spk[ii])), spk[ii], kind='linear', fill_value='extrapolate')(np.linspace(0, len(spk[ii]) - 1, len(spk[ii]) * ups))

        if len(y) == 0:
            continue  # Skip if the length is zero

        yr = np.concatenate((y[0] * vecun[0:difm], y[max(0, -difm + 1):min(ns * ups, ns * ups - difm)], y[-1] * vecun[0:-difm]))
        spk[ii, :] = yr[::ups][:len(spk[ii])]  # Fix the indexing here

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
import numpy as np
import matplotlib.pyplot as plt

def spike_registering(xb, xb_spkband, Fs, spkband, spk, ideltas, verb):
    # ... (use the updated spike_registering function from the previous responses)

  # Generate synthetic data with spikes
  Fs = 1000
  T = 5
  t = np.arange(0, T, 1/Fs)
  f_spk = 10
  spike_duration = 0.02
  Ns = 5

  # Create random data
  xb = np.random.randn(len(t))

  # Create spikes at random positions
  spk = np.zeros((Ns, len(t)))
  ideltas = np.sort(np.random.choice(range(len(t)), Ns, replace=False))

  for i, delta in enumerate(ideltas):
      spk[i, delta:int(delta + Fs * spike_duration)] = np.sin(2 * np.pi * f_spk * t[:int(Fs * spike_duration)])

  # Apply bandpass filter to the spike data
  spkband = np.zeros((Ns, len(t)))
  for i in range(Ns):
      spkband[i, :] = np.convolve(spk[i, :], np.ones(int(Fs * spike_duration)) / (Fs * spike_duration), mode='same')

  # Apply bandpass filter to the random data
  xb_spkband = np.convolve(xb, np.ones(int(Fs * spike_duration)) / (Fs * spike_duration), mode='same')

  # Apply spike registering function
  verb = True
  xbr, xb_spkbandr, ideltasr = spike_registering(xb, xb_spkband, Fs, spkband, spk, ideltas, verb)

  # Plot the results
  plt.figure(figsize=(12, 8))

  plt.subplot(4, 1, 1)
  plt.plot(t, xb, label='Raw Data')
  plt.plot(t[ideltas], xb[ideltas], 'ro', label='Spikes')
  plt.title('Raw Data with Spikes')
  plt.legend()

  plt.subplot(4, 1, 2)
  plt.plot(t, xb_spkband, label='Filtered Data (Spike Band)')
  plt.title('Filtered Data in Spike Band')
  plt.legend()

  plt.subplot(4, 1, 3)
  plt.plot(t, xbr, label='Registered Raw Data')
  plt.plot(t[ideltasr], xbr[ideltasr], 'go', label='Registered Spikes')
  plt.title('Registered Raw Data with Spikes')
  plt.legend()

  plt.subplot(4, 1, 4)
  plt.plot(t, xb_spkbandr, label='Registered Filtesred Data')
  plt.title('Registered Filtered Data with Spikes')
  plt.legend()

  plt.tight_layout()
  plt.show()

