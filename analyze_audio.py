from scipy.io import wavfile
import numpy as np
import sys

files = [('Original','Arquivo3.wav'), ('Filtrado','Arquivo3_filtrado.wav')]
cut = 5000
for name, fname in files:
    try:
        sr, data = wavfile.read(fname)
    except Exception as e:
        print(f"{name}: erro ao ler {fname}: {e}")
        continue
    if data.ndim>1:
        data = data.mean(axis=1)
    data = data.astype(float)
    N = len(data)
    freqs = np.fft.rfftfreq(N, 1/sr)
    fft = np.abs(np.fft.rfft(data))
    total = np.sum(fft**2)
    low = np.sum(fft[freqs<=cut]**2)
    high = np.sum(fft[freqs>cut]**2)
    print(f"{name} ({fname}): sr={sr}, samples={N}")
    print(f"  Energia total: {total:.3e}")
    print(f"  Energia <= {cut} Hz: {low:.3e} ({100*low/total:.2f}%)")
    print(f"  Energia > {cut} Hz: {high:.3e} ({100*high/total:.2f}%)")
    print()
