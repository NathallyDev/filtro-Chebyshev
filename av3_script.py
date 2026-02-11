import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal


def to_mono(x):
    return x if x.ndim == 1 else x.mean(axis=1)

def pcm_to_float(x):
    if np.issubdtype(x.dtype, np.integer):
        return x.astype(np.float64) / np.iinfo(x.dtype).max
    return x.astype(np.float64)

def float_to_int16(x):
    x = np.clip(x, -1, 1)
    return (x * 32767).astype(np.int16)

def cheby2_sos(fs, btype, cutoff, order=6, rs=60):
    nyq = fs / 2
    if isinstance(cutoff, (list, tuple)):
        wn = [c / nyq for c in cutoff]
    else:
        wn = cutoff / nyq
    return signal.cheby2(N=order, rs=rs, Wn=wn, btype=btype, output="sos")

def notch_sos(fs, f0, Q=35):
    # notch em f0 com fator de qualidade Q
    w0 = f0 / (fs/2)
    b, a = signal.iirnotch(w0, Q)
    return signal.tf2sos(b, a)

def sos_filtfilt(sos, x):
    return signal.sosfiltfilt(sos, x)

def plot_psd(x, y, fs, title):
    f1, p1 = signal.welch(x, fs=fs, nperseg=4096)
    f2, p2 = signal.welch(y, fs=fs, nperseg=4096)
    plt.figure()
    plt.semilogy(f1, p1, label="Antes")
    plt.semilogy(f2, p2, label="Depois")
    plt.title(title)
    plt.xlabel("Frequência (Hz)")
    plt.ylabel("Potência")
    plt.grid(True, which="both")
    plt.legend()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_wav", default="Arquivo3.wav")
    ap.add_argument("--out", dest="out_wav", default="Arquivo3_limpo.wav")

    # Ajustes práticos:
    ap.add_argument("--hp", type=float, default=80.0, help="Passa-altas (Hz) p/ tirar rumble")
    ap.add_argument("--lp", type=float, default=12000.0, help="Passa-baixas (Hz) p/ reduzir hiss sem abafar")
    ap.add_argument("--order", type=int, default=6, help="Ordem Cheby2")
    ap.add_argument("--rs", type=float, default=60.0, help="Atenuação (dB) stopband do Cheby2")

    # Notch (rede elétrica)
    ap.add_argument("--hum", type=float, default=60.0, help="Frequência do hum (60 no BR)")
    ap.add_argument("--Q", type=float, default=35.0, help="Q do notch (maior = notch mais estreito)")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    fs, x = wavfile.read(args.in_wav)
    x = pcm_to_float(to_mono(x))

    # 1) Passa-altas leve (Cheby2)
    sos_hp = cheby2_sos(fs, "highpass", args.hp, order=args.order, rs=args.rs)
    y = sos_filtfilt(sos_hp, x)

    # 2) Notch em 60 Hz e 120 Hz (harmônica)
    sos_n1 = notch_sos(fs, args.hum, Q=args.Q)
    y = sos_filtfilt(sos_n1, y)

    if 2*args.hum < fs/2:
        sos_n2 = notch_sos(fs, 2*args.hum, Q=args.Q)
        y = sos_filtfilt(sos_n2, y)

    # 3) Passa-baixas alto (Cheby2) pra tirar hiss extremo sem “abafar”
    sos_lp = cheby2_sos(fs, "lowpass", args.lp, order=args.order, rs=args.rs)
    y = sos_filtfilt(sos_lp, y)

    # normalização segura
    peak = np.max(np.abs(y))
    if peak > 1.0:
        y = y / peak

    wavfile.write(args.out_wav, fs, float_to_int16(y))
    print(f"Salvo: {args.out_wav}")

    if not args.no_plots:
        plot_psd(x, y, fs, "PSD (Welch) - Antes vs Depois")
        plt.show()

if __name__ == "__main__":
    main()
