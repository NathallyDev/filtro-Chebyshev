import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal


def to_mono(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        return x
    return x.mean(axis=1)


def pcm_to_float(x: np.ndarray) -> np.ndarray:
    # int16 -> float [-1,1]
    if np.issubdtype(x.dtype, np.integer):
        maxv = np.iinfo(x.dtype).max
        return x.astype(np.float64) / maxv
    return x.astype(np.float64)


def float_to_int16(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -1.0, 1.0)
    return (x * 32767.0).astype(np.int16)


def design_cheby2_bandpass(fs, f1, f2, order=6, rs=80):
    nyq = fs / 2.0
    w1, w2 = f1 / nyq, f2 / nyq
    if not (0 < w1 < w2 < 1):
        raise ValueError(f"Faixa inválida: f1={f1}, f2={f2}, nyq={nyq}")
    sos = signal.cheby2(
        N=order,
        rs=rs,                # atenuação na rejeição (dB)
        Wn=[w1, w2],
        btype="bandpass",
        output="sos"
    )
    return sos


def plot_response(sos, fs):
    w, h = signal.sosfreqz(sos, worN=4096, fs=fs)
    mag = 20 * np.log10(np.maximum(np.abs(h), 1e-12))
    plt.figure()
    plt.plot(w, mag)
    plt.title("Resposta em frequência (Chebyshev II - Passa-faixa)")
    plt.xlabel("Frequência (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.grid(True)


def plot_psd(x, y, fs):
    f1, p1 = signal.welch(x, fs=fs, nperseg=4096)
    f2, p2 = signal.welch(y, fs=fs, nperseg=4096)
    plt.figure()
    plt.semilogy(f1, p1, label="Antes")
    plt.semilogy(f2, p2, label="Depois")
    plt.title("PSD (Welch) - Antes vs Depois")
    plt.xlabel("Frequência (Hz)")
    plt.ylabel("Potência")
    plt.grid(True, which="both")
    plt.legend()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_wav", default="Arquivo3.wav")
    ap.add_argument("--out", dest="out_wav", default="Arquivo3_filtrado.wav")
    ap.add_argument("--f1", type=float, default=150.0, help="Corte inferior (Hz)")
    ap.add_argument("--f2", type=float, default=4000.0, help="Corte superior (Hz)")
    ap.add_argument("--order", type=int, default=6, help="Ordem do Cheby2 (ex: 6 ou 8)")
    ap.add_argument("--rs", type=float, default=80.0, help="Atenuação rejeição (dB) (ex: 60-100)")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    fs, x = wavfile.read(args.in_wav)
    print(f"Entrada: {args.in_wav} | fs={fs} | dtype={x.dtype} | shape={x.shape}")

    x = to_mono(x)
    x = pcm_to_float(x)

    print(f"Peak antes: {np.max(np.abs(x)):.4f}")

    # Projeta Cheby2 passa-faixa
    sos = design_cheby2_bandpass(fs, args.f1, args.f2, order=args.order, rs=args.rs)
    print(f"Filtro: Cheby2 passa-faixa | ordem={args.order} | rs={args.rs} dB | faixa={args.f1}-{args.f2} Hz")

    # Filtra com fase zero
    y = signal.sosfiltfilt(sos, x)

    # Normaliza apenas se precisar (evita clipping)
    peak = np.max(np.abs(y))
    print(f"Peak depois (antes de normalizar): {peak:.4f}")
    if peak > 1.0:
        y = y / peak
        print("Normalizado para evitar clipping.")

    wavfile.write(args.out_wav, fs, float_to_int16(y))
    print(f"Saída salva: {args.out_wav}")

    if not args.no_plots:
        plot_response(sos, fs)
        plot_psd(x, y, fs)
        plt.figure()
        t = np.arange(min(len(x), int(0.05*fs))) / fs
        plt.plot(t, x[:len(t)], label="Antes")
        plt.plot(t, y[:len(t)], label="Depois")
        plt.title("Forma de onda (primeiros 50 ms)")
        plt.xlabel("Tempo (s)")
        plt.grid(True)
        plt.legend()
        plt.show()


if __name__ == "__main__":
    main()
