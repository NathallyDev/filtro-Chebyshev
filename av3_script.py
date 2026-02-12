"""
PROCESSAMENTO DE AUDIO COM FILTRO CHEBYSHEV TIPO II E REDUCAO DE RUIDO
========================================================================

Trabalho implementado para a disciplina de Sistemas Lineares.

Este script aplica tecnicas de processamento de sinais para reducao de ruido
em arquivos de audio usando:

1. Wiener Filter (STFT) - Estima e subtrai ruido baseado em energia minima
2. Chebyshev Tipo II - Filtro passa-baixas de ordem alta para atenuar frequencias altas
3. Filtro Butterworth Passa-altos - Remove componentes de baixa frequencia
4. Noise Gate - Silencia picos de ruido nos periodos de silencio

Dependencias: numpy, scipy, matplotlib
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile

# =========================
# CONFIGURACAO DO PROCESSAMENTO
# =========================
INPUT_WAV  = "Arquivo3.wav"
OUTPUT_WAV = "Arquivo3_filtrado_v2.wav"
OUTPUT_FIG = "analise_filtrado_v2.png"

# Parametros STFT (Transformada de Fourier em Tempo Curto)
N_FFT = 2048
HOP = 512
WINDOW = "hamming"

# Wiener Filter - Coeficiente de agressividade e piso
WIENER_ALPHA = 5.4
WIENER_FLOOR = 0.027
NOISE_PERCENT = 0.37

# Spectral Gating - Limiar para eliminar componentes fracos
ENABLE_SPECTRAL_GATE = True
GATE_THRESHOLD = 0.055

# Noise Gate - Muta regioes abaixo do limiar
ENABLE_NOISE_GATE = True
NOISE_GATE_THRESHOLD = 0.008

# Filtros passa-baixas e passa-altos
ENABLE_CHEBY2 = True
CHEBY2_ORDER = 8
CHEBY2_RS_DB = 100
CHEBY2_CUTOFF_HZ = 7200

ENABLE_HPF = True
HPF_CUTOFF_HZ = 300

# Processamento final
OUTPUT_GAIN_DB = -0.5


# =========================
# FUNCOES DE PROCESSAMENTO
# =========================

def to_float(x: np.ndarray) -> np.ndarray:
    """Normaliza audio para formato float32 no intervalo [-1, 1]."""
    if x.dtype == np.int16:
        return x.astype(np.float32) / 32768.0
    if x.dtype == np.int32:
        return x.astype(np.float32) / 2147483648.0
    x = x.astype(np.float32)
    peak = np.max(np.abs(x)) + 1e-12
    if peak > 1.5:
        x = x / peak
    return x

def to_int16(x: np.ndarray) -> np.ndarray:
    """Converte audio de float32 para int16."""
    x = np.clip(x, -1.0, 1.0)
    return (x * 32767.0).astype(np.int16)

def apply_gain_db(x: np.ndarray, gain_db: float) -> np.ndarray:
    """Aplica ganho em decibeis ao sinal."""
    return x * (10.0 ** (gain_db / 20.0))

def denoise_wiener_spectral(x: np.ndarray, fs: int) -> np.ndarray:
    """
    Aplica Wiener Filter no dominio STFT para reducao de ruido.
    
    O filtro Wiener eh otimo para sinais Gaussianos, calculando:
    G(f,t) = SNR(f,t) / (SNR(f,t) + 1)
    
    onde SNR = P(f,t) / (alpha * noise_psd(f))
    """
    f, t, Z = signal.stft(x, fs=fs, window=WINDOW, nperseg=N_FFT, 
                          noverlap=N_FFT - HOP, boundary=None, padded=False)
    P = np.abs(Z) ** 2
    
    # Estima ruido a partir dos frames de menor energia
    frame_energy = np.mean(P, axis=0)
    n_frames = frame_energy.size
    n_noise = max(1, int(NOISE_PERCENT * n_frames))
    
    idx_silence = np.argsort(frame_energy)[:n_noise]
    noise_psd = np.mean(P[:, idx_silence], axis=1) + 1e-25
    
    # Calcula ganho Wiener
    snr = P / (WIENER_ALPHA * noise_psd[:, None])
    G = snr / (snr + 1.0)
    G = np.clip(G, WIENER_FLOOR, 1.0)
    
    # Spectral Gating: atenua componentes abaixo do limiar relativo
    if ENABLE_SPECTRAL_GATE:
        frame_max = np.max(P, axis=0)
        gate = (P / (frame_max + 1e-20)) > GATE_THRESHOLD
        G = G * gate.astype(float)
    
    Z_filtered = Z * G
    
    _, x_filtered = signal.istft(Z_filtered, fs=fs, window=WINDOW, nperseg=N_FFT,
                                  noverlap=N_FFT - HOP, boundary=None)
    
    if len(x_filtered) > len(x):
        x_filtered = x_filtered[:len(x)]
    elif len(x_filtered) < len(x):
        x_filtered = np.pad(x_filtered, (0, len(x) - len(x_filtered)))
    
    return x_filtered.astype(np.float32)

def apply_cheby2_filter(x: np.ndarray, fs: int, order: int, rs_db: int, cutoff_hz: float) -> np.ndarray:
    """
    Aplica filtro Chebyshev Tipo II passa-baixas.
    
    O filtro Chebyshev II apresenta ripl no stopband em troca de
    transicao mais abrupta na banda de passagem.
    """
    nyquist = fs / 2.0
    cutoff_normalized = min(cutoff_hz / nyquist, 0.99)
    
    sos = signal.cheby2(N=order, rs=rs_db, Wn=cutoff_normalized, 
                        btype="lowpass", output="sos")
    
    return signal.sosfiltfilt(sos, x, axis=0)

def apply_hpf_filter(x: np.ndarray, fs: int, cutoff_hz: float) -> np.ndarray:
    """
    Aplica filtro Butterworth passa-altos para remocao de ruido grave.
    
    Butterworth oferece resposta maximamente plana, sem ripple.
    """
    nyquist = fs / 2.0
    cutoff_normalized = max(cutoff_hz / nyquist, 0.01)
    
    sos = signal.butter(N=4, Wn=cutoff_normalized, btype="highpass", output="sos")
    
    return signal.sosfiltfilt(sos, x, axis=0)

def apply_noise_gate(x: np.ndarray, fs: int, threshold: float) -> np.ndarray:
    """
    Noise gate no dominio do tempo: atenua completamente regioes muito fracas.
    
    Calcula RMS por blocos de 10ms e zera sinais abaixo do limiar.
    """
    block_size = int(fs * 0.01)
    n_blocks = len(x) // block_size
    
    rms_blocks = np.array([np.sqrt(np.mean(x[i*block_size:(i+1)*block_size]**2)) 
                           for i in range(n_blocks)])
    
    gate_mask = np.repeat(rms_blocks < threshold, block_size)
    gate_mask = np.concatenate([gate_mask, np.zeros(len(x) - len(gate_mask), dtype=bool)])
    
    x_gated = x.copy()
    x_gated[gate_mask] = 0
    
    return x_gated


# =========================
# PIPELINE PRINCIPAL
# =========================

print("="*70)
print("PROCESSAMENTO DE AUDIO - REDUCAO DE RUIDO COM FILTRO CHEBYSHEV")
print("="*70)

# Carrega arquivo de audio
print("\n[1] Carregando arquivo de audio...")
try:
    fs, x = wavfile.read(INPUT_WAV)
    print("    Taxa de amostragem: {} Hz".format(fs))
    print("    Formato: {}".format(x.dtype))
    print("    Duração: {:.2f} segundos ({} amostras)".format(len(x)/fs, x.shape))
except FileNotFoundError:
    print("    ERRO: Arquivo '{}' nao encontrado.".format(INPUT_WAV))
    exit(1)

x_f = to_float(x)
if x_f.ndim == 1:
    x_f = x_f[:, None]

input_peak = float(np.max(np.abs(x_f)) + 1e-12)

# Aplica Wiener Filter
print("\n[2] Aplicando Wiener Filter com STFT...")
y = np.zeros_like(x_f, dtype=np.float32)

for ch in range(x_f.shape[1]):
    print("    Processando canal {}...".format(ch+1))
    y[:, ch] = denoise_wiener_spectral(x_f[:, ch], fs)

# Aplica Chebyshev II passa-baixas
if ENABLE_CHEBY2:
    print("\n[3] Aplicando Chebyshev II passa-baixas...")
    print("    Ordem: {}, Atenuacao: {} dB, Corte: {} Hz".format(
        CHEBY2_ORDER, CHEBY2_RS_DB, CHEBY2_CUTOFF_HZ))
    y = apply_cheby2_filter(y, fs, CHEBY2_ORDER, CHEBY2_RS_DB, CHEBY2_CUTOFF_HZ)

# Aplica passa-altos
if ENABLE_HPF:
    print("\n[4] Aplicando passa-altos...")
    print("    Butterworth Ordem 4, Corte: {} Hz".format(HPF_CUTOFF_HZ))
    y = apply_hpf_filter(y, fs, HPF_CUTOFF_HZ)

# Aplica noise gate
if ENABLE_NOISE_GATE:
    print("\n[5] Aplicando noise gate...")
    for ch in range(y.shape[1]):
        y[:, ch] = apply_noise_gate(y[:, ch], fs, NOISE_GATE_THRESHOLD)

# Aplica ganho final
print("\n[6] Aplicando normalizacao...")
y = apply_gain_db(y, OUTPUT_GAIN_DB)

# Limita para nao exceder pico da entrada
y_peak = float(np.max(np.abs(y)) + 1e-12)
if y_peak > input_peak:
    g = input_peak / y_peak
    y = y * g

y = np.clip(y, -1.0, 1.0)

# Salva arquivo processado
print("\n[7] Salvando arquivo...")
if x.ndim == 1:
    y_out = y[:, 0]
else:
    y_out = y

wavfile.write(OUTPUT_WAV, fs, to_int16(y_out))
print("    Arquivo salvo: {}".format(OUTPUT_WAV))

# Resumo final
print("\n" + "="*70)
print("PROCESSAMENTO CONCLUIDO")
print("="*70)
print("\nArquivo de saida: {}".format(OUTPUT_WAV))
print("\nParametros utilizados:")
print("  Wiener Alpha (agressividade):      {}".format(WIENER_ALPHA))
print("  Wiener Floor (piso minimo):        {}".format(WIENER_FLOOR))
print("  Chebyshev ordem:                   {}".format(CHEBY2_ORDER))
print("  Chebyshev frequencia corte:        {} Hz".format(CHEBY2_CUTOFF_HZ))
print("  Butterworth HPF corte:             {} Hz".format(HPF_CUTOFF_HZ))
print("  Noise Gate limiar:                 {}".format(NOISE_GATE_THRESHOLD))
print("  Ganho saida:                       {} dB".format(OUTPUT_GAIN_DB))
print("\n" + "="*70)
