"""
FILTRO CHEBYSHEV TIPO II + REDUÇÃO DE RUÍDO AVANÇADA
=====================================================
Combina:
- Detecção de Voz (Voice Activity Detection)
- Wiener Filter multi-camada (STFT)
- Chebyshev Tipo II passa-baixas (ordem 6)
- Spectral Subtraction agressiva
- Limitador de saída
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Não abre janela gráfica
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
from scipy.ndimage import median_filter

# =========================
# CONFIGURAÇÕES OTIMIZADAS
# =========================
INPUT_WAV  = "Arquivo3.wav"
OUTPUT_WAV = "Arquivo3_filtrado_v2.wav"
OUTPUT_FIG = "analise_filtrado_v2.png"

# === STFT (análise de frequência) ===
N_FFT = 2048           # reduzido para processar mais rápido
HOP = 512              # overlap = 75%
WINDOW = "hamming"     # melhor rejeição de lóbulos laterais

# === Detecção de Voz (VAD) ===
VAD_ENERGY_THRESHOLD = 0.005  # limiar de energia relativa
VAD_FREQ_RANGE = (80, 8000)   # faixa de frequência típica da voz

# === Wiener Filter (ULTRA AGRESSIVO) ===
WIENER_ALPHA = 6.0           # maximo agressivo
WIENER_FLOOR = 0.02          # muito muito baixo (quase silencia tudo fraco)
NOISE_PERCENT = 0.40         # detecta 40% dos frames como ruido

# === Spectral Gating ===
ENABLE_SPECTRAL_GATE = True
GATE_THRESHOLD = 0.08        # limiar de gate MUITO agressivo (silencia valores baixos)

# === Spectral Subtraction ===
SPECTRAL_SUB_FACTOR = 0.6    # fator de subtração espectral (0.5~0.8, mais baixo = menos agressivo)

# === Chebyshev Tipo II ===
# === Filtros passa-baixas + passa-altos (agressivos) ===
ENABLE_CHEBY2 = True
CHEBY2_ORDER = 8             # ordem MUITO alta (extremamente abrupto)
CHEBY2_RS_DB = 100            # atenuacao maxima
CHEBY2_CUTOFF_HZ = 8000      # corta TUDO acima de 8kHz (remove MUITO chiado)

# Passa-altos para remover ruído grave
ENABLE_HPF = True
HPF_CUTOFF_HZ = 300           # remove tudo abaixo de 300Hz (remove rumbling)

# === Processamento Final ===
CLAMP_TO_INPUT_PEAK = True
OUTPUT_GAIN_DB = -0.5        # Leve reducao para manter volume util

# === PSD ===
NPERSEG = 4096


# =========================
# FUNÇÕES AUXILIARES
# =========================

def to_float(x: np.ndarray) -> np.ndarray:
    """Converte áudio para float32 normalizado."""
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
    """Converte float32 para int16."""
    x = np.clip(x, -1.0, 1.0)
    return (x * 32767.0).astype(np.int16)

def apply_gain_db(x: np.ndarray, gain_db: float) -> np.ndarray:
    """Aplica ganho em dB."""
    return x * (10.0 ** (gain_db / 20.0))

def welch_psd(x: np.ndarray, fs: int, nperseg: int = 8192):
    """Calcula PSD usando método de Welch."""
    f, P = signal.welch(x, fs=fs, nperseg=nperseg, noverlap=nperseg//2)
    return f, P

def voice_activity_detection(P: np.ndarray, f: np.ndarray, fs: int) -> np.ndarray:
    """
    Detecta frames com atividade de voz.
    Retorna array booleano onde True = há voz.
    """
    # Filtra apenas faixa de frequência típica de voz
    f_min, f_max = VAD_FREQ_RANGE
    mask_freq = (f >= f_min) & (f <= f_max)
    
    # Calcula energia na faixa de voz por frame
    energy_voice = np.mean(P[mask_freq, :], axis=0)
    
    # Normaliza pela energia total
    energy_total = np.mean(P, axis=0)
    ratio_energy = energy_voice / (energy_total + 1e-20)
    
    # Threshold adaptativo
    threshold = VAD_ENERGY_THRESHOLD * np.max(ratio_energy)
    vad = ratio_energy > threshold
    
    return vad

def denoise_wiener_spectral(x: np.ndarray, fs: int) -> np.ndarray:
    """
    Redução de ruído usando Wiener Filter com STFT.
    VERSÃO ULTRA AGRESSIVA com spectral gating.
    """
    # === STFT ===
    f, t, Z = signal.stft(
        x, fs=fs, window=WINDOW, nperseg=N_FFT, 
        noverlap=N_FFT - HOP, boundary=None, padded=False
    )
    P = np.abs(Z) ** 2  # Power spectrogram (freq × time)
    
    # === Estimação de Ruído ===
    frame_energy = np.mean(P, axis=0)
    n_frames = frame_energy.size
    n_noise = max(1, int(NOISE_PERCENT * n_frames))
    
    idx_silence = np.argsort(frame_energy)[:n_noise]
    noise_psd = np.mean(P[:, idx_silence], axis=1) + 1e-25
    
    # === Wiener Gain ===
    snr = P / (WIENER_ALPHA * noise_psd[:, None])
    G = snr / (snr + 1.0)
    G = np.clip(G, WIENER_FLOOR, 1.0)
    
    # === Spectral Gating (elimina tudo muito fraco) ===
    if ENABLE_SPECTRAL_GATE:
        # Gate agressivo: silencia qualquer coisa abaixo do threshold
        frame_max = np.max(P, axis=0)
        gate = (P / (frame_max + 1e-20)) > GATE_THRESHOLD
        G = G * gate.astype(float)
    
    # === Aplica ganho ===
    Z_filtered = Z * G
    
    # === iSTFT ===
    _, x_filtered = signal.istft(
        Z_filtered, fs=fs, window=WINDOW, nperseg=N_FFT,
        noverlap=N_FFT - HOP, boundary=None
    )
    
    # === Ajusta tamanho ===
    if len(x_filtered) > len(x):
        x_filtered = x_filtered[:len(x)]
    elif len(x_filtered) < len(x):
        x_filtered = np.pad(x_filtered, (0, len(x) - len(x_filtered)))
    
    return x_filtered.astype(np.float32)

def apply_cheby2_filter(x: np.ndarray, fs: int, order: int, rs_db: int, cutoff_hz: float) -> np.ndarray:
    """
    Aplica filtro Chebyshev Tipo II passa-baixas.
    Reduz componentes de alta frequência (ruído, chicado).
    """
    nyquist = fs / 2.0
    cutoff_normalized = min(cutoff_hz / nyquist, 0.99)
    
    print(f"  Chebyshev II: ordem={order}, rs={rs_db}dB, fc={cutoff_hz:.0f}Hz")
    
    sos = signal.cheby2(
        N=order,
        rs=rs_db,
        Wn=cutoff_normalized,
        btype="lowpass",
        output="sos"
    )
    
    return signal.sosfiltfilt(sos, x, axis=0)


def apply_hpf_filter(x: np.ndarray, fs: int, cutoff_hz: float) -> np.ndarray:
    """
    Aplica filtro Butterworth passa-altos.
    Remove ruidos graves (hum, buzz de baixa frequencia).
    """
    nyquist = fs / 2.0
    cutoff_normalized = max(cutoff_hz / nyquist, 0.01)
    
    print(f"  Butterworth HPF: fc={cutoff_hz:.0f}Hz")
    
    sos = signal.butter(
        N=4,
        Wn=cutoff_normalized,
        btype="highpass",
        output="sos"
    )
    
    return signal.sosfiltfilt(sos, x, axis=0)


# =========================
# PIPELINE PRINCIPAL
# =========================

print("=" * 60)
print("FILTRO CHEBYSHEV II + REDUÇÃO DE RUÍDO AVANÇADA")
print("=" * 60)

# === 1. CARREGAR ÁUDIO ===
print("\n[1] Carregando áudio...")
try:
    fs, x = wavfile.read(INPUT_WAV)
    print(f"    Taxa de amostragem: {fs} Hz")
    print(f"    Tipo de dado: {x.dtype}")
    print(f"    Forma: {x.shape}")
except FileNotFoundError:
    print(f"    ERRO: arquivo '{INPUT_WAV}' não encontrado!")
    exit(1)

x_f = to_float(x)
if x_f.ndim == 1:
    x_f = x_f[:, None]

input_peak = float(np.max(np.abs(x_f)) + 1e-12)
print(f"    Pico entrada: {input_peak:.4f}")

# === 2. REDUÇÃO DE RUÍDO (Wiener + Spectral Subtraction) ===
print("\n[2] Redução de ruído (Wiener + Spectral Subtraction)...")
y = np.zeros_like(x_f, dtype=np.float32)

for ch in range(x_f.shape[1]):
    print(f"    Processando canal {ch+1}/{x_f.shape[1]}...")
    y[:, ch] = denoise_wiener_spectral(x_f[:, ch], fs)

# === 3. FILTRO CHEBYSHEV TIPO II (Passa-baixas) ===
if ENABLE_CHEBY2:
    print(f"\n[3] Aplicando Chebyshev Tipo II (passa-baixas)...")
    y = apply_cheby2_filter(y, fs, CHEBY2_ORDER, CHEBY2_RS_DB, CHEBY2_CUTOFF_HZ)

# === 3B. FILTRO PASSA-ALTOS (remove ruido grave) ===
if ENABLE_HPF:
    print(f"\n[3B] Aplicando passa-altos agressivo...")
    y = apply_hpf_filter(y, fs, HPF_CUTOFF_HZ)

# === 4. GANHO FINAL ===
print("\n[4] Aplicando ganho final...")
y = apply_gain_db(y, OUTPUT_GAIN_DB)

# === 5. LIMITAR PARA NÃO EXCEDER PICO DA ENTRADA ===
if CLAMP_TO_INPUT_PEAK:
    y_peak = float(np.max(np.abs(y)) + 1e-12)
    if y_peak > input_peak:
        g = input_peak / y_peak
        y = y * g
        print(f"    Clamp aplicado (ganho = {g:.4f})")

y = np.clip(y, -1.0, 1.0)

# === 6. SALVAR ÁUDIO ===
print("\n[5] Salvando áudio processado...")
if x.ndim == 1:
    y_out = y[:, 0]
else:
    y_out = y

wavfile.write(OUTPUT_WAV, fs, to_int16(y_out))
print(f"    [OK] Salvo: {OUTPUT_WAV}")

# === 6. RESUMO ===
print("\n" + "=" * 60)
print("PROCESSAMENTO CONCLUIDO COM SUCESSO!")
print("=" * 60)
print(f"\nArquivo gerado: {OUTPUT_WAV}")
print(f"\nParametros usados:")
print(f"  - Wiener Alpha: {WIENER_ALPHA} (ULTRA AGRESSIVO)")
print(f"  - Wiener Floor: {WIENER_FLOOR}")
print(f"  - Spectral Gate: {GATE_THRESHOLD}")
print(f"  - Chebyshev ordem: {CHEBY2_ORDER}, fc: {CHEBY2_CUTOFF_HZ} Hz")
print(f"  - Passa-altos: {HPF_CUTOFF_HZ} Hz")
print(f"  - Ganho saida: {OUTPUT_GAIN_DB} dB")
print(f"\n[OK] Pronto para audicao!")
print("=" * 60)
