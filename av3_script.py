"""
Projeto de Filtro - Grupo C
Sistemas Lineares - Avaliacao 3
Processamento de audio com filtro passa-altas (remove chiado de baixa frequencia)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile

# ====================
# 1. CARREGAR O AUDIO
# ====================
print("Carregando o audio...")
samplerate, audio_data = wavfile.read('Arquivo3.wav')

print(f"Formato original: {audio_data.dtype}")
print(f"Shape: {audio_data.shape}")

# Se o audio for estereo, converte para mono
if len(audio_data.shape) > 1:
    audio_data = np.mean(audio_data, axis=1)
    print("Audio convertido de estereo para mono")

# Converter para float e normalizar entre -1 e 1
if audio_data.dtype == np.int16:
    audio_normalized = audio_data.astype(np.float32) / 32768.0
elif audio_data.dtype == np.int32:
    audio_normalized = audio_data.astype(np.float32) / 2147483648.0
else:
    audio_normalized = audio_data.astype(np.float32)

print(f"Taxa de amostragem: {samplerate} Hz")
print(f"Duracao: {len(audio_normalized)/samplerate:.2f} segundos")
print(f"Numero de amostras: {len(audio_normalized)}")

# ====================
# 2. PROJETAR O FILTRO FIR PASSA-ALTAS
# ====================
print("\nProjetando o filtro FIR Passa-Altas...")

# Ajuste AQUI a frequencia de corte:
# - 80 a 150 Hz: remove rumble/vento
# - 150 a 300 Hz: remove "grave embolado"
# - 300+ Hz: já começa a afinar voz/música
freq_corte = 300  # Hz  (troque conforme seu audio)
numtaps = 801      # ímpar é melhor para highpass FIR (linear-phase)

print(f"Filtro projetado:")
print(f"  - Tipo: FIR (linear-phase)")
print(f"  - Classe: PASSA-ALTAS (remove chiado de baixa frequencia)")
print(f"  - Num taps: {numtaps}")
print(f"  - Frequencia de corte: {freq_corte} Hz")

# Kernel FIR highpass
fir_kernel = signal.firwin(
    numtaps,
    freq_corte,
    fs=samplerate,
    window='hamming',
    pass_zero=False  # <<< HIGH-PASS
)

# ====================
# 3. APLICAR O FILTRO NO AUDIO
# ====================
print("\nFiltrando o audio com FIR passa-altas...")

# Use filtfilt para evitar atraso (zero-phase)
audio_filtrado = signal.filtfilt(fir_kernel, [1.0], audio_normalized)

# Proteção: evitar clip ao salvar
audio_filtrado = np.clip(audio_filtrado, -1.0, 1.0)

print(f"Audio filtrado - Min: {np.min(audio_filtrado):.4f}, Max: {np.max(audio_filtrado):.4f}")

# ====================
# 4. SALVAR O AUDIO FILTRADO
# ====================
print("\nSalvando o audio filtrado...")

audio_filtrado_int16 = np.int16(audio_filtrado * 32767)
wavfile.write('Arquivo3_filtrado_passa_alta.wav', samplerate, audio_filtrado_int16)
print("Audio filtrado salvo como: Arquivo3_filtrado_passa_alta.wav")

# ====================
# 5. GERAR GRAFICOS
# ====================
print("\nGerando graficos...")

w, h = signal.freqz(fir_kernel, [1.0], worN=8000, fs=samplerate)

freqs_original = np.fft.rfftfreq(len(audio_normalized), 1/samplerate)
fft_original = np.abs(np.fft.rfft(audio_normalized))

freqs_filtrado = np.fft.rfftfreq(len(audio_filtrado), 1/samplerate)
fft_filtrado = np.abs(np.fft.rfft(audio_filtrado))

fig, axs = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Analise do Filtro FIR - Grupo C (Passa-Altas)', fontsize=16, fontweight='bold')

# GRAFICO 1: Resposta em Frequencia do Filtro
axs[0, 0].plot(w, 20 * np.log10(np.maximum(np.abs(h), 1e-12)), linewidth=2)
axs[0, 0].set_title('Resposta em Frequencia do Filtro (Passa-Altas)')
axs[0, 0].set_xlabel('Frequencia (Hz)')
axs[0, 0].set_ylabel('Magnitude (dB)')
axs[0, 0].grid(True, alpha=0.3)
axs[0, 0].axvline(freq_corte, linestyle='--', label=f'Corte: {freq_corte} Hz')
axs[0, 0].legend()
axs[0, 0].set_xlim([0, 20000])

# GRAFICO 2: Espectro de Frequencia - Original vs Filtrado
axs[0, 1].plot(freqs_original, 20*np.log10(fft_original + 1e-10), alpha=0.7, label='Original', linewidth=1)
axs[0, 1].plot(freqs_filtrado, 20*np.log10(fft_filtrado + 1e-10), alpha=0.7, label='Filtrado (passa-altas)', linewidth=1)
axs[0, 1].set_title('Espectro de Frequencia: Original vs Filtrado')
axs[0, 1].set_xlabel('Frequencia (Hz)')
axs[0, 1].set_ylabel('Magnitude (dB)')
axs[0, 1].grid(True, alpha=0.3)
axs[0, 1].legend()
axs[0, 1].set_xlim([0, 20000])

# GRAFICO 3: Forma de Onda - Original
time = np.arange(len(audio_normalized)) / samplerate
N = int(0.05 * samplerate)
axs[1, 0].plot(time[:N], audio_normalized[:N], linewidth=0.5)
axs[1, 0].set_title('Forma de Onda - Audio Original (primeiros 50ms)')
axs[1, 0].set_xlabel('Tempo (s)')
axs[1, 0].set_ylabel('Amplitude')
axs[1, 0].grid(True, alpha=0.3)

# GRAFICO 4: Forma de Onda - Filtrado
axs[1, 1].plot(time[:N], audio_filtrado[:N], linewidth=0.5)
axs[1, 1].set_title('Forma de Onda - Audio Filtrado (primeiros 50ms)')
axs[1, 1].set_xlabel('Tempo (s)')
axs[1, 1].set_ylabel('Amplitude')
axs[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('analise_filtro_passa_alta.png', dpi=300, bbox_inches='tight')
print("Graficos salvos como: analise_filtro_passa_alta.png")
plt.show()

print("\n" + "="*50)
print("PROCESSAMENTO CONCLUIDO!")
print("="*50)
print("\nArquivos gerados:")
print("  1. Arquivo3_filtrado_passa_alta.wav - Audio filtrado (passa-altas)")
print("  2. analise_filtro_passa_alta.png - Graficos de analise")
print("\nO filtro FIR passa-altas atenuou as baixas frequencias (chiado/rumble).")
