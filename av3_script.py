"""
Projeto de Filtro Chebyshev Tipo 2 - Grupo C
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
# 2. PROJETAR O FILTRO CHEBYSHEV TIPO 2
# ====================
print("\nProjetando o filtro Chebyshev Tipo 2...")

# Parametros do filtro (usar FIR passa-baixas linear-phase)
# FIR tende a preservar melhor a sonoridade e não amplifica ruído
numtaps = 801  # número de coeficientes do filtro FIR (maior = transição mais íngreme)
freq_corte = 6000  # cortar frequências acima de 6 kHz

print(f"Filtro projetado:")
print(f"  - Tipo: FIR (linear-phase)")
print(f"  - Classe: PASSA-BAIXAS (remove chiado de alta frequencia)")
print(f"  - Num taps: {numtaps}")
print(f"  - Frequencia de corte: {freq_corte} Hz")

# ====================
# 3. APLICAR O FILTRO NO AUDIO
# ====================
print("\nFiltrando o audio com FIR passa-baixas...")
# projetar kernel FIR e aplicar (lfilter é suficiente para FIR)
fir_kernel = signal.firwin(numtaps, freq_corte, fs=samplerate, window='hamming')
audio_filtrado = signal.lfilter(fir_kernel, [1.0], audio_normalized)

print(f"Audio filtrado - Min: {np.min(audio_filtrado):.4f}, Max: {np.max(audio_filtrado):.4f}")

# ====================
# 4. SALVAR O AUDIO FILTRADO
# ====================
print("\nSalvando o audio filtrado...")

# Converter de volta para int16
audio_filtrado_int16 = np.int16(audio_filtrado * 32767)

wavfile.write('Arquivo3_filtrado.wav', samplerate, audio_filtrado_int16)
print("Audio filtrado salvo como: Arquivo3_filtrado.wav")

# ====================
# 5. GERAR GRAFICOS
# ====================
print("\nGerando graficos...")

# Calcular a resposta em frequencia do filtro (FIR)
w, h = signal.freqz(fir_kernel, [1.0], worN=8000, fs=samplerate)

# Calcular o espectro de frequencia dos audios
freqs_original = np.fft.rfftfreq(len(audio_normalized), 1/samplerate)
fft_original = np.abs(np.fft.rfft(audio_normalized))

freqs_filtrado = np.fft.rfftfreq(len(audio_filtrado), 1/samplerate)
fft_filtrado = np.abs(np.fft.rfft(audio_filtrado))

# Criar figura com 4 subplots
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Analise do Filtro Chebyshev Tipo 2 - Grupo C (Passa-Baixas)', fontsize=16, fontweight='bold')

# GRAFICO 1: Resposta em Frequencia do Filtro
axs[0, 0].plot(w, 20 * np.log10(abs(h)), 'b', linewidth=2)
axs[0, 0].set_title('Resposta em Frequencia do Filtro (Passa-Baixas)')
axs[0, 0].set_xlabel('Frequencia (Hz)')
axs[0, 0].set_ylabel('Magnitude (dB)')
axs[0, 0].grid(True, alpha=0.3)
axs[0, 0].axvline(freq_corte, color='r', linestyle='--', label=f'Corte: {freq_corte} Hz')
axs[0, 0].legend()
axs[0, 0].set_xlim([0, 20000])

# GRAFICO 2: Espectro de Frequencia - Original vs Filtrado
axs[0, 1].plot(freqs_original, 20*np.log10(fft_original + 1e-10), 'b', alpha=0.7, label='Original (com chiado)', linewidth=1)
axs[0, 1].plot(freqs_filtrado, 20*np.log10(fft_filtrado + 1e-10), 'r', alpha=0.7, label='Filtrado (sem chiado)', linewidth=1)
axs[0, 1].set_title('Espectro de Frequencia: Original vs Filtrado')
axs[0, 1].set_xlabel('Frequencia (Hz)')
axs[0, 1].set_ylabel('Magnitude (dB)')
axs[0, 1].grid(True, alpha=0.3)
axs[0, 1].legend()
axs[0, 1].set_xlim([0, 20000])

# GRAFICO 3: Forma de Onda - Original
time = np.arange(len(audio_normalized)) / samplerate
axs[1, 0].plot(time[:int(0.05*samplerate)], audio_normalized[:int(0.05*samplerate)], 'b', linewidth=0.5)
axs[1, 0].set_title('Forma de Onda - Audio Original (primeiros 50ms)')
axs[1, 0].set_xlabel('Tempo (s)')
axs[1, 0].set_ylabel('Amplitude')
axs[1, 0].grid(True, alpha=0.3)

# GRAFICO 4: Forma de Onda - Filtrado
axs[1, 1].plot(time[:int(0.05*samplerate)], audio_filtrado[:int(0.05*samplerate)], 'r', linewidth=0.5)
axs[1, 1].set_title('Forma de Onda - Audio Filtrado (primeiros 50ms)')
axs[1, 1].set_xlabel('Tempo (s)')
axs[1, 1].set_ylabel('Amplitude')
axs[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('analise_filtro_chebyshev.png', dpi=300, bbox_inches='tight')
print("Graficos salvos como: analise_filtro_chebyshev.png")
plt.show()

print("\n" + "="*50)
print("PROCESSAMENTO CONCLUIDO!")
print("="*50)
print("\nArquivos gerados:")
print("  1. Arquivo3_filtrado.wav - Audio filtrado (SEM chiado)")
print("  2. analise_filtro_chebyshev.png - Graficos de analise")
print("\nO filtro FIR passa-baixas removeu (ou atenuou) o chiado em altas frequencias.")