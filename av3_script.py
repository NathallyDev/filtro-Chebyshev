"""
Projeto de Filtro Chebyshev Tipo 2 - Grupo C
Sistemas Lineares - Avaliacao 3
Processamento de audio com filtro passa-baixas
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import soundfile as sf

# ====================
# 1. CARREGAR O AUDIO
# ====================
print("Carregando o audio...")
samplerate, audio_data = wavfile.read('Arquivo3.wav')

# Se o audio for estereo, converte para mono
if len(audio_data.shape) > 1:
    audio_data = np.mean(audio_data, axis=1)

# Normalizar o audio
audio_data = audio_data.astype(float)
audio_max = np.max(np.abs(audio_data))
audio_normalized = audio_data / audio_max

print(f"Taxa de amostragem: {samplerate} Hz")
print(f"Duracao: {len(audio_normalized)/samplerate:.2f} segundos")
print(f"Numero de amostras: {len(audio_normalized)}")

# ====================
# 2. PROJETAR O FILTRO CHEBYSHEV TIPO 2
# ====================
print("\nProjetando o filtro Chebyshev Tipo 2...")

# Parametros do filtro
ordem = 3  # Ordem do filtro (igual ao QUCS)
freq_corte = 5000  # Frequencia de corte em Hz
freq_nyquist = samplerate / 2  # Frequencia de Nyquist
freq_normalizada = freq_corte / freq_nyquist  # Normalizar (0 a 1)
atenuacao_stopband = 20  # Atenuacao na banda de rejeicao em dB

# Criar o filtro Chebyshev Tipo 2
# 'cheby2' = Chebyshev Tipo 2
# 'low' = passa-baixas
b, a = signal.cheby2(ordem, atenuacao_stopband, freq_normalizada, btype='low', analog=False)

print(f"Filtro projetado:")
print(f"  - Tipo: Chebyshev Tipo 2")
print(f"  - Ordem: {ordem}")
print(f"  - Frequencia de corte: {freq_corte} Hz")
print(f"  - Atenuacao stopband: {atenuacao_stopband} dB")

# ====================
# 3. APLICAR O FILTRO NO AUDIO
# ====================
print("\nFiltrando o audio...")
audio_filtrado = signal.filtfilt(b, a, audio_normalized)

# ====================
# 4. SALVAR O AUDIO FILTRADO
# ====================
print("\nSalvando o audio filtrado...")
# Desnormalizar
audio_filtrado_int = (audio_filtrado * audio_max).astype(audio_data.dtype)
wavfile.write('Arquivo3_filtrado.wav', samplerate, audio_filtrado_int)
print("Audio filtrado salvo como: Arquivo3_filtrado.wav")

# ====================
# 5. GERAR GRAFICOS
# ====================
print("\nGerando graficos...")

# Calcular a resposta em frequencia do filtro
w, h = signal.freqz(b, a, worN=8000, fs=samplerate)

# Calcular o espectro de frequencia dos audios
freqs_original = np.fft.rfftfreq(len(audio_normalized), 1/samplerate)
fft_original = np.abs(np.fft.rfft(audio_normalized))

freqs_filtrado = np.fft.rfftfreq(len(audio_filtrado), 1/samplerate)
fft_filtrado = np.abs(np.fft.rfft(audio_filtrado))

# Criar figura com 4 subplots
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Analise do Filtro Chebyshev Tipo 2 - Grupo C', fontsize=16, fontweight='bold')

# GRAFICO 1: Resposta em Frequencia do Filtro
axs[0, 0].plot(w, 20 * np.log10(abs(h)), 'b', linewidth=2)
axs[0, 0].set_title('Resposta em Frequencia do Filtro')
axs[0, 0].set_xlabel('Frequencia (Hz)')
axs[0, 0].set_ylabel('Magnitude (dB)')
axs[0, 0].grid(True, alpha=0.3)
axs[0, 0].axvline(freq_corte, color='r', linestyle='--', label=f'Corte: {freq_corte} Hz')
axs[0, 0].legend()
axs[0, 0].set_xlim([0, 20000])

# GRAFICO 2: Espectro de Frequencia - Original vs Filtrado
axs[0, 1].plot(freqs_original, 20*np.log10(fft_original + 1e-10), 'b', alpha=0.7, label='Original', linewidth=1)
axs[0, 1].plot(freqs_filtrado, 20*np.log10(fft_filtrado + 1e-10), 'r', alpha=0.7, label='Filtrado', linewidth=1)
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
print("  1. Arquivo3_filtrado.wav - Audio filtrado")
print("  2. analise_filtro_chebyshev.png - Graficos de analise")
print("\nO filtro Chebyshev Tipo 2 removeu frequencias acima de 5 kHz.")