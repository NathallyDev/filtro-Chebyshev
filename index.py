"""
================================================================================
PROCESSAMENTO DE ÁUDIO COM FILTRO CHEBYSHEV TIPO II E REDUÇÃO DE RUÍDO
================================================================================

Disciplina  : Sistemas Lineares
Grupo       : C
Técnica     : Chebyshev Tipo 2 (passa-baixas)
Arquivo     : Arquivo3.wav

Descrição
---------
Este script implementa um pipeline de processamento digital de sinais (DSP)
para redução de ruído em sinais de áudio. A técnica principal empregada é o
filtro IIR Chebyshev Tipo II (passa-baixas), caracterizado por apresentar
resposta maximamente plana na banda de passagem e ripple (ondulação) controlado
na banda de rejeição.

Pipeline de processamento:
    1. Wiener Filter (domínio STFT) — estimativa e subtração do ruído estacionário
    2. Chebyshev Tipo II (passa-baixas) — atenuação de altas frequências [TÉCNICA PRINCIPAL]
    3. Butterworth (passa-altos) — remoção de componentes de frequência muito baixa
    4. Noise Gate (domínio temporal) — silenciamento de blocos abaixo do limiar de energia

Fundamentação teórica
---------------------
O filtro Chebyshev Tipo II é definido pela função de transferência cujo módulo
ao quadrado é dado por:

    |H(jΩ)|² = 1 / [1 + ε² · Tₙ²(Ωₛ/Ω) / Tₙ²(Ωₛ/Ωc)]

onde:
    - Tₙ(x) é o polinômio de Chebyshev de ordem n
    - ε é o parâmetro que controla o ripple no stopband
    - Ωₛ é a frequência de início do stopband
    - Ωc é a frequência de corte

Diferente do Tipo I, o Chebyshev Tipo II possui:
    - Banda de passagem equiripple no STOPBAND (não no passband)
    - Resposta monotônica (sem ripple) na banda de passagem
    - Zeros de transmissão no eixo imaginário (atenuação infinita em certas frequências)

Dependências
------------
    numpy  >= 1.21
    scipy  >= 1.7
    matplotlib >= 3.4
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import signal
from scipy.io import wavfile


# ==============================================================================
# SEÇÃO 1 — PARÂMETROS DE CONFIGURAÇÃO
# ==============================================================================

# Arquivos de entrada e saída
INPUT_WAV  = "Arquivo3.wav"
OUTPUT_WAV = "Arquivo3_filtrado_v2.wav"
OUTPUT_FIG = "av3_analise.png"          # Figura de análise espectral
OUTPUT_FIG_RESP = "av3_resposta_freq.png"  # Figura da resposta em frequência do filtro

# --- Parâmetros da STFT (Short-Time Fourier Transform) ---
# N_FFT  : Tamanho da janela de análise (amostras). Quanto maior, melhor a
#           resolução em frequência, porém menor a resolução temporal.
# HOP    : Deslocamento entre janelas consecutivas (amostras).
# WINDOW : Tipo de janela de análise. A janela de Hamming reduz o vazamento
#           espectral (spectral leakage) com sidelobes de -43 dB.
N_FFT   = 2048
HOP     = 512
WINDOW  = "hamming"

# --- Parâmetros do Filtro de Wiener (STFT) ---
# WIENER_ALPHA  : Fator de superestimação do ruído. Valores maiores aumentam a
#                 agressividade da redução, podendo introduzir artefatos musicais.
# WIENER_FLOOR  : Ganho mínimo aplicado ao espectro, evitando supressão total
#                 do sinal (artefatos de "buraco" espectral).
# NOISE_PERCENT : Fração dos frames de menor energia usada para estimar o PSD
#                 (Power Spectral Density) do ruído de fundo.
WIENER_ALPHA   = 5.4
WIENER_FLOOR   = 0.027
NOISE_PERCENT  = 0.37

# --- Spectral Gate ---
# Máscara binária que zera componentes cuja energia relativa ao pico do frame
# é inferior ao limiar GATE_THRESHOLD. Suprime ruído residual de baixa energia.
ENABLE_SPECTRAL_GATE = True
GATE_THRESHOLD       = 0.055

# --- Noise Gate (domínio temporal) ---
# Silencia completamente blocos de 10 ms cujo RMS esteja abaixo do limiar.
# Eficaz para eliminar ruído de fundo em intervalos de silêncio.
ENABLE_NOISE_GATE       = True
NOISE_GATE_THRESHOLD    = 0.008

# --- Filtro Chebyshev Tipo II (passa-baixas) — TÉCNICA PRINCIPAL ---
# CHEBY2_ORDER    : Ordem do filtro. Quanto maior a ordem, mais íngreme a
#                   transição passband→stopband, porém maior o custo computacional
#                   e maior a distorção de fase de grupo.
# CHEBY2_RS_DB    : Atenuação mínima no stopband em dB. Define o ripple máximo
#                   permitido na banda de rejeição. Valor de 100 dB garante
#                   atenuação muito elevada das componentes indesejadas.
# CHEBY2_CUTOFF_HZ: Frequência de corte (início do stopband) em Hz. Valor de
#                   7200 Hz preserva conteúdo de voz e música (80 Hz – 7 kHz)
#                   enquanto atenua ruído de alta frequência.
ENABLE_CHEBY2    = True
CHEBY2_ORDER     = 8
CHEBY2_RS_DB     = 100
CHEBY2_CUTOFF_HZ = 7200

# --- Filtro Butterworth (passa-altos) ---
# Remove componentes de frequência muito baixa (ruído de hum, vibrações
# mecânicas). O filtro Butterworth é maximamente plano na passband (sem ripple).
# HPF_CUTOFF_HZ : Frequência de corte do passa-altos em Hz.
ENABLE_HPF    = True
HPF_CUTOFF_HZ = 300

# Ganho de saída aplicado após o processamento (em dB)
OUTPUT_GAIN_DB = -0.5


# ==============================================================================
# SEÇÃO 2 — FUNÇÕES AUXILIARES DE CONVERSÃO
# ==============================================================================

def to_float(x: np.ndarray) -> np.ndarray:
    """
    Normaliza o sinal de áudio para o formato ponto flutuante (float32)
    no intervalo [-1, 1].

    Parâmetros
    ----------
    x : np.ndarray
        Sinal de entrada em formato inteiro (int16 ou int32) ou float.

    Retorno
    -------
    np.ndarray
        Sinal normalizado em float32 no intervalo [-1, 1].

    Notas
    -----
    A normalização é necessária para compatibilidade com os algoritmos de
    processamento que assumem sinal de amplitude unitária.
    """
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
    """
    Converte o sinal de float32 normalizado para inteiro de 16 bits (PCM).

    Parâmetros
    ----------
    x : np.ndarray
        Sinal em float32 no intervalo [-1, 1].

    Retorno
    -------
    np.ndarray
        Sinal em int16 (formato WAV padrão).
    """
    x = np.clip(x, -1.0, 1.0)
    return (x * 32767.0).astype(np.int16)


def apply_gain_db(x: np.ndarray, gain_db: float) -> np.ndarray:
    """
    Aplica ganho linear equivalente a um valor em decibéis.

    A relação entre ganho linear G e ganho em dB é:
        G = 10^(gain_db / 20)

    Parâmetros
    ----------
    x       : np.ndarray — Sinal de entrada.
    gain_db : float      — Ganho em dB (positivo amplifica, negativo atenua).

    Retorno
    -------
    np.ndarray
        Sinal com ganho aplicado.
    """
    return x * (10.0 ** (gain_db / 20.0))


# ==============================================================================
# SEÇÃO 3 — FILTRO DE WIENER NO DOMÍNIO STFT
# ==============================================================================

def denoise_wiener_spectral(x: np.ndarray, fs: int) -> np.ndarray:
    """
    Aplica o Filtro de Wiener no domínio da STFT para redução de ruído.

    O filtro de Wiener espectral calcula um ganho tempo-frequência G(f, t)
    que minimiza o erro quadrático médio entre o sinal limpo estimado e o
    sinal ruidoso observado. Para cada bin de frequência f e frame t:

        G(f, t) = SNR(f, t) / [SNR(f, t) + 1]

    onde:
        SNR(f, t) = P(f, t) / [α · N(f)]
        P(f, t)   = |X(f, t)|²  — PSD do sinal observado
        N(f)      = estimativa do PSD do ruído (frames de menor energia)
        α         = WIENER_ALPHA (fator de superestimação)

    O ganho G ∈ [WIENER_FLOOR, 1], onde o piso evita supressão total.

    Parâmetros
    ----------
    x  : np.ndarray — Sinal mono em float32.
    fs : int        — Taxa de amostragem em Hz.

    Retorno
    -------
    np.ndarray
        Sinal com ruído estacionário reduzido.
    """
    # Calcula a STFT do sinal: X[frequência, tempo]
    f, t, Z = signal.stft(x, fs=fs, window=WINDOW, nperseg=N_FFT,
                          noverlap=N_FFT - HOP, boundary=None, padded=False)

    # Densidade espectral de potência (PSD) do sinal observado
    P = np.abs(Z) ** 2

    # Estimação do PSD do ruído: seleciona os frames de menor energia total,
    # assumindo que esses correspondem a intervalos de silêncio/ruído puro
    frame_energy = np.mean(P, axis=0)
    n_frames     = frame_energy.size
    n_noise      = max(1, int(NOISE_PERCENT * n_frames))
    idx_silence  = np.argsort(frame_energy)[:n_noise]
    noise_psd    = np.median(P[:, idx_silence], axis=1) + 1e-25

    # Subtração espectral suave para reduzir componente estacionária do ruído
    spectral_sub_factor = 0.5
    P_clean = np.maximum(P - spectral_sub_factor * noise_psd[:, None], 1e-25)

    # Razão sinal-ruído estimada para cada bin tempo-frequência (após subtração)
    snr = P_clean / (WIENER_ALPHA * noise_psd[:, None])

    # Ganho de Wiener inicial
    G = snr / (snr + 1.0)
    G = np.clip(G, WIENER_FLOOR, 1.0)

    # Spectral Gate: máscara suave (rampa) em vez de binária para evitar cortes
    if ENABLE_SPECTRAL_GATE:
        frame_max = np.max(P, axis=0)
        rel = P / (frame_max + 1e-20)
        gate_factor = np.clip((rel - GATE_THRESHOLD) / (1.0 - GATE_THRESHOLD), 0.0, 1.0)
        G = G * gate_factor

    # Suavização temporal do ganho por enquadramento para reduzir "musical noise"
    TF_SMOOTH_FRAMES = 7
    if TF_SMOOTH_FRAMES > 1:
        win = np.ones(TF_SMOOTH_FRAMES, dtype=float) / float(TF_SMOOTH_FRAMES)
        # operação por bin de frequência (eixo 0 -> frequência)
        G = np.array([np.convolve(G_i, win, mode="same") for G_i in G])

    # Aplica o ganho no domínio espectral e reconstrói o sinal pelo ISTFT
    Z_filtered = Z * G
    _, x_filtered = signal.istft(Z_filtered, fs=fs, window=WINDOW, nperseg=N_FFT,
                                 noverlap=N_FFT - HOP, boundary=None)

    # Ajusta comprimento para coincidir com o sinal original
    if len(x_filtered) > len(x):
        x_filtered = x_filtered[:len(x)]
    elif len(x_filtered) < len(x):
        x_filtered = np.pad(x_filtered, (0, len(x) - len(x_filtered)))

    return x_filtered.astype(np.float32)


# ==============================================================================
# SEÇÃO 4 — FILTRO CHEBYSHEV TIPO II (TÉCNICA PRINCIPAL)
# ==============================================================================

def apply_cheby2_filter(x: np.ndarray, fs: int,
                        order: int, rs_db: int, cutoff_hz: float) -> np.ndarray:
    """
    Aplica o filtro IIR Chebyshev Tipo II passa-baixas.

    O Chebyshev Tipo II (também chamado de Chebyshev Inverso) é projetado
    de forma a apresentar:
        - Banda de passagem MONOTÔNICA (sem ripple)
        - Banda de rejeição EQUIRIPPLE com atenuação mínima de rs_db decibéis
        - Zeros de transmissão no eixo jΩ (atenuação infinita em frequências
          específicas dentro do stopband)

    A implementação utiliza representação em Seções de Segunda Ordem (SOS),
    que oferece maior estabilidade numérica em comparação com coeficientes
    polinomiais (forma direta) para filtros de alta ordem.

    O método sosfiltfilt aplica o filtro em duas passagens (forward e backward),
    resultando em resposta de fase zero — essencial para preservar a forma de
    onda do sinal de áudio.

    Parâmetros
    ----------
    x         : np.ndarray — Sinal de entrada em float.
    fs        : int        — Taxa de amostragem em Hz.
    order     : int        — Ordem do filtro (número de polos).
    rs_db     : int        — Atenuação mínima no stopband em dB.
    cutoff_hz : float      — Frequência de início do stopband em Hz.

    Retorno
    -------
    np.ndarray
        Sinal filtrado.
    """
    nyquist            = fs / 2.0
    # Frequência normalizada: Wn ∈ (0, 1], onde 1 = frequência de Nyquist
    cutoff_normalized  = min(cutoff_hz / nyquist, 0.99)

    # Projeto do filtro Chebyshev Tipo II em formato SOS
    sos = signal.cheby2(N=order, rs=rs_db, Wn=cutoff_normalized,
                        btype="lowpass", output="sos")

    # Filtragem bidirecional (fase zero)
    return signal.sosfiltfilt(sos, x, axis=0)


# ==============================================================================
# SEÇÃO 5 — FILTRO BUTTERWORTH PASSA-ALTOS
# ==============================================================================

def apply_hpf_filter(x: np.ndarray, fs: int, cutoff_hz: float) -> np.ndarray:
    """
    Aplica filtro IIR Butterworth passa-altos de ordem 4.

    O filtro Butterworth é caracterizado por resposta em frequência
    maximamente plana na banda de passagem (sem ripple), descrita por:

        |H(jΩ)|² = 1 / [1 + (Ωc/Ω)^(2n)]

    A ausência de ripple torna o Butterworth adequado para filtragem
    de componentes de baixa frequência indesejados (hum de 60 Hz,
    vibrações mecânicas, ruído DC) sem distorção na banda de interesse.

    Parâmetros
    ----------
    x         : np.ndarray — Sinal de entrada em float.
    fs        : int        — Taxa de amostragem em Hz.
    cutoff_hz : float      — Frequência de corte em Hz.

    Retorno
    -------
    np.ndarray
        Sinal filtrado.
    """
    nyquist           = fs / 2.0
    cutoff_normalized = max(cutoff_hz / nyquist, 0.01)

    sos = signal.butter(N=4, Wn=cutoff_normalized, btype="highpass", output="sos")
    return signal.sosfiltfilt(sos, x, axis=0)


# ==============================================================================
# SEÇÃO 6 — NOISE GATE (DOMÍNIO TEMPORAL)
# ==============================================================================

def apply_noise_gate(x: np.ndarray, fs: int, threshold: float) -> np.ndarray:
    """
    Aplica Noise Gate no domínio temporal para silenciar regiões de baixa energia.

    O sinal é segmentado em blocos de 10 ms. O RMS (Root Mean Square) de cada
    bloco é calculado e comparado ao limiar. Blocos com RMS abaixo do limiar
    são zerados, eliminando ruído de fundo nos intervalos de silêncio.

    RMS de um bloco x[i] = sqrt( (1/N) · Σ x²[k] )

    Esta técnica é complementar ao Filtro de Wiener: enquanto o Wiener atua
    no domínio tempo-frequência, o Noise Gate atua diretamente no domínio
    temporal, sendo eficaz contra ruído não estacionário de curta duração.

    Parâmetros
    ----------
    x         : np.ndarray — Sinal de entrada em float.
    fs        : int        — Taxa de amostragem em Hz.
    threshold : float      — Limiar de RMS. Blocos abaixo são silenciados.

    Retorno
    -------
    np.ndarray
        Sinal com regiões silenciosas zeradas.
    """
    block_size = int(fs * 0.01)     # 10 ms por bloco
    n_blocks   = len(x) // block_size

    # Calcula RMS para cada bloco
    rms_blocks = np.array([
        np.sqrt(np.mean(x[i * block_size:(i + 1) * block_size] ** 2))
        for i in range(n_blocks)
    ])

    # Gera máscara booleana: True = silenciar o bloco
    gate_mask = np.repeat(rms_blocks < threshold, block_size)
    gate_mask = np.concatenate([gate_mask,
                                np.zeros(len(x) - len(gate_mask), dtype=bool)])

    x_gated          = x.copy()
    x_gated[gate_mask] = 0.0
    return x_gated


# ==============================================================================
# SEÇÃO 7 — FUNÇÕES DE VISUALIZAÇÃO E ANÁLISE ESPECTRAL
# ==============================================================================

def plot_frequency_response(fs: int, fig_path: str) -> None:
    """
    Gera e salva a figura da resposta em frequência do filtro Chebyshev Tipo II.

    Exibe:
        (a) Magnitude |H(f)| em dB em escala linear de frequência
        (b) Magnitude |H(f)| em dB em escala logarítmica de frequência
        (c) Fase do filtro em graus
        (d) Atraso de grupo (group delay) em amostras

    O atraso de grupo é definido como:
        τ(Ω) = -d∠H(Ω)/dΩ

    e representa o atraso de cada componente de frequência.
    Para filtros de fase linear, τ(Ω) é constante; para filtros IIR
    como o Chebyshev II, o atraso de grupo varia com a frequência
    (porém sosfiltfilt zera este efeito na prática).

    Parâmetros
    ----------
    fs       : int — Taxa de amostragem em Hz.
    fig_path : str — Caminho do arquivo PNG de saída.
    """
    nyquist           = fs / 2.0
    cutoff_normalized = min(CHEBY2_CUTOFF_HZ / nyquist, 0.99)

    # Projeto do filtro para análise
    sos = signal.cheby2(N=CHEBY2_ORDER, rs=CHEBY2_RS_DB,
                        Wn=cutoff_normalized, btype="lowpass", output="sos")

    # Converte SOS para coeficientes b, a para uso no freqz
    b_total = np.array([1.0])
    a_total = np.array([1.0])
    for section in sos:
        b_total = np.polymul(b_total, section[:3])
        a_total = np.polymul(a_total, section[3:])

    # Resposta em frequência: H(e^{jω})
    w, H = signal.freqz(b_total, a_total, worN=8192, fs=fs)
    mag_db    = 20 * np.log10(np.abs(H) + 1e-12)
    phase_deg = np.angle(H, deg=True)

    # Atraso de grupo (group delay)
    w_gd, gd = signal.group_delay((b_total, a_total), w=8192, fs=fs)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(
        f"Resposta em Frequência — Filtro Chebyshev Tipo II\n"
        f"Ordem {CHEBY2_ORDER}, Stopband {CHEBY2_RS_DB} dB, "
        f"Corte {CHEBY2_CUTOFF_HZ} Hz, fs = {fs} Hz",
        fontsize=13, fontweight="bold"
    )

    # (a) Magnitude — escala linear de frequência
    ax = axes[0, 0]
    ax.plot(w, mag_db, color="#1f77b4", linewidth=1.8)
    ax.axvline(CHEBY2_CUTOFF_HZ, color="red", linestyle="--", linewidth=1.2,
               label=f"Corte: {CHEBY2_CUTOFF_HZ} Hz")
    ax.axhline(-CHEBY2_RS_DB, color="orange", linestyle=":", linewidth=1.2,
               label=f"Stopband: −{CHEBY2_RS_DB} dB")
    ax.axhline(-3, color="green", linestyle=":", linewidth=1.0,
               label="−3 dB")
    ax.set_xlim(0, fs / 2)
    ax.set_ylim(-120, 5)
    ax.set_xlabel("Frequência (Hz)")
    ax.set_ylabel("Magnitude (dB)")
    ax.set_title("(a) Magnitude — Escala Linear")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.4)

    # (b) Magnitude — escala logarítmica de frequência
    ax = axes[0, 1]
    ax.semilogx(w + 1, mag_db, color="#1f77b4", linewidth=1.8)
    ax.axvline(CHEBY2_CUTOFF_HZ, color="red", linestyle="--", linewidth=1.2,
               label=f"Corte: {CHEBY2_CUTOFF_HZ} Hz")
    ax.axhline(-CHEBY2_RS_DB, color="orange", linestyle=":", linewidth=1.2,
               label=f"Stopband: −{CHEBY2_RS_DB} dB")
    ax.set_xlim(10, fs / 2)
    ax.set_ylim(-120, 5)
    ax.set_xlabel("Frequência (Hz) — log")
    ax.set_ylabel("Magnitude (dB)")
    ax.set_title("(b) Magnitude — Escala Logarítmica")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.4, which="both")

    # (c) Fase
    ax = axes[1, 0]
    ax.plot(w, phase_deg, color="#2ca02c", linewidth=1.5)
    ax.axvline(CHEBY2_CUTOFF_HZ, color="red", linestyle="--", linewidth=1.2,
               label=f"Corte: {CHEBY2_CUTOFF_HZ} Hz")
    ax.set_xlim(0, fs / 2)
    ax.set_xlabel("Frequência (Hz)")
    ax.set_ylabel("Fase (graus)")
    ax.set_title("(c) Resposta de Fase")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.4)

    # (d) Atraso de grupo
    ax = axes[1, 1]
    ax.plot(w_gd, np.clip(gd, -50, 500), color="#d62728", linewidth=1.5)
    ax.axvline(CHEBY2_CUTOFF_HZ, color="red", linestyle="--", linewidth=1.2,
               label=f"Corte: {CHEBY2_CUTOFF_HZ} Hz")
    ax.set_xlim(0, fs / 2)
    ax.set_xlabel("Frequência (Hz)")
    ax.set_ylabel("Atraso de Grupo (amostras)")
    ax.set_title("(d) Atraso de Grupo")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.4)

    plt.tight_layout()
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Figura salva: {fig_path}")


def plot_analysis(x_orig: np.ndarray, y_proc: np.ndarray,
                  fs: int, fig_path: str) -> None:
    """
    Gera e salva a figura de análise comparativa entre o sinal original
    e o sinal processado.

    Exibe seis painéis organizados em três linhas:
        Linha 1 — Formas de onda temporal (original vs. processado)
        Linha 2 — Espectrogramas (STFT) em escala logarítmica de potência
        Linha 3 — Densidades Espectrais de Potência (PSD) via método de Welch

    O método de Welch estima o PSD dividindo o sinal em segmentos com
    sobreposição, aplicando uma janela a cada segmento e calculando a
    média dos periodogramas resultantes, reduzindo a variância da estimativa.

    Parâmetros
    ----------
    x_orig  : np.ndarray — Sinal original (mono, float32).
    y_proc  : np.ndarray — Sinal processado (mono, float32).
    fs      : int        — Taxa de amostragem em Hz.
    fig_path: str        — Caminho do arquivo PNG de saída.
    """
    N    = len(x_orig)
    t    = np.linspace(0, N / fs, N)

    # Estimação do PSD pelo método de Welch
    f_w, psd_orig = signal.welch(x_orig, fs=fs, nperseg=N_FFT, noverlap=HOP)
    f_w, psd_proc = signal.welch(y_proc,  fs=fs, nperseg=N_FFT, noverlap=HOP)

    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(
        "Análise Comparativa — Sinal Original vs. Processado\n"
        "Grupo C | Filtro Chebyshev Tipo II | Arquivo3.wav",
        fontsize=13, fontweight="bold"
    )
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

    # ---- Linha 1: Formas de onda -------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(t, x_orig, color="#1f77b4", linewidth=0.4, alpha=0.85)
    ax1.set_title("(a) Sinal Original — Forma de Onda")
    ax1.set_xlabel("Tempo (s)")
    ax1.set_ylabel("Amplitude")
    ax1.set_xlim(t[0], t[-1])
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(t, y_proc, color="#d62728", linewidth=0.4, alpha=0.85)
    ax2.set_title("(b) Sinal Processado — Forma de Onda")
    ax2.set_xlabel("Tempo (s)")
    ax2.set_ylabel("Amplitude")
    ax2.set_xlim(t[0], t[-1])
    ax2.grid(True, alpha=0.3)

    # ---- Linha 2: Espectrogramas (STFT) ------------------------------------
    ax3 = fig.add_subplot(gs[1, 0])
    f_s, t_s, Zxx_orig = signal.stft(x_orig, fs=fs, window=WINDOW,
                                      nperseg=N_FFT, noverlap=N_FFT - HOP)
    spec_orig = 10 * np.log10(np.abs(Zxx_orig) ** 2 + 1e-12)
    im1 = ax3.pcolormesh(t_s, f_s, spec_orig, shading="auto",
                         cmap="inferno", vmin=-100, vmax=0)
    ax3.set_title("(c) Espectrograma Original (dB)")
    ax3.set_xlabel("Tempo (s)")
    ax3.set_ylabel("Frequência (Hz)")
    ax3.set_ylim(0, min(fs / 2, 12000))
    ax3.axhline(CHEBY2_CUTOFF_HZ, color="cyan", linewidth=1.2, linestyle="--",
                label=f"Corte Cheby2: {CHEBY2_CUTOFF_HZ} Hz")
    ax3.legend(fontsize=7, loc="upper right")
    plt.colorbar(im1, ax=ax3, label="Potência (dB)")

    ax4 = fig.add_subplot(gs[1, 1])
    f_s, t_s, Zxx_proc = signal.stft(y_proc, fs=fs, window=WINDOW,
                                      nperseg=N_FFT, noverlap=N_FFT - HOP)
    spec_proc = 10 * np.log10(np.abs(Zxx_proc) ** 2 + 1e-12)
    im2 = ax4.pcolormesh(t_s, f_s, spec_proc, shading="auto",
                         cmap="inferno", vmin=-100, vmax=0)
    ax4.set_title("(d) Espectrograma Processado (dB)")
    ax4.set_xlabel("Tempo (s)")
    ax4.set_ylabel("Frequência (Hz)")
    ax4.set_ylim(0, min(fs / 2, 12000))
    ax4.axhline(CHEBY2_CUTOFF_HZ, color="cyan", linewidth=1.2, linestyle="--",
                label=f"Corte Cheby2: {CHEBY2_CUTOFF_HZ} Hz")
    ax4.legend(fontsize=7, loc="upper right")
    plt.colorbar(im2, ax=ax4, label="Potência (dB)")

    # ---- Linha 3: PSD (Método de Welch) ------------------------------------
    ax5 = fig.add_subplot(gs[2, :])
    psd_orig_db = 10 * np.log10(psd_orig + 1e-25)
    psd_proc_db = 10 * np.log10(psd_proc  + 1e-25)
    ax5.plot(f_w, psd_orig_db, color="#1f77b4", linewidth=1.2,
             alpha=0.85, label="Original")
    ax5.plot(f_w, psd_proc_db, color="#d62728", linewidth=1.2,
             alpha=0.85, label="Processado")
    ax5.axvline(CHEBY2_CUTOFF_HZ, color="orange", linestyle="--", linewidth=1.3,
                label=f"Corte Cheby2: {CHEBY2_CUTOFF_HZ} Hz")
    ax5.axvline(HPF_CUTOFF_HZ, color="green", linestyle=":", linewidth=1.1,
                label=f"Corte HPF: {HPF_CUTOFF_HZ} Hz")
    ax5.set_title("(e) Densidade Espectral de Potência — Método de Welch")
    ax5.set_xlabel("Frequência (Hz)")
    ax5.set_ylabel("PSD (dB/Hz)")
    ax5.set_xlim(0, min(fs / 2, 16000))
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.4)

    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Figura salva: {fig_path}")


def print_signal_metrics(x_orig: np.ndarray, y_proc: np.ndarray, fs: int) -> None:
    """
    Calcula e exibe métricas comparativas entre o sinal original e processado.

    Métricas calculadas:
        - RMS  : Valor eficaz (Root Mean Square) — indica nível médio de energia
        - Peak : Valor de pico absoluto
        - SNRest: Estimativa de melhoria na SNR (razão de energias de sinal/ruído)
        - Dynamic Range: Diferença entre pico e RMS em dB
    """
    rms_orig  = np.sqrt(np.mean(x_orig ** 2))
    rms_proc  = np.sqrt(np.mean(y_proc  ** 2))
    peak_orig = np.max(np.abs(x_orig))
    peak_proc = np.max(np.abs(y_proc))

    print("\n  Métricas do sinal:")
    print(f"  {'Métrica':<30} {'Original':>12}  {'Processado':>12}")
    print("  " + "-" * 58)
    print(f"  {'RMS (amplitude)':<30} {rms_orig:>12.5f}  {rms_proc:>12.5f}")
    print(f"  {'Pico (amplitude)':<30} {peak_orig:>12.5f}  {peak_proc:>12.5f}")
    print(f"  {'RMS (dBFS)':<30} {20*np.log10(rms_orig+1e-12):>11.2f}  "
          f"{20*np.log10(rms_proc+1e-12):>11.2f} dB")
    print(f"  {'Pico (dBFS)':<30} {20*np.log10(peak_orig+1e-12):>11.2f}  "
          f"{20*np.log10(peak_proc+1e-12):>11.2f} dB")


# ==============================================================================
# SEÇÃO 8 — PIPELINE PRINCIPAL DE PROCESSAMENTO
# ==============================================================================

print("=" * 70)
print("  PROCESSAMENTO DE ÁUDIO — REDUÇÃO DE RUÍDO")
print("  Disciplina: Sistemas Lineares | Grupo C | Chebyshev Tipo II")
print("=" * 70)

# ---- Etapa 1: Carregamento do arquivo de áudio ----
print("\n[1/8] Carregando arquivo de áudio...")
try:
    fs, x = wavfile.read(INPUT_WAV)
    print(f"      Arquivo       : {INPUT_WAV}")
    print(f"      Taxa amostral : {fs} Hz")
    print(f"      Formato       : {x.dtype}")
    print(f"      Canais        : {1 if x.ndim == 1 else x.shape[1]}")
    print(f"      Duração       : {len(x)/fs:.3f} s  ({x.shape[0]} amostras)")
except FileNotFoundError:
    print(f"      ERRO: Arquivo '{INPUT_WAV}' não encontrado.")
    exit(1)

# Converte para float32 normalizado e garante formato (N, canais)
x_f = to_float(x)
if x_f.ndim == 1:
    x_f = x_f[:, None]

input_peak = float(np.max(np.abs(x_f)) + 1e-12)

# ---- Etapa 2: Filtro de Wiener (STFT) ----
print("\n[2/8] Aplicando Filtro de Wiener (domínio STFT)...")
print(f"      Alpha (agressividade) : {WIENER_ALPHA}")
print(f"      Floor (piso mínimo)   : {WIENER_FLOOR}")
print(f"      Noise frames          : {NOISE_PERCENT*100:.0f}% dos frames")
y = np.zeros_like(x_f, dtype=np.float32)
for ch in range(x_f.shape[1]):
    print(f"      → Processando canal {ch + 1}/{x_f.shape[1]}...")
    y[:, ch] = denoise_wiener_spectral(x_f[:, ch], fs)

# ---- Etapa 3: Filtro Chebyshev Tipo II (técnica principal) ----
if ENABLE_CHEBY2:
    print("\n[3/8] Aplicando Filtro Chebyshev Tipo II [TÉCNICA PRINCIPAL]...")
    print(f"      Ordem               : {CHEBY2_ORDER}")
    print(f"      Atenuação stopband  : {CHEBY2_RS_DB} dB")
    print(f"      Frequência de corte : {CHEBY2_CUTOFF_HZ} Hz")
    print(f"      Implementação       : SOS + sosfiltfilt (fase zero)")
    y = apply_cheby2_filter(y, fs, CHEBY2_ORDER, CHEBY2_RS_DB, CHEBY2_CUTOFF_HZ)

# ---- Etapa 4: Filtro Butterworth passa-altos ----
if ENABLE_HPF:
    print("\n[4/8] Aplicando Filtro Butterworth Passa-Altos...")
    print(f"      Ordem               : 4")
    print(f"      Frequência de corte : {HPF_CUTOFF_HZ} Hz")
    print(f"      Implementação       : SOS + sosfiltfilt (fase zero)")
    y = apply_hpf_filter(y, fs, HPF_CUTOFF_HZ)

# ---- Etapa 5: Noise Gate ----
if ENABLE_NOISE_GATE:
    print("\n[5/8] Aplicando Noise Gate (domínio temporal)...")
    print(f"      Limiar RMS   : {NOISE_GATE_THRESHOLD}")
    print(f"      Tamanho bloco: 10 ms")
    for ch in range(y.shape[1]):
        y[:, ch] = apply_noise_gate(y[:, ch], fs, NOISE_GATE_THRESHOLD)
# ---- Etapa 6: Ganho de saída e normalização ----
print("\n[6/8] Aplicando ganho de saída e normalização...")
y        = apply_gain_db(y, OUTPUT_GAIN_DB)
y_peak   = float(np.max(np.abs(y)) + 1e-12)
if y_peak > input_peak:                  # Evita amplificação além do pico original
    y = y * (input_peak / y_peak)
y = np.clip(y, -1.0, 1.0)

# ---- Etapa 7: Geração das figuras de análise ----
print("\n[7/8] Gerando figuras de análise...")

# Seleciona canal 0 para análise visual (mono ou L de estéreo)
x_mono = x_f[:, 0]
y_mono = y[:, 0]

print("      → Resposta em frequência do filtro Chebyshev II...")
plot_frequency_response(fs, OUTPUT_FIG_RESP)

print("      → Análise comparativa original vs. processado...")
plot_analysis(x_mono, y_mono, fs, OUTPUT_FIG)

print_signal_metrics(x_mono, y_mono, fs)

# ---- Etapa 8: Salvamento do arquivo WAV processado ----
print("\n[8/8] Salvando arquivo de saída...")
y_out = y[:, 0] if x.ndim == 1 else y
wavfile.write(OUTPUT_WAV, fs, to_int16(y_out))
print(f"      Arquivo salvo: {OUTPUT_WAV}")

# ---- Resumo final ----
print("\n" + "=" * 70)
print("  PROCESSAMENTO CONCLUÍDO")
print("=" * 70)
print(f"\n  Arquivo de entrada  : {INPUT_WAV}")
print(f"  Arquivo de saída    : {OUTPUT_WAV}")
print(f"  Figura de análise   : {OUTPUT_FIG}")
print(f"  Figura do filtro    : {OUTPUT_FIG_RESP}")
print("\n  Parâmetros utilizados:")
print(f"    Wiener Alpha           : {WIENER_ALPHA}")
print(f"    Wiener Floor           : {WIENER_FLOOR}")
print(f"    Chebyshev II — Ordem   : {CHEBY2_ORDER}")
print(f"    Chebyshev II — Corte   : {CHEBY2_CUTOFF_HZ} Hz")
print(f"    Chebyshev II — Rs      : {CHEBY2_RS_DB} dB")
print(f"    Butterworth HPF — Corte: {HPF_CUTOFF_HZ} Hz")
print(f"    Noise Gate — Limiar    : {NOISE_GATE_THRESHOLD}")
print(f"    Ganho de saída         : {OUTPUT_GAIN_DB} dB")
print("\n" + "=" * 70)