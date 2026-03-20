# Processamento de Áudio com Filtro Chebyshev Tipo II

🎵 Script Python para redução de ruído em sinais de áudio utilizando o filtro IIR Chebyshev Tipo II como técnica principal.

---

## 📋 Descrição

Este projeto implementa um **pipeline completo de processamento digital de sinais (DSP)** para remover ruído de arquivo de áudio. A técnica principal é o **filtro Chebyshev Tipo II (passa-baixas)**, que se destaca por apresentar:

- ✅ Resposta **maximamente plana** na banda de passagem (sem ripple)
- ✅ Atenuação controlada e **equiripple** na banda de rejeição
- ✅ Transição abrupta passband→stopband
- ✅ Zeros de transmissão para atenuação infinita em certas frequências

### Pipeline de Processamento

O script executa os seguintes estágios em sequência:

1. **Wiener Filter (domínio STFT)** — Estimativa e subtração do ruído estacionário
2. **Chebyshev Tipo II (passa-baixas)** — Atenuação de altas frequências
3. **Butterworth (passa-altos)** — Remoção de componentes muito baixas (< 300 Hz)
4. **Noise Gate (domínio temporal)** — Silenciamento de blocos abaixo do limiar de energia

---

## 🔧 Dependências

```
numpy >= 1.21
scipy >= 1.7
matplotlib >= 3.4
```

### Instalação

```bash
pip install numpy scipy matplotlib
```

---

## 🚀 Como Usar

### Configuração Rápida

1. Coloque seu arquivo de áudio em `.wav` na raiz do projeto (ou ajuste `INPUT_WAV` no script)
2. Execute:

```bash
python index.py
```

### Saídas Geradas

O script produz automaticamente:

- 📁 `Arquivo3_filtrado_v2.wav` — Áudio processado (16-bit PCM)
- 📊 `av3_analise.png` — Gráfico de análise espectral (antes/depois)
- 📈 `av3_resposta_freq.png` — Resposta em frequência do filtro

---

## ⚙️ Parâmetros de Configuração

### Entrada/Saída
| Parâmetro | Padrão | Descrição |
|-----------|--------|-----------|
| `INPUT_WAV` | `Arquivo3.wav` | Arquivo de áudio de entrada |
| `OUTPUT_WAV` | `Arquivo3_filtrado_v2.wav` | Nome do arquivo processado |
| `OUTPUT_GAIN_DB` | `-0.5` | Ganho aplicado após processamento (em dB) |

### STFT (Transformada de Fourier em Tempo Curto)
| Parâmetro | Padrão | Descrição |
|-----------|--------|-----------|
| `N_FFT` | `2048` | Tamanho da janela de análise (amostras) |
| `HOP` | `512` | Deslocamento entre janelas |
| `WINDOW` | `"hamming"` | Tipo de janela (reduz vazamento espectral) |

### Filtro de Wiener (Domínio STFT)
| Parâmetro | Padrão | Descrição |
|-----------|--------|-----------|
| `WIENER_ALPHA` | `5.4` | Fator de superestimação do ruído |
| `WIENER_FLOOR` | `0.027` | Ganho mínimo (evita supressão total) |
| `NOISE_PERCENT` | `0.37` | Fração de frames para estimar ruído (37%) |
| `ENABLE_SPECTRAL_GATE` | `True` | Ativa máscara espectral |
| `GATE_THRESHOLD` | `0.055` | Limiar da máscara espectral |

### Filtro Chebyshev Tipo II (Passa-Baixas) — **TÉCNICA PRINCIPAL**
| Parâmetro | Padrão | Descrição |
|-----------|--------|-----------|
| `ENABLE_CHEBY2` | `True` | Ativa o filtro Chebyshev |
| `CHEBY2_ORDER` | `8` | Ordem do filtro (transição mais abrupta) |
| `CHEBY2_RS_DB` | `100` | Atenuação mínima no stopband (dB) |
| `CHEBY2_CUTOFF_HZ` | `7200` | Frequência de corte (Hz) |

### Filtro Butterworth (Passa-Altos)
| Parâmetro | Padrão | Descrição |
|-----------|--------|-----------|
| `ENABLE_HPF` | `True` | Ativa o filtro passa-altos |
| `HPF_CUTOFF_HZ` | `300` | Frequência de corte inferior (Hz) |

### Noise Gate (Domínio Temporal)
| Parâmetro | Padrão | Descrição |
|-----------|--------|-----------|
| `ENABLE_NOISE_GATE` | `True` | Ativa silenciamento de blocos baixos |
| `NOISE_GATE_THRESHOLD` | `0.008` | Limiar de energia RMS |

---

## 📐 Fundamentação Teórica

### Filtro Chebyshev Tipo II

A função de transferência ao quadrado é definida por:

$$|H(j\Omega)|^2 = \frac{1}{1 + \varepsilon^2 \cdot \frac{T_n^2(\Omega_s/\Omega)}{T_n^2(\Omega_s/\Omega_c)}}$$

Onde:
- $T_n(x)$ é o polinômio de Chebyshev de ordem $n$
- $\varepsilon$ é o parâmetro de ripple no stopband
- $\Omega_s$ é a frequência de início do stopband
- $\Omega_c$ é a frequência de corte

**Vantagens sobre o Chebyshev Tipo I:**
- Sem ripple na banda de passagem
- Resposta monotônica no passband
- Transição muito abrupta

### Filtro de Wiener Espectral

Para cada bin tempo-frequência:

$$G(f,t) = \frac{\text{SNR}(f,t)}{\text{SNR}(f,t) + 1}$$

Onde $\text{SNR}(f,t) = \frac{P(f,t)}{\alpha \cdot N(f)}$ combina o espectro do sinal com estimativa de ruído.

---

## 📁 Estrutura de Pastas

```
filtro-Chebyshev/
├── index.py                    # Script principal
├── README.md                   # Esta documentação
├── Audios/                     # Pasta para áudios de entrada
├── Gráficos/                   # Pasta para gráficos de saída
└── *.wav                       # Arquivos de áudio processados
```

---

## 💡 Dicas de Otimização

### Para reduzir mais ruído:
- Aumente `WIENER_ALPHA` (p.ex. 7.0)
- Diminua `WIENER_FLOOR` (p.ex. 0.01)
- Aumente `CHEBY2_ORDER` (p.ex. 10)

### Para preservar mais detalhes:
- Diminua `WIENER_ALPHA` (p.ex. 3.0)
- Aumente `WIENER_FLOOR` (p.ex. 0.05)
- Aumente `CHEBY2_CUTOFF_HZ` (p.ex. 8000)

### Para processar mais rápido:
- Diminua `N_FFT` (p.ex. 1024)
- Aumente `HOP` (p.ex. 1024)

---

## ✨ Autor

Desenvolvido com ❤️ para processamento digital de sinais de áudio.
