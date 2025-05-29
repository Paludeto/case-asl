import matplotlib.pyplot as plt
import numpy as np
import wfdb

# Lê canal MLII (Modified Limb Lead II derivação de membro II modificada)
record = wfdb.rdrecord('201', channels=[0])

fs = record.fs

# Pega array de valores em milivolts
periodo_afib = record.p_signal[0:60 * fs]
periodo_normal = record.p_signal[1430 * fs:1490 * fs]

# np.arange(periodo_afib.size) gera um array de inteiros [0, 1, 2, …, N-1], onde N = periodo_afib.size (número de amostras no trecho AFIB).
# Ao dividir esse array por fs (amostras por segundo), cada índice i vira i/fs segundos — ou seja, o instante de tempo de cada ponto do sinal.
t_afib = np.arange(periodo_afib.size) / fs
t_normal = np.arange(periodo_normal.size) / fs

# Plot normal
plt.figure(figsize=(12, 4))
plt.xlabel('t (s)')
plt.ylabel('A (mV)')
plt.plot(t_normal, periodo_normal)
plt.title('Batimentos Normais')
plt.grid(True)

# Plot fibrilação
plt.figure(figsize=(12, 4))
plt.xlabel('t (s)')
plt.ylabel('A (mV)')
plt.plot(t_afib, periodo_afib)
plt.title('Fibrilação Atrial')
plt.grid(True)

# FFT fibrilação atrial
afib_fft = np.fft.fft(periodo_afib)
afib_fft_amp = np.abs(afib_fft)
afib_fft_freq = np.fft.fftfreq(periodo_afib.size, 1 / fs)

plt.figure(figsize=(12, 4))
mask = afib_fft_freq >= 0
plt.plot(afib_fft_freq[mask], afib_fft_amp[mask])
plt.xlabel('Frequência (Hz)')
plt.ylabel('Amplitude')
plt.title('Espectro AFIB')
plt.grid(True)

# FFT batimentos normais
normal_fft = np.fft.fft(periodo_normal)
normal_fft_amp = np.abs(normal_fft)
normal_fft_freq = np.fft.fftfreq(periodo_normal.size, 1 / fs)

plt.figure(figsize=(12, 4))
mask = normal_fft_freq >= 0
plt.plot(normal_fft_freq[mask], normal_fft_amp[mask])
plt.xlabel('Frequência (Hz)')
plt.ylabel('Amplitude')
plt.title('Espectro Normal')
plt.grid(True)

plt.show()