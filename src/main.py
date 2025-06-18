"""
Uso:
    >>> e = ECGEntry(201)
    >>> e.plot(beat=159)
    >>> plt.show()
"""

from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import wfdb
from wfdb import processing
import pywt
from scipy.signal import spectrogram

# Classe na qual armazenamos informações do sinal
class ECGEntry:

    _record:            wfdb.Record
    _ann:               wfdb.Annotation
    _signal_mv:         np.ndarray
    _sample_indices:    np.ndarray
    _r_peaks_indices:   np.ndarray
    _annotated_r_peaks: List[Tuple[int, str]]

    # Construtor
    def __init__(self, idx: int):

        self._load_data(idx)
        self._detect_peaks()     
        self._annotate_peaks()  

    # Carrega dados para o objeto ECGEntry
    def _load_data(self, idx: int, db_dir: str = 'db') -> None:

        path = Path(__file__).resolve().parent.parent / db_dir / f"{idx}"

        try:
            self._record = wfdb.rdrecord(str(path), channels=[0])
            self._ann = wfdb.rdann(str(path), extension="atr")
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Registro {idx} não encontrado em {path.parent}") from exc

        # Extração de atributos do record
        self._signal_mv: np.ndarray = self._record.p_signal[:, 0]
        self._sample_indices: np.ndarray = np.arange(self._record.sig_len)
    
    # Usa XQRS para detecção de picos automática
    def _detect_peaks(self) -> None:

        # Detecção de picos R em sinais QRS
        det = processing.XQRS(self._signal_mv, fs=self._record.fs)
        det.detect()
        self._r_peaks_indices: np.ndarray = det.qrs_inds

    # Combina número de amostra + tipo de batida em uma tupla
    def _annotate_peaks(self) -> None:

        lookup = {s: sym for s, sym in zip(self._ann.sample, self._ann.symbol)}
        self._annotated_r_peaks: List[Tuple[int, str]] = [
            (i, lookup[i]) for i in self._r_peaks_indices if i in lookup
        ]

    # Funções de plot, usando janela de 100 segundos
    def plot(self, beat: int, window: int = 100) -> None:

        idx = self._annotated_r_peaks[beat][0]
        
        sl = slice(idx - window, idx + window)

        plt.plot(list(range(0,200)), self._signal_mv[sl])
        plt.xlabel("Amostras")
        plt.ylabel("mV")

    def multi_plot(self, normal: int, atipico: int, titulo: str) -> None:

        plt.figure()
        plt.grid(True)
        plt.title(titulo)
        self.plot(beat=normal)
        self.plot(beat=atipico)

        plt.legend(["Batimento Normal", "Arritmia"])
        plt.tight_layout()
        plt.show()

    
    # Cria um dicionário que mapeia cada tipo de batimento para uma lista de amostras em que esses batimentos ocorrem no sinal
    def build_annotation_table(self) -> dict:
        tabela = {}

        for idx, tipo in self._annotated_r_peaks:
            if tipo not in tabela:
                tabela[tipo] = []
            tabela[tipo].append(idx)

        return tabela
    
    # Retorna uma lista com as posições na lista de batimentos anotados que correspondem ao tipo informado
    def get_beats_by_annotation(self, tipo: str) -> List[int]:
        return [i for i, (_, t) in enumerate(self._annotated_r_peaks) if t == tipo]
    
    # Calcula a forma de onda média para um tipo de batimento
    def calculate_average_beat(self, beat_type: str, window_size: int = 100) -> np.ndarray | None:
        tabela = self.build_annotation_table()
        locations = tabela.get(beat_type, [])

        all_beats_signals = []
        for loc in locations:
            start = loc - window_size
            end = loc + window_size
            if start >= 0 and end < len(self._signal_mv):
                all_beats_signals.append(self._signal_mv[start:end])

        if not all_beats_signals:
            print(f"Não foi possível extrair janelas de sinal para o tipo '{beat_type}'.")
            return None
        
        return np.mean(np.array(all_beats_signals), axis=0)
    
    # Plota a média das ondas no gráfio
    def plot_waveforms(self, waveforms: dict, titulo: str) -> None:
        plt.figure(figsize=(12, 7))
        plt.title(titulo)

        for label, waveform in waveforms.items():
            if waveform is not None:
                window_size = len(waveform) // 2
                x_axis = np.arange(-window_size, window_size)
                plt.plot(x_axis, waveform, label = f"{label} (média dos batimentos)")
                plt.xlabel("Amostras (em relação ao pico R)")
        plt.ylabel("mV")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    # Plota Sinal com Fourier
    def plot_frequency_spectrum(self, waveform: np.ndarray, titulo: str) -> None:
        fs = self._record.fs
        N = len(waveform)

        yf = np.fft.rfft(waveform)
        yf_magnitude = 2.0/N * np.abs(yf)

        xf = np.fft.rfftfreq(N, 1/fs)

        plt.figure(figsize=(12, 7))
        plt.plot(xf, yf_magnitude)
        plt.title(titulo)
        plt.xlabel("Frequência (Hz)")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.xlim(0, 60)
        plt.tight_layout()
        plt.show()

    def plot_wavelet_scalogram(self, waveform: np.ndarray, titulo: str) -> None:
        # Frequencia de amostragem do sinal 
        fs = self._record.fs

        # Frequencia ao longo do tempo com Wavelet usando morlet
        scales = np.arange(1, 128)
        coef, freqs = pywt.cwt(waveform, scales, 'morl', sampling_period=1/fs)

        # Definição do tempo
        duracao = len(waveform) / fs
        t = np.linspace(0, duracao, num=len(waveform), endpoint=False)

        plt.figure(figsize=(12, 6))
        plt.pcolormesh(t, freqs, np.abs(coef), cmap='viridis', shading='gouraud')
        plt.title(titulo)
        plt.xlabel('Tempo (s)')
        plt.ylabel('Frequência (Hz)')
        plt.colorbar(label='Magnitude do Coeficiente CWT')
        plt.tight_layout()

    def plot_fourier_scalogram(self, waveform: np.ndarray, titulo: str) -> None:
        fs = self._record.fs

        # Calcula o espectrograma (STFT)
        f, t, Sxx = spectrogram(waveform, fs=fs, window='hann', nperseg=64, noverlap=48)

        plt.figure(figsize=(12, 6))
        plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud', cmap='viridis')
        plt.title(f"Espectrograma de Fourier (STFT) - {titulo}")
        plt.xlabel("Tempo (s)")
        plt.ylabel("Frequência (Hz)")
        plt.colorbar(label="Intensidade (dB)")
        plt.tight_layout()
        plt.show()
            



    


if __name__ == "__main__":

    entry = ECGEntry(100)
    entry2 = ECGEntry(201)
    entry3 = ECGEntry(105)
    entry4 = ECGEntry(200)

    normais_100 = entry.get_beats_by_annotation('N')
    ventriculares_100 = entry.get_beats_by_annotation('V')

    normais_201 = entry2.get_beats_by_annotation('N')
    ventriculares_201 = entry2.get_beats_by_annotation('V')

    normais_105 = entry3.get_beats_by_annotation('N')
    ventriculares_105 = entry3.get_beats_by_annotation('V')

    normais_200 = entry4.get_beats_by_annotation('N')
    ventriculares_200 = entry4.get_beats_by_annotation('V')

    entry.multi_plot(normal=normais_100[0], atipico=ventriculares_100[0], titulo=f'Batimento Normal x PVC ({100})')

    entry2.multi_plot(normal=normais_201[0], atipico=ventriculares_201[0], titulo=f'Batimento Normal x PVC ({201})')

    entry3.multi_plot(normal=normais_105[1], atipico=ventriculares_105[0], titulo=f'Batimento Normal x PVC ({105})')

    entry4.multi_plot(normal=normais_200[0], atipico=ventriculares_200[0], titulo=f'Batimento Normal x PVC ({200})')

    
    avg_normal_beat = entry.calculate_average_beat(beat_type='N')
    avg_pvc_beat = entry.calculate_average_beat(beat_type='V')

    formas_de_onda_para_plotar = {
        "Normal": avg_normal_beat,
        "PVC": avg_pvc_beat,
    }

    entry.plot_waveforms(formas_de_onda_para_plotar, "media")

    entry.plot_frequency_spectrum(avg_normal_beat, "Fourier Normal")
    entry.plot_frequency_spectrum(avg_pvc_beat, "Fourier PVC")

    entry.plot_wavelet_scalogram(avg_normal_beat, "normal")
    entry.plot_wavelet_scalogram(avg_pvc_beat, "pvc")

    entry.plot_fourier_scalogram(avg_normal_beat, "normal")
    entry.plot_fourier_scalogram(avg_pvc_beat, "pvc")

    plt.show()