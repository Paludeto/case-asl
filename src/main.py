"""
Uso:
    >>> e = ECGEntry(201)
    >>> e.plot(beat=159)
    >>> plt.show()
"""

from pathlib import Path
from typing import List, Tuple
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import wfdb
from wfdb import processing

# Classe na qual armazenamos informações do sinal
class ECGEntry:

    """
    Representa um único registro MIT-BIH (apenas 1 canal).

    Atributos públicos
    ------------------
    signal_mv : np.ndarray
        Amplitudes do ECG em milivolts.
    sampling_rate_hz : float
        Frequência de amostragem.
    sample_indices : np.ndarray
        Índices inteiros de 0 … N-1.
    r_peaks_indices : np.ndarray
        Posições estimadas dos picos R (QRS) pelo algoritmo XQRS.
    annotated_r_peaks : List[Tuple[int, str]]
        Pares (índice, símbolo) onde o R-peak coincide com a anotação `atr`.
    """

    # Construtor
    def __init__(self, idx: int, db_dir: str | Path = "db", channel: int = 0):
        
        path = Path(__file__).resolve().parent.parent / db_dir / f"{idx}"

        try:
            self.record = wfdb.rdrecord(str(path), channels=[channel])
            self.ann = wfdb.rdann(str(path), extension="atr")
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Registro {idx} não encontrado em {path.parent}") from exc

        # Extração de atributos do record
        self.signal_mv: np.ndarray = self.record.p_signal[:, 0]
        self.sample_indices: np.ndarray = np.arange(self.record.sig_len)

        # Detecção de picos R em sinais QRS
        det = processing.XQRS(self.signal_mv, fs=self.record.fs)
        det.detect()
        self.r_peaks_indices: np.ndarray = det.qrs_inds

        # Combina instantes dos picos R com anotações
        lookup = {s: sym for s, sym in zip(self.ann.sample, self.ann.symbol)}
        self.annotated_r_peaks: List[Tuple[int, str]] = [
            (i, lookup[i]) for i in self.r_peaks_indices if i in lookup
        ]

    # Funções de plot, usando janela de 100 segundos
    def plot(self, beat: int, window: int = 50) -> None:
        
        idx = self.annotated_r_peaks[beat][0]
        
        sl = slice(idx - window, idx + window)

        plt.plot(list(range(0,100)), self.signal_mv[sl])
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

    def busca_ritmo(self, tag: str) -> list[int]:
        return [s for s, note in self.rhythm_events if note.startswith(tag)]

if __name__ == "__main__":

    entry = ECGEntry(100)
    entry2 = ECGEntry(201)
    entry3 = ECGEntry(105)
    entry4 = ECGEntry(200)

    # print(entry4.ann.aux_note)

    print(entry.annotated_r_peaks[1553])
    print(entry2.annotated_r_peaks[1134])
    print(entry3.annotated_r_peaks[1572])

    entry.multi_plot(normal=0, atipico=1553, titulo=f'Batimento Normal x Contração Ventricular Prematura ({100})')  # ~25:13 PVCs
    entry2.multi_plot(normal=0, atipico=1134, titulo=f'Batimento Normal x Contração Ventricular Prematura ({201})') # ~24:15 Aberrated atrial couplet, fusion PVC
    entry3.multi_plot(normal=1, atipico=1572, titulo=f'Batimento Normal x Contração Ventricular Prematura ({105})')