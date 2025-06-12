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
    def plot(self, beat: int, window: int = 50) -> None:

        idx = self._annotated_r_peaks[beat][0]
        
        sl = slice(idx - window, idx + window)

        plt.plot(list(range(0,100)), self._signal_mv[sl])
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

    print(entry._annotated_r_peaks[1553])
    print(entry2._annotated_r_peaks[1134])
    print(entry3._annotated_r_peaks[1572])

    entry.multi_plot(normal=0, atipico=1553, titulo=f'Batimento Normal x Contração Ventricular Prematura ({100})')  # ~25:13 PVCs
    entry2.multi_plot(normal=0, atipico=1134, titulo=f'Batimento Normal x Contração Ventricular Prematura ({201})') # ~24:15 Aberrated atrial couplet, fusion PVC
    entry3.multi_plot(normal=1, atipico=1572, titulo=f'Batimento Normal x Contração Ventricular Prematura ({105})')