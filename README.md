# Análise Comparativa entre Contrações Ventriculares Prematuras e Batimentos Cardíacos Regulares via Transformada Wavelet

## Descrição

Este repositório contém o código, os scripts de análise e os resultados do trabalho **“Análise Comparativa entre Contrações Ventriculares Prematuras e Batimentos Cardíacos Regulares via Transformada Wavelet”**, desenvolvido como parte das atividades da disciplina **Análise de Sistemas Lineares** do curso de Engenharia de Computação da UTFPR – Apucarana.

O objetivo principal é identificar e caracterizar diferenças morfológicas e espectrais entre batimentos cardíacos regulares e contrações ventriculares prematuras (CVPs) utilizando técnicas de análise tempo-frequência, com ênfase na **Transformada Wavelet Contínua (CWT)**.

## Contexto

As doenças cardiovasculares são uma das principais causas de mortalidade no mundo. A análise de sinais eletrocardiográficos (ECG) é essencial para detectar arritmias de forma precoce e precisa. Este projeto utiliza registros do **MIT-BIH Arrhythmia Database**, aplicando transformadas de Fourier, Fourier de Tempo Curto (STFT) e Wavelet para uma análise comparativa aprofundada.

## Tecnologias e Ferramentas

- **Python**
- **WFDB** (Waveform Database)
- **NumPy**
- **PyWavelets**
- **Matplotlib**

## Estrutura do Repositório

```
.
├── db/            # Scripts de download e pré-processamento dos sinais
├── src/             # Scripts principais de processamento e análise
├── README.md        # Este arquivo
├── requirements.txt # Dependências do projeto
```

## Como Executar

1. Clone o repositório:
   ```bash
   git clone https://github.com/Paludeto/case-asl.git
   cd case-asl
   ```

2. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```

3. Execute o script principal:
   ```bash
   python src/main.py
   ```

   Os resultados serão gerados na pasta `results/`.

## Autores

- **Gabriel Paludeto** — [paludeto@alunos.utfpr.edu.br](mailto:paludeto@alunos.utfpr.edu.br)
- **Gabriel Oliveira de Jesus** — [gabrieljesus@alunos.utfpr.edu.br](mailto:gabrieljesus@alunos.utfpr.edu.br)
- **Julia Romanetto dos Santos** — [juliaromanetto@alunos.utfpr.edu.br](mailto:juliaromanetto@alunos.utfpr.edu.br)

## Referências

- MIT-BIH Arrhythmia Database — [PhysioNet](https://physionet.org/physiobank/database/mitdb/)
- Guyton & Hall (2017); Reis et al. (2013); Moody & Mark (2001); entre outros.

## Licença

Este projeto é disponibilizado para fins acadêmicos e educacionais, no contexto da disciplina **Análise de Sistemas Lineares** — UTFPR Apucarana.