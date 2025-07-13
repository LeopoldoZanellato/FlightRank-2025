# ✈️ FlightRank 2025 — Recomendador de Voos Corporativos

Este projeto resolve um desafio de sistemas de recomendação no contexto de viagens corporativas, com foco em **ranking supervisionado de opções de voo**.

## 🎯 Objetivo

Prever qual voo será selecionado por um viajante corporativo, ordenando as opções disponíveis em uma busca. Utiliza-se o algoritmo **LightGBM LambdaRank**, com engenharia de atributos robusta e otimização por `ndcg@3`.

## 📁 Estrutura do Projeto

```
├── main_pt.ipynb           # Notebook principal com pipeline completo em português
├── main_en.ipynb           # Versão em inglês (se aplicável)
├── main_pt.py              # Script em Python com as mesmas etapas do notebook
├── modelo_final.txt        # Modelo LightGBM treinado com todos os dados
├── submission.csv          # Arquivo de submissão final com ranking gerado
├── requirements.txt        # Dependências do projeto
├── README.md               # Este arquivo
└── data/
    └── aeroclub/
        ├── train.parquet            # Dados de treino
        ├── test.parquet             # Dados de teste
        ├── sample_submission.csv    # Exemplo de submissão
        └── jsons_structure.md       # JSON Completo
```

## 📊 Dados

- **Origem**: Competição **AeroClub RecSys 2025** (Kaggle)
- **Formato**: `.parquet`
- **Agrupamento por** `ranker_id` (sessões de busca)
- Apenas **um voo é o correto** (`selected = 1`) por grupo

## 🛠️ Pipeline

1. **Download e extração automática** via API do Kaggle  
2. **Pré-processamento** e redução de memória  
3. **Engenharia de features**:
   - Datas e horários  
   - Quantidade de segmentos  
   - Fidelidade do usuário (`frequentFlyer`)  
   - Duração real dos voos  
   - Regras tarifárias  
   - Origem/destino da rota (`searchRoute`)  
4. **Divisão treino/validação** com `GroupShuffleSplit`  
5. **Treinamento com LightGBM LambdaRank**  
6. **Busca manual de hiperparâmetros**  
7. **Geração de submissão final com modelo salvo**  

## 📏 Métrica

**HitRate@3** — verifica se o voo correto está entre os 3 primeiros colocados no ranking de cada sessão com mais de 10 voos.

## 🧪 Requisitos

```bash
pip install -r requirements.txt
```

**Dependências principais**:

```text
- pandas
- numpy
- lightgbm
- scikit-learn
- matplotlib
```

## 🧠 Modelo

- **Modelo**: LightGBM (LambdaRank)  
- **Métrica de validação**: `ndcg@3`  
- **Parâmetros**: otimizados manualmente (grid search)  
- **Salvo como**: `modelo_final.txt`

## 📝 Submissão

- **Arquivo**: `submission.csv`  
- **Formato**:

```csv
Id, ranker_id, selected
```

A coluna `selected` indica a posição do voo no ranking do modelo (1 = topo).

## 🔄 Execução

Para treinar e gerar a submissão:

```bash
python main_pt.py
```

Ou use o notebook:

```bash
jupyter notebook main_pt.ipynb
```

## 🧩 Observações

- As sessões de busca com menos de 10 voos são ignoradas pela métrica oficial.  
- O pipeline é modular e facilmente adaptável para outros contextos de ranking.
- Tem a opção de rodar com todos os dados ou apenas com um numero limitado para testes