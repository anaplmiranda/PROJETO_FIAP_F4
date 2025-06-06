# PROJETO: Previsão de Preço de Fechamento de Ações com LSTM

DESCRIÇÃO:
Este projeto utiliza redes neurais LSTM (Long Short-Term Memory) para prever o valor de fechamento da ação da Apple Inc. (AAPL), com uma janela de observação de 45 dias. O modelo foi treinado com dados históricos obtidos do Yahoo Finance e está disponível em produção via API desenvolvida com ....

ARQUIVOS:

1. FIAP_CHALLENGE_4_analise_exploratoria.ipynb
   - Notebook com a análise exploratória dos dados históricos da AAPL.

2. modelo_LSTM_45dias.ipynb
   - Notebook com o desenvolvimento e treinamento do modelo LSTM.

3. modelo_45dias.h5
   - Arquivo do modelo LSTM treinado salvo no formato HDF5.

4. scaler_45dias.pkl
   - Objeto de scaler MinMaxScaler treinado, usado para normalização dos dados.

5. modelo_prod_lstm_45dias.py
   - Script FastAPI para servir o modelo em produção.

6. df_finance.csv
   - Base de dados histórica da ação AAPL usada na modelagem.

FUNCIONALIDADES:

- API REST disponível em FastAPI.
- Endpoint principal:
  - GET /prever?data_final=YYYY-MM-DD
  - Exemplo: /prever?data_final=2025-05-20
  - Retorno: Previsão do próximo preço de fechamento da ação AAPL.

REQUISITOS:

- Python 3.11+
- Bibliotecas:
  - fastapi
  - uvicorn
  - yfinance
  - joblib
  - tensorflow
  - scikit-learn

EXECUÇÃO LOCAL:

1. Instale os pacotes:
   pip install fastapi uvicorn yfinance joblib tensorflow scikit-learn

2. Execute a API:
   uvicorn modelo_prod_lstm_45dias:app --reload

AVALIAÇÃO:

- O modelo foi avaliado utilizando métricas MAE, RMSE e MAPE.
- A janela de 45 dias com retorno suavizado (MM3) demonstrou bom desempenho.

OBSERVAÇÃO:

- A coleta de dados é automática via yfinance e ocorre dentro do script de produção.
- A previsão é feita com base nos últimos 45 dias da série temporal disponíveis até a data especificada.

