<div align="center">
  <p float="left" align="middle">
    <img src="https://upload.wikimedia.org/wikipedia/commons/f/fa/Apple_logo_black.svg" alt="Apple Logo" width="200"/>
    &nbsp;&nbsp;&nbsp;&nbsp;
    <img src="https://www.fiap.com.br/wp-content/themes/fiap2016/images/sharing/fiap.png" alt="FIAP Logo" width="300"/>
  </p>
</div>

# 📈 Previsão de Preço de Fechamento de Ações com LSTM

[![Python Version](https://img.shields.io/badge/python-3.11-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-REST-green.svg)](https://fastapi.tiangolo.com/)
[![Tensorflow](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-production-green)](/)
[![Documentation](https://img.shields.io/badge/docs-openapi-green)](/docs)
[![Model](https://img.shields.io/badge/model-LSTM-brightgreen.svg)](/)
[![Dataset](https://img.shields.io/badge/data-yfinance-blue)](https://finance.yahoo.com/)
[![FIAP](https://img.shields.io/badge/FIAP-project-red.svg)](https://www.fiap.com.br)

---

## 📋 Sobre o Projeto

Este projeto aplica redes neurais recorrentes do tipo **LSTM (Long Short-Term Memory)** para prever o preço de fechamento da ação da **Apple Inc. (AAPL)** com base nos últimos **45 dias úteis** de dados históricos do mercado. A solução foi desenvolvida em Python com treinamento supervisionado e está disponível via **API REST** utilizando **FastAPI**.

---

## 🧠 Visão Geral

- 🧠 Modelo LSTM treinado com dados reais da AAPL via `yfinance`
- 🔁 Janela de previsão baseada nos últimos 45 dias da série temporal
- 🌐 API REST para consulta de previsões por data
- 📉 Dados suavizados por média móvel simples (MM3)
- 📊 Avaliação com MAE, RMSE e MAPE

---

## 📁 Estrutura do Projeto

```
API_PRECOS/
├── app/
│   ├── routes/               # Rotas da API
│   └── utils/                # Utilitários e helpers
├── env/                      # Ambiente virtual Python
├── requirements.txt          # Dependências do projeto
└── run.py                    # Arquivo de inicialização

MODELO/
|FIAP_CHALLENGE_4_analise_exploratoria.ipynb       # Notebook com a análise exploratória dos dados históricos da AAPL
|modelo_LSTM_45dias.ipynb                          # Notebook com o desenvolvimento e treinamento do modelo LSTM
|modelo_45dias.h5                                  # Arquivo do modelo LSTM treinado salvo no formato HDF5
|scaler_45dias.pkl                                 # Objeto de scaler MinMaxScaler treinado, usado para normalização dos dados
|modelo_prod_lstm_45dias.py                        # Script FastAPI para servir o modelo em produção
|df_finance.csv                                    # Base de dados histórica da ação AAPL usada na modelagem
```

## 📁 Documentação

```
Swagger: http://35.198.47.221:5000/apidocs/
Deploy: http://35.198.47.221
Server: Google Cloud

```


🔗 Endpoints

### Previsão
- `POST /prever_com_arquivo`: Previsão utilizando arquivo csv
- `POST /prever_com_data`: Previsão utilizando data final

## 🔐 Autenticação

A API utiliza autenticação basica.

User test: admin
Senha test: secret


## 📊 Avaliação do Modelo
O modelo foi avaliado com as seguintes métricas:

MAE – Mean Absolute Error

RMSE – Root Mean Squared Error

MAPE – Mean Absolute Percentage Error

O melhor desempenho foi obtido utilizando uma janela de 45 dias com média móvel de 3 dias (MM3).

## ℹ️ Observações
A coleta dos dados da rota prever_com_data é feita automaticamente pelo yfinance a partir da API.

A previsão considera os 45 dias úteis anteriores à data solicitada.

Não há necessidade de baixar datasets manualmente.
