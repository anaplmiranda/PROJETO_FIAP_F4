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

python -m venv env
source env/bin/activate  # Linux/MacOS
# ou
.\env\Scripts\activate   # Windows


pip install -r requirements.txt

uvicorn modelo_prod_lstm_45dias:app --reload

curl "http://localhost:8000/prever?data_final=2025-05-20"


🔗 Endpoint Principal
GET /prever?data_final=YYYY-MM-DD
Exemplo: /prever?data_final=2025-05-20
Retorno: Previsão do próximo valor de fechamento da AAPL

http://35.198.47.221:5000/apidocs/


📊 Avaliação do Modelo
O modelo foi avaliado com as seguintes métricas:

MAE – Mean Absolute Error

RMSE – Root Mean Squared Error

MAPE – Mean Absolute Percentage Error

O melhor desempenho foi obtido utilizando uma janela de 45 dias com média móvel de 3 dias (MM3).

ℹ️ Observações
A coleta dos dados é feita automaticamente pelo yfinance a partir da API.

A previsão considera os 45 dias úteis anteriores à data solicitada.

Não há necessidade de baixar datasets manualmente.

📄 
