import os
import time
import json
import psutil
import logging
from flask import request, jsonify, g
from app import app, auth
import pandas as pd
import joblib
from werkzeug.utils import secure_filename
import numpy as np
import yfinance as yf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Configurações
app.config['df_finance.csv'] = 'uploads'
os.makedirs(app.config['df_finance.csv'], exist_ok=True)
os.makedirs("logs", exist_ok=True)

# Logger
logger = logging.getLogger("monitoramento")
logger.setLevel(logging.INFO)
handler = logging.FileHandler("logs/monitoramento.log")
formatter = logging.Formatter('%(asctime)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# Carregamento do modelo
modelo = load_model('modelo/modelo_45dias.h5')
scaler = joblib.load('modelo/scaler_45dias.pkl')

# Antes de cada requisição
@app.before_request
def before_request_monitoramento():
    g.start_time = time.time()
    g.cpu_start = psutil.cpu_percent(interval=None)
    g.mem_start = psutil.virtual_memory().percent
    g.request_path = request.path
    g.payload_size = len(request.data)

# Após cada requisição
@app.after_request
def after_request_monitoramento(response):
    duration = time.time() - g.start_time
    cpu_end = psutil.cpu_percent(interval=None)
    mem_end = psutil.virtual_memory().percent

    log_data = {
        "rota": g.request_path,
        "tempo_total_s": round(duration, 3),
        "cpu_inicio_%": g.cpu_start,
        "cpu_fim_%": cpu_end,
        "mem_inicio_%": g.mem_start,
        "mem_fim_%": mem_end,
        "status_code": response.status_code,
        "payload_bytes": g.payload_size
    }

    logger.info(json.dumps(log_data, ensure_ascii=False))
    return response

# Função auxiliar para criar sequências de 45 dias
def criar_sequencias(dados, janela=45):
    return np.array([dados[i:i+janela] for i in range(len(dados) - janela + 1)])

@app.route('/prever_com_arquivo', methods=['POST'])
@auth.login_required
def prever_com_arquivo():

    """
    Prever Proximo Faturamento Utilizando o arquivo.
    ---   
    tags:
      - Prever Faturamento
    consumes:
      - multipart/form-data
    parameters:
      - name: df_finance.csv
        in: formData
        type: file
        required: true
        description: Arquivo CSV com dados financeiros
    responses:
      200:
        description: Upload bem-sucedido    
      400:
        description: Bad Request
      500:
        description: Internal Error          
    """

    if 'df_finance.csv' not in request.files:
        return jsonify({'erro': 'Arquivo não enviado'}), 400

    file = request.files['df_finance.csv']
    if file.filename == '':
        return jsonify({'erro': 'Nome de arquivo vazio'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['df_finance.csv'], filename)
    file.save(filepath)

    try:
        dados = pd.read_csv(filepath)
        dados['Retorno_Diário'] = dados['Preço_Fechamento'].pct_change()
        dados['Retorno_MM3'] = dados['Retorno_Diário'].rolling(window=3).mean()
        dados.dropna(inplace=True)

        dados = dados[scaler.feature_names_in_]
        dados = dados.apply(pd.to_numeric, errors='coerce').fillna(0)

        if dados.shape[1] == 0:
            return jsonify({'erro': 'Nenhuma coluna numérica válida encontrada.'}), 400

        dados_escalados = scaler.transform(dados)
        sequencias = criar_sequencias(dados_escalados)

        if len(sequencias) == 0:
            return jsonify({'erro': 'Dados insuficientes para formar sequências de 45 dias.'}), 400

        inicio_modelo = time.time()
        cpu_modelo_inicio = psutil.cpu_percent(interval=None)
        mem_modelo_inicio = psutil.virtual_memory().percent

        previsao_normalizada = modelo.predict(sequencias[-1].reshape(1, 45, -1))

        duracao_modelo = time.time() - inicio_modelo
        cpu_modelo_fim = psutil.cpu_percent(interval=None)
        mem_modelo_fim = psutil.virtual_memory().percent

        logger.info(json.dumps({
            "etapa": "inferencia_modelo",
            "rota": request.path,
            "tempo_inferencia_s": round(duracao_modelo, 3),
            "cpu_inicio_%": cpu_modelo_inicio,
            "cpu_fim_%": cpu_modelo_fim,
            "mem_inicio_%": mem_modelo_inicio,
            "mem_fim_%": mem_modelo_fim
        }, ensure_ascii=False))

        previsao_real = scaler.inverse_transform(
            np.hstack([previsao_normalizada, np.zeros((1, len(scaler.feature_names_in_) - 1))])
        )[0, 0]

        return jsonify({'previsao_proximo_fechamento': round(previsao_real, 2)})

    except Exception as e:
        return jsonify({'erro': str(e)}), 500

@app.route('/prever_com_data', methods=['POST'])
@auth.login_required
def prever_com_data():

    """
    Prever Próximo Faturamento Utilizando Data Final.
    ---
    tags:
      - Prever Faturamento
    consumes:
      - multipart/form-data
    parameters:
      - name: data_final
        in: formData
        type: string
        required: true
        description: Data final no formato (YYYY-MM-DD)
    responses:
      200:
        description: Previsão gerada com sucesso
      400:
        description: Requisição inválida
      500:
        description: Erro interno do servidor
    """
    
    data_final = request.form.get('data_final')
    if not data_final:
        return jsonify({'erro': 'Parâmetro data_final não informado'}), 400

    try:
        df = yf.download("AAPL", start="2018-01-01", end=data_final)
        df.columns = df.columns.droplevel(0) if isinstance(df.columns, pd.MultiIndex) else df.columns
        df.columns.name = None
        df.columns = ['Preço_Fechamento', 'Máxima', 'Mínima', 'Preço_Abertura', 'Volume_Negociado']
        df.reset_index(inplace=True)

        df['Retorno_Diário'] = df['Preço_Fechamento'].pct_change()
        df['Retorno_MM3'] = df['Retorno_Diário'].rolling(window=3).mean()
        df.dropna(inplace=True)

        variaveis = ['Preço_Fechamento', 'Máxima', 'Mínima', 'Volume_Negociado', 'Retorno_MM3']
        ultimos_dados = df[variaveis].iloc[-45:]
        X_input = scaler.transform(ultimos_dados).reshape(1, 45, len(variaveis))

        inicio_modelo = time.time()
        cpu_modelo_inicio = psutil.cpu_percent(interval=None)
        mem_modelo_inicio = psutil.virtual_memory().percent

        previsao_normalizada = modelo.predict(X_input)

        duracao_modelo = time.time() - inicio_modelo
        cpu_modelo_fim = psutil.cpu_percent(interval=None)
        mem_modelo_fim = psutil.virtual_memory().percent

        logger.info(json.dumps({
            "etapa": "inferencia_modelo",
            "rota": request.path,
            "tempo_inferencia_s": round(duracao_modelo, 3),
            "cpu_inicio_%": cpu_modelo_inicio,
            "cpu_fim_%": cpu_modelo_fim,
            "mem_inicio_%": mem_modelo_inicio,
            "mem_fim_%": mem_modelo_fim
        }, ensure_ascii=False))

        previsao = scaler.inverse_transform(
            np.hstack([previsao_normalizada, np.zeros((1, len(variaveis) - 1))])
        )[0, 0]

        return jsonify({"previsao_proximo_fechamento": round(previsao, 2)})

    except Exception as e:
        return jsonify({'erro': str(e)}), 500
