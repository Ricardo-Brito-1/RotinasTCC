###############################################################################
# Curso: MBA em Data Science e Analytics
# Trabalho de Conclusão de Curso - TCC

# Resultados Preliminares

# Predição de Ações B3: Comparativo setorial com Random Forest e Redes Neurais

# Objetivo: Desenvolver dois modelos de ML – RF e RNAs – para predizer o preço de
# ações de quatro setores distintos da B3 (Financeiro, Varejo, Petróleo, Gás e
# Biocombustíveis, e Materiais Básicos), utilizando dados históricos a partir de
# janeiro de 2015, a fim de comparar o desempenho dos modelos com base em métricas
# de erro e a adaptabilidade de cada modelo aos diferentes setores.

# Ações: ITUB4, MGLU3, PETR4 e VALE3

###############################################################################

#%% In[1]: Importou-se bibliotecas necessárias para a análise, instalou-se o yfinance e
# buscou-se os dados para início do TCC

# Busca dos dados no yfinance
!pip install yfinance
import yfinance as yf

# Manipulação e análise de dados
import pandas as pd
import numpy as np
from itertools import product

# Visualização
import matplotlib.pyplot as plt
import seaborn as sns

# Trabalho com datas e frequências (Sérires Temporais)
from datetime import datetime
import matplotlib.dates as mdates

# Para estatísticas de séries temporais
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Para modelagem com ARIMA dos 4 papéis (modelo tradicional)
# Instalou-se a biblioteca para função auto_arima - Identificação automática para modelo ARIMA
!pip install pmdarima
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Para escalonamento e divisão dos dados
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

# Para importar o modelo de ML Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor

# Para importar os modelos modelos LSTM e GRU
!pip install tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU
from tensorflow.keras.optimizers import Adam

#%% In[2]: Definiu-se os ativos (ticker) e o intervalo de datas, de inicio e de término.

tickers = {
    'ITUB4': 'ITUB4.SA',
    'MGLU3': 'MGLU3.SA',
    'PETR4': 'PETR4.SA',
    'VALE3': 'VALE3.SA'
}

start_date = "2015-01-01"
end_date = "2025-05-30"

#%% In[3]: Definiu-se a função a ser aplicada para download dos dados

def obter_dados(ticker):
    return yf.download(ticker, start=start_date, end=end_date)

#%% In[4]: Coletou-se os dados dos ativos e os salva em arquivos .csv

dados = {nome: obter_dados(tk) for nome, tk in tickers.items()}

for nome, df in dados.items():
    df.to_csv(f"dados_{nome.lower()}.csv", sep=';', decimal=',')

#%% In[5]: Efetou-se a leitura dos arquivos .csv e tratamento de colunas

for nome in tickers.keys():
    df = pd.read_csv(
        f"dados_{nome.lower()}.csv",
        sep=';',
        decimal=',',
        header=[0, 1],        # <- lê os dois níveis de cabeçalho
        index_col=0,          # <- índice é a coluna "Date"
        parse_dates=True
    )
    df.columns = df.columns.droplevel(1)  # remove o ticker
    df.index = pd.to_datetime(df.index)   # garante datetime
    dados[nome] = df

#%% In[6]: Verificou-se estrutura e consistência

for nome, df in dados.items():
    print(f"\n===== {nome} =====")
    print("Colunas:", df.columns.tolist())
    print("Tipo de índice:", type(df.index))
    print("Período:", df.index.min().date(), "→", df.index.max().date())
    print("Total de registros:", len(df))
    print("Valores ausentes por coluna:\n", df.isnull().sum())

#%% In[7]: Verificou-se a existência de lacunas nas datas

def verificar_gaps(df, nome):
    all_days = pd.date_range(start=df.index.min(), end=df.index.max(), freq='B')
    missing = all_days.difference(df.index)
    print(f"\n===== {nome} – Dias últis ausentes =====\n{len(missing)} dias faltando")
    if len(missing) > 0:
        print(missing[:10])

for nome, df in dados.items():
    verificar_gaps(df, nome)

#%% In[8]: Efetuou-se a leitura do consolidado com cabeçalho duplo e datas no índice
data = pd.read_csv("dados_consolidados.csv", sep=';', decimal=',', header=[0, 1], index_col=0, parse_dates=True)

# Leitura dos arquivos individuais com dois níveis de cabeçalho (campos e ticker)
itub4 = pd.read_csv("dados_bancarios.csv", sep=';', decimal=',', header=[0, 1], index_col=0, parse_dates=True)
mglu3 = pd.read_csv("dados_varejo.csv", sep=';', decimal=',', header=[0, 1], index_col=0, parse_dates=True)
petr4 = pd.read_csv("dados_petroleo.csv", sep=';', decimal=',', header=[0, 1], index_col=0, parse_dates=True)
vale3 = pd.read_csv("dados_mineracao.csv", sep=';', decimal=',', header=[0, 1], index_col=0, parse_dates=True)
        
#%% In[9]: Removeu-se o segundo nível das colunas

data.columns = data.columns.droplevel(1)
itub4.columns = itub4.columns.droplevel(1)
mglu3.columns = mglu3.columns.droplevel(1)
petr4.columns = petr4.columns.droplevel(1)
vale3.columns = vale3.columns.droplevel(1)

#%% In[10]: Buscou-se as estatísticas descritivas

colunas_base = ['Close', 'Open', 'High', 'Low', 'Volume']

for nome, df in dados.items():
    print(f"\n===== {nome} – Estatísticas Descritivas =====")
    print(df[colunas_base].describe().round(4))

#%% In[11]: Cria uma lista para armazenar estatísticas de cada ativo

estatisticas = []

# Itera sobre cada ativo e seu DataFrame correspondente
for nome, df in dados.items():
   
    # Calcula a média do Volume Negociado por dia, no período de 02/01/2015 a 30/05/2015
    mean_volume = df['Volume'].mean()
    
   # Calcula o desvio padrão do preço de fechamento (volatilidade)
    std_close = df['Close'].std()

    # Calcula a média do preço de fechamento
    mean_close = df['Close'].mean()

    # Calcula o Coeficiente de Variação do preço de fechamento (CV)
    # Garante que a média não é zero para evitar erro de divisão
    if mean_close != 0:
        cv_close = (std_close / mean_close) * 100
    else:
        cv_close = 0 # Define CV como 0 se a média for 0 (ou outro valor apropriado)

    # Adiciona todas as estatísticas calculadas para o ativo atual à lista
    estatisticas.append({
        'Ativo': nome, 
        'Volume_Mean': mean_volume,
        'Close_Std': std_close,
        'Close_Mean': mean_close,
        'Close_CV (%)': cv_close
    })

# Opcional: Para visualizar os resultados, você pode converter a lista em um DataFrame
df_stats = pd.DataFrame(estatisticas)
print(df_stats)

#%% In[12]: Plotou-se o gráfico de volatilidade dos preços de fechamento

plt.figure(figsize=(10, 6))
sns.barplot(data=df_stats, x='Ativo', y='Close_Std', palette='viridis')
plt.title('Volatilidade dos Preços de Fechamento (Desvio Padrão)')
plt.ylabel('Desvio Padrão (R$)')
plt.xlabel('Ativo')
plt.grid(axis='y')
plt.tight_layout()
plt.savefig('volatilidade_precos.png', dpi=300)
plt.show()

#%% In[13]: Plotou-se o gráfico de Volume Médio Negociado

plt.figure(figsize=(10, 6))
sns.barplot(data=df_stats, x='Ativo', y='Volume_Mean')
plt.title('Volume Médio Diário Negociado por Ativo')
plt.ylabel('Volume Médio (x 10 milhões)')
plt.xlabel('Ativo')
plt.tight_layout()
plt.savefig('volume_medio.png')

#%% In[14]: Plotou-se o gráfico de Coeficiente de Variação

# Criando o Gráfico de Barras do Coeficiente de Variação
plt.figure(figsize=(10, 6)) # Define o tamanho da figura (largura, altura)

# Cria o gráfico de barras
sns.barplot(x='Ativo', y='Close_CV (%)', data=df_stats, palette='viridis')

# Adiciona título e rótulos
plt.title('Coeficiente de Variação (CV) dos Preços de Fechamento por Ativo', fontsize=16)
plt.xlabel('Ativo', fontsize=12)
plt.ylabel('Coeficiente de Variação (%)', fontsize=12)

# Adiciona os valores do CV nas barras para melhor leitura
for index, row in df_stats.iterrows():
    plt.text(index, row['Close_CV (%)'] + 2, f"{row['Close_CV (%)']:.2f}%",
             color='black', ha="center", va='bottom', fontsize=10) # Ajuste +2 para posicionar o texto acima da barra

# Ajusta o layout para evitar cortes nos rótulos
plt.tight_layout()

# Exibe o gráfico
plt.show()

# Opcional: Salvar o gráfico em um arquivo
plt.savefig('coeficiente_variacao_ativos.png', dpi=300, bbox_inches='tight')

#%% In[15]: Estabeleceu-se a função para plotar gráficos por variável

def plotar_graficos(df, nome):
    fig, axs = plt.subplots(5, 1, figsize=(12, 10), sharex=True)
    cores = ['blue', 'green', 'red', 'purple', 'brown']
    titulos = ['Abertura', 'Mínima', 'Máxima', 'Fechamento', 'Volume']
    colunas = ['Open', 'Low', 'High', 'Close', 'Volume']

    for i, (col, cor, titulo) in enumerate(zip(colunas, cores, titulos)):
        axs[i].plot(df.index, df[col], label=titulo, color=cor)
        axs[i].set_ylabel(titulo)
        axs[i].legend(loc='upper left')
        axs[i].grid(True)
        if i == 0:
            axs[i].set_title(f"{nome} – Comportamento das Variáveis Temporais")
        if i == 4:
            axs[i].set_xlabel("Data")

    plt.tight_layout()
    plt.savefig(f"grafico_temporal_{nome.lower()}.png", dpi=300)
    plt.show()

#%% In[16]: Gerou-se os gráficos para os 4 ativos

for nome, df in dados.items():
    plotar_graficos(df, nome)

#%% In[17]: Calculou-se EMA e MACD para cada papel

def adicionar_ema_macd(df):
    df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['Sinal_MACD'] = df['MACD'].ewm(span=9, adjust=False).mean()
    return df

itub4 = adicionar_ema_macd(itub4)
mglu3 = adicionar_ema_macd(mglu3)
petr4 = adicionar_ema_macd(petr4)
vale3 = adicionar_ema_macd(vale3)
df = adicionar_ema_macd(df)

#%% In[18]: Aplicou-se a função auto_arima para identificação automática de parâmetros do modelo ARIMA (auto_arima)

def auto_arima_previsao(df, nome):
    try:
        print(f"\n===== {nome} – ARIMA automático com auto_arima() =====")

        serie = df['Close'].dropna()
        corte = int(len(serie) * 0.8)
        treino = serie[:corte]
        teste = serie[corte:]

        # Ajuste automático do modelo (sem componente sazonal)
        modelo = auto_arima(
            treino,
            seasonal=False,       # Não estamos tratando séries sazonais
            stepwise=True,
            suppress_warnings=True,
            error_action='ignore',
            trace=False
        )

        print(f"Melhor modelo identificado: {modelo.order}")

        # Previsão para o período de teste
        previsao = modelo.predict(n_periods=len(teste))

        # Avaliação do desempenho
        mae = mean_absolute_error(teste, previsao)
        rmse = np.sqrt(mean_squared_error(teste, previsao))

        print(f"MAE: {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")

        # Gráfico
        plt.figure(figsize=(10, 4))
        plt.plot(teste.index, teste, label='Valor Real', color='blue')
        plt.plot(teste.index, previsao, label='Previsão ARIMA Auto', linestyle='--', color='orange')
        plt.title(f"{nome} – Previsão com auto_arima {modelo.order}")
        plt.xlabel("Data")
        plt.ylabel("Preço de Fechamento")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Erro no modelo auto_arima para {nome}: {e}")

#%% In[19]: Aplicou-se a função auto_arima para cada papel:

auto_arima_previsao(itub4, "ITUB4")
auto_arima_previsao(mglu3, "MGLU3")
auto_arima_previsao(petr4, "PETR4")
auto_arima_previsao(vale3, "VALE3")

#%% In[20]: Função para testar múltiplas combinações de hiperparâmetros

def testar_parametros_rf(df, nome_ativo):
    features = ['EMA12', 'MACD', 'Sinal_MACD', 'Open', 'High', 'Low', 'Volume']
    target = 'Close'
    df = df.dropna()

    X = df[features]
    y = df[target]
    datas = df.index

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    split_index = int(0.8 * len(df))
    X_train, X_test = X_scaled[:split_index], X_scaled[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    datas_teste = datas[split_index:]

    # Lista de combinações a serem testadas
    combinacoes = [
        {'n_estimators': 100, 'max_depth': 5, 'min_samples_split': 2},
        {'n_estimators': 100, 'max_depth': 10, 'min_samples_split': 2},
        {'n_estimators': 100, 'max_depth': 15, 'min_samples_split': 2},   
        {'n_estimators': 100, 'max_depth': None, 'min_samples_split': 2},        
        {'n_estimators': 100, 'max_depth': 5, 'min_samples_split': 5},
        {'n_estimators': 100, 'max_depth': 10, 'min_samples_split': 5},
        {'n_estimators': 100, 'max_depth': 15, 'min_samples_split': 5},   
        {'n_estimators': 100, 'max_depth': None, 'min_samples_split': 5},     
        {'n_estimators': 100, 'max_depth': 5, 'min_samples_split': 10},
        {'n_estimators': 100, 'max_depth': 10, 'min_samples_split': 10},
        {'n_estimators': 100, 'max_depth': 15, 'min_samples_split': 10},  
        {'n_estimators': 100, 'max_depth': None, 'min_samples_split': 10}, 
        {'n_estimators': 200, 'max_depth': 5, 'min_samples_split': 2},
        {'n_estimators': 200, 'max_depth': 10, 'min_samples_split': 2},
        {'n_estimators': 200, 'max_depth': 15, 'min_samples_split': 2},   
        {'n_estimators': 200, 'max_depth': None, 'min_samples_split': 2},  
        {'n_estimators': 200, 'max_depth': 5, 'min_samples_split': 5},
        {'n_estimators': 200, 'max_depth': 10, 'min_samples_split': 5},
        {'n_estimators': 200, 'max_depth': 15, 'min_samples_split': 5},   
        {'n_estimators': 200, 'max_depth': None, 'min_samples_split': 5},  
        {'n_estimators': 200, 'max_depth': 5, 'min_samples_split': 10},
        {'n_estimators': 200, 'max_depth': 10, 'min_samples_split': 10},
        {'n_estimators': 200, 'max_depth': 15, 'min_samples_split': 10},  
        {'n_estimators': 200, 'max_depth': None, 'min_samples_split': 10},  
        {'n_estimators': 300, 'max_depth': 5, 'min_samples_split': 2},
        {'n_estimators': 300, 'max_depth': 10, 'min_samples_split': 2},
        {'n_estimators': 300, 'max_depth': 15, 'min_samples_split': 2},   
        {'n_estimators': 300, 'max_depth': None, 'min_samples_split': 2},  
        {'n_estimators': 300, 'max_depth': 5, 'min_samples_split': 5},
        {'n_estimators': 300, 'max_depth': 10, 'min_samples_split': 5},
        {'n_estimators': 300, 'max_depth': 15, 'min_samples_split': 5},   
        {'n_estimators': 300, 'max_depth': None, 'min_samples_split': 5},  
        {'n_estimators': 300, 'max_depth': 5, 'min_samples_split': 10},
        {'n_estimators': 300, 'max_depth': 10, 'min_samples_split': 10},
        {'n_estimators': 300, 'max_depth': 15, 'min_samples_split': 10},   
        {'n_estimators': 300, 'max_depth': None, 'min_samples_split': 10}, 
        {'n_estimators': 500, 'max_depth': 5, 'min_samples_split': 5},
        {'n_estimators': 500, 'max_depth': 10, 'min_samples_split': 5},
        {'n_estimators': 500, 'max_depth': 15, 'min_samples_split': 5},
        {'n_estimators': 500, 'max_depth': None, 'min_samples_split': 5},
    ]

    resultados = []

    for i, params in enumerate(combinacoes, 1):
        model = RandomForestRegressor(
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            min_samples_split=params['min_samples_split'],
            min_samples_leaf=1,       # define mínimo de amostras por folha
            max_features=None,      # define o número de variáveis consideradas por divisão        
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)

        resultados.append({
            'Configuração': f"Modelo {i}",
            'n_estimators': params['n_estimators'],
            'max_depth': params['max_depth'],
            'min_samples_split': params['min_samples_split'],
            'RMSE': round(rmse, 4),
            'MAE': round(mae, 4)
        })

        print(f"Modelo {i}: RMSE = {rmse:.4f} | MAE = {mae:.4f}")

        # Gráfico de linha Real vs Previsto
        plt.figure(figsize=(10, 4))
        plt.plot(datas_teste, y_test, label='Preço Real', color='blue')
        plt.plot(datas_teste, y_pred, label='Previsão RF', linestyle='--', color='orange')
        plt.title(f"{nome_ativo} – Previsão com {params}")
        plt.xlabel("Data")
        plt.ylabel("Preço de Fechamento (R$)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'previsao_{nome_ativo.lower()}_modelo_{i}.png', dpi=300)
        plt.show()

    return pd.DataFrame(resultados)

#%% In[21]: Recebe um dicionário com os DataFrames de resultados por ativo e imprime os melhores modelos (menor RMSE) e seus hiperparâmetros.
    
def identificar_melhor_modelo_por_ativo(resultados_dict):
    for nome_ativo, df_resultados in resultados_dict.items():
        melhor_modelo = df_resultados.loc[df_resultados['RMSE'].idxmin()]
        print(f"\nMelhor modelo para {nome_ativo}:")
        print(f"  Modelo: {melhor_modelo['Configuração']}")
        print(f"  n_estimators: {melhor_modelo['n_estimators']}")
        print(f"  max_depth: {melhor_modelo['max_depth']}")
        print(f"  min_samples_split: {melhor_modelo['min_samples_split']}")
        print(f"  RMSE: {melhor_modelo['RMSE']}")
        print(f"  MAE: {melhor_modelo['MAE']}")

#%% ln[22]: Dicionário com os resultados dos quatro ativos

resultados_itub4_rf = testar_parametros_rf(itub4, "ITUB4")
resultados_mglu3_rf = testar_parametros_rf(mglu3, "MGLU3")
resultados_petr4_rf = testar_parametros_rf(petr4, "PETR4")
resultados_vale3_rf = testar_parametros_rf(vale3, "VALE3")

#%% ln[23]: Armazenou-se os resultados dos quatro ativos

resultados_rf = {
    'ITUB4': resultados_itub4_rf,
    'MGLU3': resultados_mglu3_rf,
    'PETR4': resultados_petr4_rf,
    'VALE3': resultados_vale3_rf
}

#%% ln[24]: Identificou-se o melhor modelo para cada Ativo

identificar_melhor_modelo_por_ativo(resultados_rf)

#%% ln[25]: Definiu-se a função para gerar apenas o gráfico do melhor modelo de RF para cada Ativo

def plotar_melhor_modelo(df, nome_ativo, df_resultados):
    
    # Identifica melhor modelo (menor RMSE)
    melhor = df_resultados.loc[df_resultados['RMSE'].idxmin()]
    n_estimators = int(melhor['n_estimators'])
    max_depth = None if pd.isna(melhor['max_depth']) else int(melhor['max_depth'])
    min_samples_split = int(melhor['min_samples_split'])

    print(f"\n{nome_ativo} – Melhor Modelo:")
    print(f"  n_estimators: {n_estimators}")
    print(f"  max_depth: {max_depth}")
    print(f"  min_samples_split: {min_samples_split}")

    # Prepara os dados
    df = df.dropna()
    features = ['EMA12', 'MACD', 'Sinal_MACD', 'Open', 'High', 'Low', 'Volume']
    target = 'Close'
    X = df[features]
    y = df[target]
    datas = df.index

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    split_index = int(0.8 * len(df))
    X_train, X_test = X_scaled[:split_index], X_scaled[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    datas_teste = datas[split_index:]

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=1,
        max_features=None,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    # Gera o gráfico
    plt.figure(figsize=(12, 5))
    plt.plot(datas_teste, y_test, label='Preço Real', color='blue', linewidth=2)
    plt.plot(datas_teste, y_pred, label='Previsão RF', linestyle='--', color='orange', linewidth=2)
    plt.title(f'{nome_ativo} – Previsão com Melhor Modelo RF\nRMSE: {rmse:.4f} | MAE: {mae:.4f}')
    plt.xlabel("Data")
    plt.ylabel("Preço de Fechamento (R$)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'melhor_rf_{nome_ativo.lower()}.png', dpi=300)
    plt.show()

#%% ln [26]: Executou-se a plotagem dos gráficos de melhor modelo para os 4 ativos
    
plotar_melhor_modelo(itub4, 'ITUB4', resultados_itub4_rf)
plotar_melhor_modelo(mglu3, 'MGLU3', resultados_mglu3_rf)
plotar_melhor_modelo(petr4, 'PETR4', resultados_petr4_rf)
plotar_melhor_modelo(vale3, 'VALE3', resultados_vale3_rf)

#%%  In[27]: Registrou-se os dados de desempenho do RF e construiu-se 
# os gráficos comparativos das métricas de erro por Ativo

resultados_rf = pd.DataFrame({
    'Ativo': ['ITUB4', 'MGLU3', 'PETR4', 'VALE3'],
    'Setor': ['Financeiro', 'Varejo', 'Petróleo e Gás', 'Materiais Básicos'],
    'RMSE': [1.2539, 0.4704, 10.4532, 0.3902],
    'MAE': [0.5394, 0.3631, 9.3375, 0.3053]
})

# Gráfico de RMSE
plt.figure(figsize=(8, 5))
sns.barplot(data=resultados_rf, x='Ativo', y='RMSE', palette='magma')
plt.title('Erro Quadrático Médio (RMSE) por Ativo – Random Forest')
plt.ylabel('RMSE')
plt.xlabel('Ativo')
plt.grid(axis='y')
plt.tight_layout()
plt.savefig('grafico_rf_rmse.png', dpi=300)
plt.show()

# Gráfico de MAE
plt.figure(figsize=(8, 5))
sns.barplot(data=resultados_rf, x='Ativo', y='MAE', palette='viridis')
plt.title('Erro Médio Absoluto (MAE) por Ativo – Random Forest')
plt.ylabel('MAE')
plt.xlabel('Ativo')
plt.grid(axis='y')
plt.tight_layout()
plt.savefig('grafico_rf_mae.png', dpi=300)
plt.show()

#%% In[28]: Definiu-se a função de preparação da série temporal para LSTM

def preparar_dados_sequenciais(df, feature_col='Close', n_past=60):
    series = df[feature_col].dropna().values.reshape(-1, 1)
    scaler = MinMaxScaler()
    series_scaled = scaler.fit_transform(series)

    X, y = [], []
    for i in range(n_past, len(series_scaled)):
        X.append(series_scaled[i - n_past:i, 0])
        y.append(series_scaled[i, 0])
    
    X, y = np.array(X), np.array(y)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    return X_train, X_test, y_train, y_test, scaler

#%% In[29]: Definiu-se a função para rodar NRAs - modelo LSTM e GRU

def treinar_modelo_rna(nome_ativo, df, tipo='LSTM', n_past=60, units=50, epochs=50, batch_size=32):
    """
    Treina um modelo recorrente (LSTM ou GRU) para previsão de séries temporais univariadas (Close).
    Retorna: y_test_inv, y_pred_inv, rmse, mae
    """
    X_train, X_test, y_train, y_test, scaler = preparar_dados_sequenciais(df, n_past=n_past)

    model = Sequential()
    if tipo == 'LSTM':
        model.add(LSTM(units=units, activation='tanh', return_sequences=False, input_shape=(n_past, 1)))
    elif tipo == 'GRU':
        model.add(GRU(units=units, activation='tanh', return_sequences=False, input_shape=(n_past, 1)))
    else:
        raise ValueError("Tipo deve ser 'LSTM' ou 'GRU'")

    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

    y_pred = model.predict(X_test)
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
    y_pred_inv = scaler.inverse_transform(y_pred)

    rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
    mae = mean_absolute_error(y_test_inv, y_pred_inv)

    return y_test_inv, y_pred_inv, rmse, mae

#%% In[30]: Definiu-se a função para plotar LSTM

def plot_real_vs_pred(y_real, y_pred, title, ativo, modelo, rmse=None, mae=None):
    plt.figure(figsize=(10, 4))
    plt.plot(y_real, label='Real', linewidth=2)
    plt.plot(y_pred, label='Previsto', linestyle='--')

    titulo = f'{ativo} – Real vs Previsto – {modelo}'
    if rmse is not None and mae is not None:
        titulo += f'\nRMSE: {rmse:.4f} | MAE: {mae:.4f}'
    
    plt.title(titulo, fontsize=12)
    plt.xlabel('Dias de Teste')
    plt.ylabel('Preço de Fechamento (R$)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'plot_{ativo.lower()}_{modelo.lower()}_real_vs_previsto.png', dpi=300)
    plt.show()
    
#%% In[31]: Rodou-se os modelos LSTM para cada ativo

# Resultados LSTM
print("\n===== Resultados com LSTM =====")
res_lstm_itub4 = treinar_modelo_rna("ITUB4", itub4, tipo='LSTM')
y_test_itub4_lstm, y_pred_itub4_lstm, rmse_itub4_lstm, mae_itub4_lstm = treinar_modelo_rna("ITUB4", itub4, tipo='LSTM')
y_test_mglu3_lstm, y_pred_mglu3_lstm, rmse_mglu3_lstm, mae_mglu3_lstm = treinar_modelo_rna("MGLU3", mglu3, tipo='LSTM')
y_test_petr4_lstm, y_pred_petr4_lstm, rmse_petr4_lstm, mae_petr4_lstm = treinar_modelo_rna("PETR4", petr4, tipo='LSTM')
y_test_vale3_lstm, y_pred_vale3_lstm, rmse_vale3_lstm, mae_vale3_lstm = treinar_modelo_rna("VALE3", vale3, tipo='LSTM')

#%% In[32]: Plotou-se o melhor gráfico LSTM por Ativo
# ITUB4
plot_real_vs_pred(y_test_itub4_lstm, y_pred_itub4_lstm, 'ITUB4', 'ITUB4', 'LSTM', rmse=rmse_itub4_lstm, mae=mae_itub4_lstm)

# MGLU3
plot_real_vs_pred(y_test_mglu3_lstm, y_pred_mglu3_lstm, 'MGLU3', 'MGLU3', 'LSTM', rmse=rmse_mglu3_lstm, mae=mae_mglu3_lstm)

# PETR4
plot_real_vs_pred(y_test_petr4_lstm, y_pred_petr4_lstm, 'PETR4', 'PETR4', 'LSTM', rmse=rmse_petr4_lstm, mae=mae_petr4_lstm)

# VALE3
plot_real_vs_pred(y_test_vale3_lstm, y_pred_vale3_lstm, 'VALE3', 'VALE3', 'LSTM', rmse=rmse_vale3_lstm, mae=mae_vale3_lstm)

#%% In[33] Dados de desempenho da LSTM

resultados_lstm = pd.DataFrame({
    'Ativo': ['ITUB4', 'MGLU3', 'PETR4', 'VALE3'],
    'Setor': ['Financeiro', 'Varejo', 'Petróleo e Gás', 'Materiais Básicos'],
    'RMSE': [rmse_itub4_lstm, rmse_mglu3_lstm, rmse_petr4_lstm, rmse_vale3_lstm],
    'MAE': [mae_itub4_lstm, mae_mglu3_lstm, mae_petr4_lstm, mae_vale3_lstm]
})

#%% In[34] Gráfico de RMSE (LSTM)

plt.figure(figsize=(8, 5))
sns.barplot(data=resultados_lstm, x='Ativo', y='RMSE', palette='magma')
plt.title('Erro Quadrático Médio (RMSE) por Ativo – LSTM')
plt.ylabel('RMSE')
plt.xlabel('Ativo')
plt.grid(axis='y')
plt.tight_layout()
plt.savefig('grafico_lstm_rmse.png', dpi=300)
plt.show()

#%% In [35] Gráfico de MAE (LSTM)

plt.figure(figsize=(8, 5))
sns.barplot(data=resultados_lstm, x='Ativo', y='MAE', palette='viridis')
plt.title('Erro Médio Absoluto (MAE) por Ativo – LSTM')
plt.ylabel('MAE')
plt.xlabel('Ativo')
plt.grid(axis='y')
plt.tight_layout()
plt.savefig('grafico_lstm_mae.png', dpi=300)
plt.show()

#%% In [36] Treinamento GRU por ativo

print("\n===== Resultados com GRU =====")
y_test_itub4_gru,  y_pred_itub4_gru,  rmse_itub4_gru,  mae_itub4_gru  = treinar_modelo_rna("ITUB4", itub4,  tipo='GRU')
y_test_mglu3_gru,  y_pred_mglu3_gru,  rmse_mglu3_gru,  mae_mglu3_gru  = treinar_modelo_rna("MGLU3", mglu3,  tipo='GRU')
y_test_petr4_gru,  y_pred_petr4_gru,  rmse_petr4_gru,  mae_petr4_gru  = treinar_modelo_rna("PETR4", petr4,  tipo='GRU')
y_test_vale3_gru,  y_pred_vale3_gru,  rmse_vale3_gru,  mae_vale3_gru  = treinar_modelo_rna("VALE3", vale3,  tipo='GRU')

#%% In[37] Gráficos Real vs Previsto (GRU)

# ITUB4
plot_real_vs_pred(y_test_itub4_gru, y_pred_itub4_gru, 'ITUB4', 'ITUB4', 'GRU', rmse=rmse_itub4_gru, mae=mae_itub4_gru)

# MGLU3
plot_real_vs_pred(y_test_mglu3_gru, y_pred_mglu3_gru, 'MGLU3', 'MGLU3', 'GRU', rmse=rmse_mglu3_gru, mae=mae_mglu3_gru)

# PETR4
plot_real_vs_pred(y_test_petr4_gru, y_pred_petr4_gru, 'PETR4', 'PETR4', 'GRU', rmse=rmse_petr4_gru, mae=mae_petr4_gru)

# VALE3
plot_real_vs_pred(y_test_vale3_gru, y_pred_vale3_gru, 'VALE3', 'VALE3', 'GRU', rmse=rmse_vale3_gru, mae=mae_vale3_gru)

#%% In[38] Tabela de desempenho (GRU)

resultados_gru = pd.DataFrame({
    'Ativo': ['ITUB4', 'MGLU3', 'PETR4', 'VALE3'],
    'Setor': ['Financeiro', 'Varejo', 'Petróleo e Gás', 'Materiais Básicos'],
    'RMSE': [rmse_itub4_gru, rmse_mglu3_gru, rmse_petr4_gru, rmse_vale3_gru],
    'MAE' : [mae_itub4_gru, mae_mglu3_gru, mae_petr4_gru, mae_vale3_gru]
})
print(resultados_gru.round(4))

#%% In[39] Gráfico RMSE (GRU)

plt.figure(figsize=(8,5))
sns.barplot(data=resultados_gru, x='Ativo', y='RMSE', palette='magma')
plt.title('Erro Quadrático Médio (RMSE) por Ativo – GRU')
plt.ylabel('RMSE'); plt.xlabel('Ativo')
plt.grid(axis='y'); plt.tight_layout()
plt.savefig('grafico_gru_rmse.png', dpi=300)
plt.show()

#%% In[40] Gráfico MAE (GRU)

plt.figure(figsize=(8,5))
sns.barplot(data=resultados_gru, x='Ativo', y='MAE', palette='viridis')
plt.title('Erro Médio Absoluto (MAE) por Ativo – GRU')
plt.ylabel('MAE'); plt.xlabel('Ativo')
plt.grid(axis='y'); plt.tight_layout()
plt.savefig('grafico_gru_mae.png', dpi=300)
plt.show()

#%% #%% In[41] Comparativo entre as métricas de erro dos modelos LSTM vs GRU por ativo 
# gráfico único contrastando as duas arquiteturas.

# Garante que resultados_lstm já existe
comparativo = pd.concat([
    resultados_lstm.assign(Modelo='LSTM'),
    resultados_gru.assign(Modelo='GRU')
], ignore_index=True)

# RMSE comparativo
plt.figure(figsize=(9,5))
sns.barplot(data=comparativo, x='Ativo', y='RMSE', hue='Modelo')
plt.title('RMSE por Ativo – LSTM vs GRU')
plt.ylabel('RMSE'); plt.xlabel('Ativo')
plt.grid(axis='y'); plt.tight_layout()
plt.savefig('comparativo_rmse_lstm_gru.png', dpi=300)
plt.show()

# MAE comparativo
plt.figure(figsize=(9,5))
sns.barplot(data=comparativo, x='Ativo', y='MAE', hue='Modelo')
plt.title('MAE por Ativo – LSTM vs GRU')
plt.ylabel('MAE'); plt.xlabel('Ativo')
plt.grid(axis='y'); plt.tight_layout()
plt.savefig('comparativo_mae_lstm_gru.png', dpi=300)
plt.show()

###############################################################################
###############################################################################
#                               *** F I M ***
###############################################################################
