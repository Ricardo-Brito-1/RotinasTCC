###############################################################################
# Universidade de São Paulo [USP]
# Escola Superior de Agricultura Luiz de Queiroz [Esalq]
# Campus da USP em Piracicaba/SP

# Curso: MBA em Data Science e Analytics - EAD

# Trabalho de Conclusão de Curso - TCC
# Predição de Ações B3: Comparativo setorial com Random Forest [RF] e Redes Neurais [RNAs]

# O objetivo deste trabalho é desenvolver modelos de Machine Learning [ML] — RF e RNAs — para
# predizer o preço de ações de quatro setores distintos da B3, utilizando dados históricos do
# período de 02/01/2015 a 30/05/2025, coletados na  Yahoo Finance API e  no site da B3, a fim 
# de comparar o desempenho dos modelos com base em métricas de erro e avaliar a adaptabilidade
# de cada um aos diferentes setores.

# Ações: ITUB4, MGLU3, PETR4 e VALE3

###############################################################################

#%% In[1]: Importou-se bibliotecas necessárias para a análise, instalou-se o yfinance e 
# buscou-se os dados para início do TCC

# Busca dos dados no yfinance
import yfinance as yf

# Manipulação e análise de dados
import pandas as pd
import numpy as np
from itertools import product

# Visualização
import matplotlib.pyplot as plt
import seaborn as sns

# Trabalho com datas e frequências (Séries Temporais)
from datetime import datetime
import matplotlib.dates as mdates

# Para estatísticas de séries temporais
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.stats import kstest

# Para modelagem com ARIMA dos 4 papéis (ITUB4, MGLU3, PETR4 e VALE3)
# Biblioteca para função auto_arima - Identificação automática para modelo ARIMA
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Para escalonamento e divisão dos dados
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

# Para o modelo de ML Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor

# Para os modelos modelos LSTM e GRU
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU
from tensorflow.keras.optimizers import Adam

#%% In[2]: Definiu-se os ativos (tickers) e o intervalo de datas, de inicio e de término.

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
    df.columns = df.columns.droplevel(1)  # removeu-se um dos níveis do cabeçalho
    df.index = pd.to_datetime(df.index)   # garante que seja datetime
    dados[nome] = df

#%% In[6]: Verificou-se estrutura e consistência dos dados

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

#%% In[8]: Buscou-se um resuno das estatísticas descritivas

colunas_base = ['Close', 'Open', 'High', 'Low', 'Volume']

for nome, df in dados.items():
    print(f"\n===== {nome} – Estatísticas Descritivas =====")
    print(df[colunas_base].describe().round(2))

#%% In[9]: Criou-se uma lista para armazenar as estatísticas média e desvio-padrão de
# cada um dos ativos

estatisticas = []

# Iteração de cada ativo e seu DataFrame correspondente
for nome, df in dados.items():
   
    # Calculou-se a média do Volume Negociado por dia, no período de 02/01/2015 a 30/05/2015
    mean_volume = df['Volume'].mean()
    
   # Calculou-se o desvio padrão do preço de fechamento (volatilidade)
    std_close = df['Close'].std()

    # Calculou-se a média do preço de fechamento
    mean_close = df['Close'].mean()

    # Calculou-se o Coeficiente de Variação do preço de fechamento (CV)
    # Garantiu-se que a média não seja zero para evitar erro de divisão
    if mean_close != 0:
        cv_close = (std_close / mean_close) * 100
    else:
        cv_close = 0 # Define CV como 0 se a média for 0 (ou outro valor apropriado)

    # Adicionou-se todas as estatísticas calculadas para o ativo atual à lista
    estatisticas.append({
        'Ativo': nome, 
        'Volume_Mean': mean_volume,
        'Close_Std': std_close,
        'Close_Mean': mean_close,
        'Close_CV (%)': cv_close
    })

# Opcional: Para visualizar os resultados, pode-se converter a lista em um DataFrame
df_stats = pd.DataFrame(estatisticas)
print(df_stats)

#%% In[10]: Plotou-se o gráfico de volatilidade dos preços de fechamento dos ativos

plt.figure(figsize=(10, 6))
sns.barplot(data=df_stats, x='Ativo', y='Close_Std', palette='viridis')
plt.title('Volatilidade dos Preços de Fechamento (Desvio Padrão)')
plt.ylabel('Desvio Padrão (R$)')
plt.xlabel('Ativo')
plt.grid(axis='y')
plt.tight_layout()
plt.savefig('volatilidade_precos.png', dpi=300)
plt.show()

#%% In[11]: Plotou-se o gráfico de Volume Médio Negociado dos ativos

plt.figure(figsize=(10, 6))
sns.barplot(data=df_stats, x='Ativo', y='Volume_Mean')
plt.title('Volume Médio Diário Negociado por Ativo')
plt.ylabel('Volume Médio (x 10 milhões)')
plt.xlabel('Ativo')
plt.tight_layout()
plt.savefig('volume_medio.png')
plt.show()

#%% In[12]: Plotou-se o gráfico do Coeficiente de Variação por ativo

# Criou-se o Gráfico de Barras do Coeficiente de Variação
plt.figure(figsize=(10, 6)) # Define o tamanho da figura (largura, altura)

# Criou-se o gráfico de barras
sns.barplot(x='Ativo', y='Close_CV (%)', data=df_stats, palette='viridis')

# Adicionou-se título e rótulos
plt.title('Coeficiente de Variação (CV) dos Preços de Fechamento por Ativo', fontsize=16)
plt.xlabel('Ativo', fontsize=12)
plt.ylabel('Coeficiente de Variação (%)', fontsize=12)

# Adicionou-se os valores do CV nas barras para melhor leitura
for index, row in df_stats.iterrows():
    plt.text(index, row['Close_CV (%)'] + 2, f"{row['Close_CV (%)']:.2f}%",
             color='black', ha="center", va='bottom', fontsize=10) # Ajuste +2 para posicionar o texto acima da barra

# Ajustou-se o layout para evitar cortes nos rótulos
plt.tight_layout()

# Opcional: Salvar o gráfico em um arquivo
plt.savefig('coeficiente_variacao_ativos.png', dpi=300, bbox_inches='tight')

# Exibiu-se o gráfico do Coeficiente de Variação por ativo
plt.show()

#%% In[13]: Estabeleceu-se a função para plotar gráficos por variável - Preços de
# Abertura, Mínima, Máxima, Fechamento e Volume, para cada um dos ativos

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

#%% In[14]: Gerou-se os gráficos para os quatro ativos

for nome, df in dados.items():
    plotar_graficos(df, nome)

#%% In[15]: Calculou-se EMA e MACD para cada um dos ativos

def adicionar_ema_macd(df):
    df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['Sinal_MACD'] = df['MACD'].ewm(span=9, adjust=False).mean()
    return df

# 1. Aplicou-se a função no dicionário 'dados', onde os DataFrames estão armazenados
for nome, df_ativo in dados.items():
    # Sobrescreveu-se a entrada no dicionário com o DataFrame que tem os novos indicadores
    dados[nome] = adicionar_ema_macd(df_ativo)

# 2. Extraiu-se os DataFrames do dicionário para as variáveis individuais que o script
# utilizará a partir de In[16].

itub4 = dados['ITUB4']
mglu3 = dados['MGLU3']
petr4 = dados['PETR4']
vale3 = dados['VALE3']

#%%##################################################################################
# Verificação dos modelos ARIMA(p,d q) 
#####################################################################################
#% In[16]: Checou-se o Tipo de Dados das séries do ativos

# 1. Checou-se o tipo de dados do índice MGLU3
if pd.api.types.is_datetime64_any_dtype(mglu3.index):
    print("Para o ativo MGLU3, o tipo de dados do índice está correto (datetime).")
else:
    print("ERRO: Para o ativo MGLU3, o tipo de dados do índice não é datetime. Converta-o com pd.to_datetime().") 
  
# 2. Checou-se o tipo de dados do índice ITUB4
if pd.api.types.is_datetime64_any_dtype(itub4.index):
    print("Para o ativo ITUB4, o tipo de dados do índice está correto (datetime).")
else:
    print("ERRO: Para o ativo ITUB4, o tipo de dados do índice não é datetime. Converta-o com pd.to_datetime().")  
        
# 3. Checou-se o tipo de dados do índice PETR4
if pd.api.types.is_datetime64_any_dtype(petr4.index):
    print("Para o ativo PETR4, o tipo de dados do índice está correto (datetime).")
else:
    print("ERRO: Para o ativo PETR4, o tipo de dados do índice não é datetime. Converta-o com pd.to_datetime().")
        
# 4. Checou-se o tipo de dados do índice VALE3
if pd.api.types.is_datetime64_any_dtype(vale3.index):
    print("Para o ativo VALE3, o tipo de dados do índice está correto (datetime).")
else:
    print("ERRO: Para o ativo VALE3, o tipo de dados do índice não é datetime. Converta-o com pd.to_datetime().")
    
#%%  In[17]: Definiu-se a frequência 'B' - frequência de "dias úteis de negócio" 
# 'B' - Bunisses day - e preencheu-se as lacunas

# Reamostrou-se os dados para uma frequência de dias úteis ('B')
# Preencheu-se os valores NaN (dias sem negociação) com o valor anterior
mglu3 = mglu3.asfreq('B', method='pad')
itub4 = itub4.asfreq('B', method='pad')
petr4 = petr4.asfreq('B', method='pad')
vale3 = vale3.asfreq('B', method='pad')

# Verificou-se a frequência e os NaNs foram definidos corretamente
print(f"Nova frequência de MGLU3: {mglu3.index.freq}")
print(f"Novos valores ausentes em MGLU3: {mglu3.isnull().sum()}")

print(f"Nova frequência de ITUB4: {itub4.index.freq}")
print(f"Novos valores ausentes em ITUB4: {itub4.isnull().sum()}")

print(f"Nova frequência de PETR4: {petr4.index.freq}")
print(f"Novos valores ausentes em PETR4: {petr4.isnull().sum()}")

print(f"Nova frequência de VALE3: {vale3.index.freq}")
print(f"Novos valores ausentes em VALE3: {vale3.isnull().sum()}")

# In[18]: Plotou-se as séries Preço de Fechamento (Close) dos Ativos

# Plotou-se a série de preços de fechamento [Close] do ativo MGLU3
plt.figure(figsize=(10, 6))
plt.plot(mglu3['Close'], label='Preço de Fechamento da Ação MGLU3')
plt.title("Preço de Fechamento diário da Ação MGLU3")
plt.xlabel('Data')
plt.ylabel('Preço (R$)')
plt.grid(True)
plt.savefig('fechamento_diario_mglu3.png')
plt.show()

# Plotou-se a série de preços de fechamento [Close] do ativo ITUB4
plt.figure(figsize=(10, 6))
plt.plot(itub4['Close'], label='Preço de Fechamento da Ação MGLU3')
plt.title("Preço de Fechamento diário da Ação ITUB4")
plt.xlabel('Data')
plt.ylabel('Preço (R$)')
plt.grid(True)
plt.savefig('fechamento_diario_itub4.png')
plt.show()

# Plotou-se a série de preços de fechamento [Close] do ativo PETR4
plt.figure(figsize=(10, 6))
plt.plot(petr4['Close'], label='Preço de Fechamento da Ação PETR4')
plt.title("Preço de Fechamento diário da Ação PETR4")
plt.xlabel('Data')
plt.ylabel('Preço (R$)')
plt.grid(True)
plt.savefig('fechamento_diario_petr4.png')
plt.show()

# Plotou-se a série de preços de fechamento [Close] do ativo VALE3
plt.figure(figsize=(10, 6))
plt.plot(vale3['Close'], label='Preço de Fechamento da Ação VALE3')
plt.title("Preço de Fechamento diário da Ação VALE3")
plt.xlabel('Data')
plt.ylabel('Preço (R$)')
plt.grid(True)
plt.savefig('fechamento_diario_vale3.png')
plt.show()

#%% In[19]: Dividiu-se as séries em treino e teste

# Dividiu-se a série do ativo MGLU3 em treino e teste
mglu3treino = mglu3[:'2023-04-23']
mglu3teste = mglu3['2023-04-24':]

# Dividiu-se a série do ativo ITUB4 em treino e teste
itub4treino = itub4[:'2023-04-23']
itub4teste = itub4['2023-04-24':]

# Dividiu-se a série do ativo em treino e teste
petr4treino = petr4[:'2023-04-23']
petr4teste = petr4['2023-04-24':]

# Dividiu-se a série do ativo VALE3 em treino e teste
vale3treino = vale3[:'2023-04-23']
vale3teste = vale3['2023-04-24':]

# Checou-se do tamanho do conjunto de teste
print(f"Comprimento da série de teste MGLU3: {len(mglu3teste)}")
print(f"Comprimento da série de teste ITUB4: {len(itub4teste)}")
print(f"Comprimento da série de teste PETR4: {len(petr4teste)}")
print(f"Comprimento da série de teste VALE3: {len(vale3teste)}")

#%% In[20]: Testou-se a Estacionariedade das Séries de todos os ativos - ADF (Dickey-Fuller)

from statsmodels.tsa.stattools import adfuller

# Definiu-se a função para o teste de estacionariedade
def teste_estacionariedade_adf(df_serie, nome):
    print(f'\nResultados do Teste ADF para {nome}')
   
   # O teste é aplicado na coluna 'Close', que é o preço de fechamento das ações
   # --- Nível (sem diferenciação) ---
    resultado_nivel = adfuller(df_serie['Close'].dropna(), autolag='AIC')
    print("\n>> Nível")
    print('Estatística de Teste ADF:', round(resultado_nivel[0], 4))
    print('Valor-p (p-value):', round(resultado_nivel[1], 4))
    print('Lags usados:', resultado_nivel[2])
    print('Nobs:', resultado_nivel[3])
    print('Valores Críticos:', resultado_nivel[4])
    if resultado_nivel[1] <= 0.05:
        print("Decisão: Rejeita H0 (série estacionária).")
    else:
        print("Decisão: Não rejeita H0 (série não estacionária).")

    # --- Primeira diferença (Δ1) ---
    serie_diff = df_serie['Close'].diff().dropna()
    resultado_diff = adfuller(serie_diff, autolag='AIC')
    print("\n>> Primeira Diferença (Δ1)")
    print('Estatística de Teste ADF:', round(resultado_diff[0], 4))
    print('Valor-p (p-value):', round(resultado_diff[1], 4))
    print('Lags usados:', resultado_diff[2])
    print('Nobs:', resultado_diff[3])
    print('Valores Críticos:', resultado_diff[4])
    if resultado_diff[1] <= 0.05:
        print("Decisão: Rejeita H0 (série estacionária).")
    else:
        print("Decisão: Não rejeita H0 (série não estacionária).")
        
# Aplicou-se o teste SOMENTE nas séries de TREINO dos ativos
teste_estacionariedade_adf(mglu3treino, 'MGLU3 (Treino)')
teste_estacionariedade_adf(itub4treino, 'ITUB4 (Treino)')
teste_estacionariedade_adf(petr4treino, 'PETR4 (Treino)')
teste_estacionariedade_adf(vale3treino, 'VALE3 (Treino)')

#%% In[21]: Tornou-se as séries dos ativos estacionárias e encontrou-se o parâmetro 'd'

# É importante certificar-se de que "adfuller" foi importado
# from statsmodels.tsa.stattools import adfuller

def fazer_estacionaria(series, nome_serie, max_diff=2):
    """
    Diferencia uma série temporal repetidamente até que ela se torne estacionária (com base no teste ADF).
    Retorna a série diferenciada e o número de diferenciações aplicadas (parâmetro 'd').
    """
    diff_count = 0
    # A série já é a coluna 'Close'
    df_temp = series.copy() 
    
    print(f"\nVerificando estacionariedade de: {nome_serie}")
    
    while diff_count < max_diff:
        resultado_adf = adfuller(df_temp) 
        p_valor = resultado_adf[1]
        
        # Verificou-se para saber se a série já é estacionária
        if p_valor <= 0.05:
            print(f"Série estacionária com {diff_count} diferenciação(ões). p-valor = {p_valor:.4f}")
            # Retorna a série diferenciada (se diff_count > 0) e o d
            return df_temp, diff_count
        
        # Aplicou-se a diferenciação - somente nos casos em que a série não é estacionária
        df_temp = df_temp.diff().dropna()
        diff_count += 1
        
    # Definiu-se a Mensagem para os casos em que a série não se torne estacionária em 'max_diff'
    print(f"Não foi possível tornar a série estacionária após {max_diff} diferenciações. p-valor = {p_valor:.4f}")
    return df_temp, diff_count

# In[22]: Aplicou-se o processo e armazenamento das séries de treino transformada e do parâmetro 'd'
# Apenas a COLUNA 'Close' foi passada para a função

# MGLU3
mglu3_treino_est, mglu3_d = fazer_estacionaria(mglu3treino['Close'], 'MGLU3 Treino')

# ITUB4
itub4_treino_est, itub4_d = fazer_estacionaria(itub4treino['Close'], 'ITUB4 Treino')

# PETR4
petr4_treino_est, petr4_d = fazer_estacionaria(petr4treino['Close'], 'PETR4 Treino')

# VALE3
vale3_treino_est, vale3_d = fazer_estacionaria(vale3treino['Close'], 'VALE3 Treino')


# Checou-se os parâmetros 'd' encontrados
print("\nParâmetros 'd' (ordem de diferenciação) encontrados:")
print(f"MGLU3 (d): {mglu3_d}")
print(f"ITUB4 (d): {itub4_d}")
print(f"PETR4 (d): {petr4_d}")
print(f"VALE3 (d): {vale3_d}")

# In[23]: Plotou-se as séries de treino e teste juntas para todos os quatro ativos

# MGLU3
plt.figure(figsize=(10, 6))
plt.plot(mglu3['Close'], label='Série Completa')
plt.plot(mglu3treino['Close'], label='Série de Treino')
plt.plot(mglu3teste['Close'], label='Série de Teste')
plt.title("Série de Treino e de Teste - MGLU3")
plt.xlabel('Data')
plt.ylabel('Preço (R$)')
plt.legend()
plt.grid(True)
plt.savefig('treino_teste_mglu3.png') # <--- Nome para salvar
plt.show()

# ITUB4
plt.figure(figsize=(10, 6))
plt.plot(itub4['Close'], label='Série Completa')
plt.plot(itub4treino['Close'], label='Série de Treino')
plt.plot(itub4teste['Close'], label='Série de Teste')
plt.title("Série de Treino e de Teste - ITUB4")
plt.xlabel('Data')
plt.ylabel('Preço (R$)')
plt.legend()
plt.grid(True)
plt.savefig('treino_teste_itub4.png') # <--- Nome para salvar
plt.show()

# PETR4
plt.figure(figsize=(10, 6))
plt.plot(petr4['Close'], label='Série Completa')
plt.plot(petr4treino['Close'], label='Série de Treino')
plt.plot(petr4teste['Close'], label='Série de Teste')
plt.title("Série de Treino e de Teste - PETR4")
plt.xlabel('Data')
plt.ylabel('Preço (R$)')
plt.legend()
plt.grid(True)
plt.savefig('treino_teste_petr4.png') # <--- Nome para salvar
plt.show()

# VALE3
plt.figure(figsize=(10, 6))
plt.plot(vale3['Close'], label='Série Completa')
plt.plot(vale3treino['Close'], label='Série de Treino')
plt.plot(vale3teste['Close'], label='Série de Teste')
plt.title("Série de Treino e de Teste - VALE3")
plt.xlabel('Data')
plt.ylabel('Preço (R$)')
plt.legend()
plt.grid(True)
plt.savefig('treino_teste_vale3.png') # <--- Nome para salvar
plt.show()

#%% In[24]: Aplicou-se a função auto_arima para identificação automática de parâmetros do modelo ARIMA

def aplicar_auto_arima(treino_serie, teste_serie, nome_ativo):
    """
    Aplica o modelo auto_arima, faz previsões e avalia o desempenho.
    """
    try:
        print(f"\n===== {nome_ativo} – Modelo ARIMA Automático =====")

        modelo_fit = auto_arima(
            treino_serie.dropna(),
            start_p=1, start_q=1,
            max_p=5, max_q=5,
            m=1,
            seasonal=True,
            trace=True,
            error_action='ignore',
            suppress_warnings=True,
            stepwise=True
        )

        print(f"Melhor modelo identificado: {modelo_fit.order}")

        # Previu-se para o período de teste - gerou-se para todos os dias do período de teste
        previsao_completa = modelo_fit.predict(n_periods=len(teste_serie))
        previsao = pd.Series(previsao_completa, index=teste_serie.index)

        # Criou-se um DataFrame temporário para garantir que os índices coincidam
        # Buscou-se assim o alinhamento dos dados de teste e da previsão, antes de serem calculadas as métricas
        df_temp = pd.DataFrame({'real': teste_serie, 'previsao': previsao})
        df_temp.dropna(inplace=True) # Remove as linhas com NaN de ambos

        # Avaliou-se o desempenho do modelo usando os dados alinhados
        mae = mean_absolute_error(df_temp['real'], df_temp['previsao'])
        rmse = np.sqrt(mean_squared_error(df_temp['real'], df_temp['previsao']))

        print(f"MAE (Erro Médio Absoluto): {mae:.4f}")
        print(f"RMSE (Raiz do Erro Quadrático Médio): {rmse:.4f}")
        
        return modelo_fit, mae, rmse

    except Exception as e:
        print(f"Erro ao processar o modelo para {nome_ativo}: {e}")
        return None, None, None

# %% In[25]: Aplicou-se a função para cada ativo

# Criou-se um dicionário para armazenar os resultados
resultados = {}

# As séries já estão limpas (sem NaNs) - foram tratadas no bloco 19, não sendo necessário o ".dropna()".

# MGLU3
modelo_mglu3, mae_mglu3, rmse_mglu3 = aplicar_auto_arima(mglu3treino['Close'], mglu3teste['Close'], 'MGLU3')
resultados['MGLU3'] = {'modelo': modelo_mglu3, 'MAE': mae_mglu3, 'RMSE': rmse_mglu3}

# ITUB4
modelo_itub4, mae_itub4, rmse_itub4 = aplicar_auto_arima(itub4treino['Close'], itub4teste['Close'], 'ITUB4')
resultados['ITUB4'] = {'modelo': modelo_itub4, 'MAE': mae_itub4, 'RMSE': rmse_itub4}

# PETR4
modelo_petr4, mae_petr4, rmse_petr4 = aplicar_auto_arima(petr4treino['Close'], petr4teste['Close'], 'PETR4')
resultados['PETR4'] = {'modelo': modelo_petr4, 'MAE': mae_petr4, 'RMSE': rmse_petr4}

# VALE3
modelo_vale3, mae_vale3, rmse_vale3 = aplicar_auto_arima(vale3treino['Close'], vale3teste['Close'], 'VALE3')
resultados['VALE3'] = {'modelo': modelo_vale3, 'MAE': mae_vale3, 'RMSE': rmse_vale3}

#%% In[26]: Gerou-se a tabela de resumo

df_resumo = pd.DataFrame([
    {
        'Ativo': nome,
        'Melhor Modelo ARIMA': res['modelo'].order if res['modelo'] else 'Erro',
        'MAE': f"{res['MAE']:.4f}" if res['MAE'] else 'Erro',
        'RMSE': f"{res['RMSE']:.4f}" if res['RMSE'] else 'Erro'
    } for nome, res in resultados.items()
])

print("\n### Tabela de Resumo dos Modelos ARIMA ###")
print(df_resumo)

#%% In[27]: Validação e Diagnóstico

# Usou-se os objetos de modelo que foram armazenados
print("\n### Resumos Detalhados dos Modelos ###")
for nome, res in resultados.items():
    if res['modelo']:
        print(f"\n-------------------------------------------")
        print(f"Resumo do Modelo para {nome}:")
        print(res['modelo'].summary().as_text())
        
#%% In[28]: Aplicaou-se o Teste de Ljung-Box para verificar autocorrelação dos resíduos

# Usou-se o dicionário 'resultados', onde o "caminho" 'resultados'['MGLU3']['modelo'] contém o modelo
def realizar_teste_ljung_box(modelos_dict):
    for nome, dados in modelos_dict.items():
        if dados['modelo']:
            residuos = dados['modelo'].resid()
            ljung_box = sm.stats.acorr_ljungbox(residuos, lags=[10], return_df=True)
            p_valor = ljung_box.loc[10, 'lb_pvalue']

            print(f"\n--- Teste de Ljung-Box para {nome} ---")
            print(f'Resultado do teste:\n{ljung_box}')
            if p_valor > 0.05:
                print(f"O p-valor é {p_valor:.4f}. Não há autocorrelação nos resíduos.")
            else:
                print(f"O p-valor é {p_valor:.4f}. Existe autocorrelação nos resíduos. O modelo precisa de ajustes.")

# Chamou-se a função com o dicionário de resultados
realizar_teste_ljung_box(resultados)

# In[29]: Aplicou-se o Teste de Normalidade dos Resíduos (Kolmogorov-Smirnov)

def realizar_teste_ks(modelos_dict):
    """
    Realiza o teste de Kolmogorov-Smirnov para os resíduos de cada modelo.
    """
    print("\n### Teste de Normalidade dos Resíduos (Kolmogorov-Smirnov) ###")
    for nome, dados in modelos_dict.items():
        if dados['modelo']:
            residuos = dados['modelo'].resid()
            # Certifique-se de que os resíduos não são uma série vazia
            if len(residuos) > 0:
                ks_stat, p_value = kstest(residuos, 'norm', args=(np.mean(residuos), np.std(residuos)))
                
                print(f"\n--- Teste para {nome} ---")
                print(f'Teste de Kolmogorov-Smirnov: p-valor = {p_value:.4f}')

                if p_value > 0.01:
                    print("Os resíduos seguem uma distribuição normal.")
                else:
                    print("Os resíduos não seguem uma distribuição normal.")
            else:
                print(f"\n--- Teste para {nome} ---")
                print("Não foi possível realizar o teste: os resíduos estão vazios.")

# Certificou-se, novamente, de que 'resultados' é o dicionário que armazena os modelos
realizar_teste_ks(resultados)

#%% In[30]: Gerou-se gráficos de previsão do modelo ARIMA para todos os ativos

# Mapeou-se para acessar as séries individuais - assumiu-se as que foram definidas no bloco In[21]
mapa_treino = {
    'MGLU3': mglu3treino, 'ITUB4': itub4treino,
    'PETR4': petr4treino, 'VALE3': vale3treino
}
mapa_teste = {
    'MGLU3': mglu3teste, 'ITUB4': itub4teste,
    'PETR4': petr4teste, 'VALE3': vale3teste
}

# Iteração de cada ativo com o dicionário 'resultados'
for nome_ativo in resultados.keys():
    print(f"Gerando gráfico para o ativo: {nome_ativo}")
    
    # 1. Obteve-se a ordem do modelo e as séries de treino e teste
    # A ordem está no atributo .order do objeto 'modelo'
    modelo_ordem = resultados[nome_ativo]['modelo'].order
    
    # Acessou-se as séries dos dicionários de mapeamento
    treino = mapa_treino[nome_ativo]['Close']
    teste = mapa_teste[nome_ativo]['Close']
    
    # 2. Criou-se e ajustou-se o modelo ARIMA usando a ordem encontrada
    # Entende-se que estar treinando o modelo novamente seja aceitável, embora menos eficiente
    # do que usar o modelo já ajustado (modelo_fit) do bloco In[27].
    modelo = ARIMA(treino, order=modelo_ordem)
    modelo_fit = modelo.fit()
    
    # 3. Gerou-se a previsão para o período de teste e obteve-se o intervalo de 95% de confiança
    previsao_completa = modelo_fit.get_forecast(steps=len(teste))
    previsao = previsao_completa.predicted_mean
    intervalo_confianca = previsao_completa.conf_int(alpha=0.05)
    
    # 4. Plotou-se o gráfico completo
    plt.figure(figsize=(12, 8))
    
    # Plotou-se a série de treino e teste
    plt.plot(treino.index, treino, color='blue', label='Série de Treino')
    plt.plot(teste.index, teste, color='green', label='Série de Teste (Valor Real)')
    
    # Plotou-se a previsão e a área definida pelo Intervalo de Confiança [IC]
    plt.plot(previsao.index, previsao, color='red', linestyle='--', label='Previsão ARIMA')
    plt.fill_between(
        intervalo_confianca.index,
        intervalo_confianca.iloc[:, 0],
        intervalo_confianca.iloc[:, 1],
        color='red',
        alpha=0.2,
        label='Intervalo de Confiança (95%)'
    )
    
    # Adicionar legendas, títulos e salvar o gráfico
    plt.title(f"Previsão {nome_ativo} com ARIMA {modelo_ordem}")
    plt.xlabel("Data")
    plt.ylabel("Preço de Fechamento")
    plt.legend()
    plt.grid(True)
    
    # Salvar o gráfico antes de mostrar
    plt.savefig(f'previsao_{nome_ativo.lower()}_completo.png', dpi=300)
    plt.show()

#%%##############################################################################################################
# Aplicação das Funções de Autocorrelação [ACF] e de Autocorrelação Parcial [PACF]
# Apresentou-se em complemento e entendeu-se inexistir obrigatoriedade uma vez que usou-se a função auto_arima()
#################################################################################################################

#% In[31]: Definiu-se as funções ACF e PACF para as séries de treino diferenciada dos Ativos
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

#%% In[32]: Plotou-se as funções ACF e PACF para as séries de treino diferenciada dos Ativos

# Gráfico ACF e PACF para MGLU3
# Usou-se: mglu3_treino_est
fig, axes = plt.subplots(1, 2, figsize=(16,4))
plot_acf(mglu3_treino_est.dropna(), lags=24, ax=axes[0], title="MGLU3 Diferenciado - ACF")
plot_pacf(mglu3_treino_est.dropna(), lags=24, ax=axes[1], method='ywm', title="MGLU3 Diferenciado - PACF")
plt.savefig('acf_pacf_mglu3.png', dpi=300) # <--- CORRIGIDO
plt.show()

# Gráfico ACF e PACF para ITUB4
# Usou-se: itub4_treino_est
fig, axes = plt.subplots(1, 2, figsize=(16,4))
plot_acf(itub4_treino_est.dropna(), lags=24, ax=axes[0], title="ITUB4 Diferenciado - ACF")
plot_pacf(itub4_treino_est.dropna(), lags=24, ax=axes[1], method='ywm', title="ITUB4 Diferenciado - PACF")
plt.savefig('acf_pacf_itub4.png', dpi=300) # <--- CORRIGIDO
plt.show()

# Gráfico ACF e PACF para PETR4
# Usou-se: petr4_treino_est
fig, axes = plt.subplots(1, 2, figsize=(16,4))
plot_acf(petr4_treino_est.dropna(), lags=24, ax=axes[0], title="PETR4 Diferenciado - ACF")
plot_pacf(petr4_treino_est.dropna(), lags=24, ax=axes[1], method='ywm', title="PETR4 Diferenciado - PACF")
plt.savefig('acf_pacf_petr4.png', dpi=300) # <--- CORRIGIDO
plt.show()

# Gráfico ACF e PACF para VALE3
# Usou-se: vale3_treino_est
fig, axes = plt.subplots(1, 2, figsize=(16,4))
plot_acf(vale3_treino_est.dropna(), lags=24, ax=axes[0], title="VALE3 Diferenciado - ACF")
plot_pacf(vale3_treino_est.dropna(), lags=24, ax=axes[1], method='ywm', title="VALE3 Diferenciado - PACF")
plt.savefig('acf_pacf_vale3.png', dpi=300) # <--- CORRIGIDO
plt.show()

#%%####################################################################
# Início da Aplicação dos Modelos de Machine Learning
# Modelo Random Forest  [RF]
#######################################################################

#% In[33]: RF - Definiu-se a função para testar múltiplas combinações de hiperparâmetros

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

    # Listou-se as combinações a serem testadas
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

        # Gráfico de linha Real vs Previsto - optou-se por não rodar esses gráficos pelo "esforço" computacional
        #plt.figure(figsize=(10, 4))
        #plt.plot(datas_teste, y_test, label='Preço Real', color='blue')
        #plt.plot(datas_teste, y_pred, label='Previsão RF', linestyle='--', color='orange')
        #plt.title(f"{nome_ativo} – Previsão com {params}")
        #plt.xlabel("Data")
        #plt.ylabel("Preço de Fechamento (R$)")
        #plt.legend()
        #plt.grid(True)
        #plt.tight_layout()
        #plt.savefig(f'previsao_{nome_ativo.lower()}_modelo_{i}.png', dpi=300)
        #plt.show()

    return pd.DataFrame(resultados)

#%% In[34]: Definiu-se uma função para criar dicionário com os DataFrames de resultados por ativo e imprime os melhores modelos (menor RMSE) e seus hiperparâmetros.
    
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

#%% In[35]: Definiu-se dicionários com os resultados dos ativos

resultados_itub4_rf = testar_parametros_rf(itub4, "ITUB4")
resultados_mglu3_rf = testar_parametros_rf(mglu3, "MGLU3")
resultados_petr4_rf = testar_parametros_rf(petr4, "PETR4")
resultados_vale3_rf = testar_parametros_rf(vale3, "VALE3")

#%% In[36]: Armazenou-se os resultados dos quatro ativos em seus respectivos dicionários

resultados_rf = {
    'ITUB4': resultados_itub4_rf,
    'MGLU3': resultados_mglu3_rf,
    'PETR4': resultados_petr4_rf,
    'VALE3': resultados_vale3_rf
}


#%% In[37]: Identificou-se o melhor modelo para cada Ativo

identificar_melhor_modelo_por_ativo(resultados_rf)

#%% In[38]: Definiu-se a função para gerar apenas o gráfico do melhor modelo de RF para cada Ativo

def plotar_melhor_modelo(df, nome_ativo, df_resultados):
    
    # Identificou-se o melhor modelo (menor RMSE)
    melhor = df_resultados.loc[df_resultados['RMSE'].idxmin()]
    n_estimators = int(melhor['n_estimators'])
    max_depth = None if pd.isna(melhor['max_depth']) else int(melhor['max_depth'])
    min_samples_split = int(melhor['min_samples_split'])

    print(f"\n{nome_ativo} – Melhor Modelo:")
    print(f"  n_estimators: {n_estimators}")
    print(f"  max_depth: {max_depth}")
    print(f"  min_samples_split: {min_samples_split}")

    # Preparou-se os dados
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

    # Gerou-se os gráficos
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

#%% In[39]: Gerou-se a Tabela de Resumo dos Melhores Modelos Random Forest (RF)

# Criou-se uma lista para armazenar os dados de resumo
dados_resumo_rf = []

for nome_ativo, df_resultados in resultados_rf.items():
    # Identificou-se a linha com o menor RMSE (melhor modelo)
    melhor_modelo = df_resultados.loc[df_resultados['RMSE'].idxmin()]
    
    # Armazenou-se os dados relevantes
    dados_resumo_rf.append({
        'Ativo': nome_ativo,
        'n_estimators': melhor_modelo['n_estimators'],
        'max_depth': melhor_modelo['max_depth'],
        'min_samples_split': melhor_modelo['min_samples_split'],
        'MAE': melhor_modelo['MAE'],
        'RMSE': melhor_modelo['RMSE']
    })

# Criou-se um DataFrame de resumo
df_resumo_rf_final = pd.DataFrame(dados_resumo_rf)

# Aplicou-se a seguinte formatação para melhor visualização
df_resumo_rf_final['max_depth'] = df_resumo_rf_final['max_depth'].replace({np.nan: 'None'})
df_resumo_rf_final['MAE'] = df_resumo_rf_final['MAE'].apply(lambda x: f"{x:.4f}")
df_resumo_rf_final['RMSE'] = df_resumo_rf_final['RMSE'].apply(lambda x: f"{x:.4f}")

print("\n### Tabela de Resumo dos Melhores Modelos Random Forest (RF) ###")
print(df_resumo_rf_final)

#%% In [40]: Executou-se a plotagem dos gráficos de melhor modelo para os 4 ativos
    
plotar_melhor_modelo(itub4, 'ITUB4', resultados_itub4_rf)
plotar_melhor_modelo(mglu3, 'MGLU3', resultados_mglu3_rf)
plotar_melhor_modelo(petr4, 'PETR4', resultados_petr4_rf)
plotar_melhor_modelo(vale3, 'VALE3', resultados_vale3_rf)

#%%  In[41]: Registrou-se os dados de desempenho do RF e construiu-se os gráficos comparativos das métricas de erro por Ativo

resultados_rf = pd.DataFrame({
    'Ativo': ['ITUB4', 'MGLU3', 'PETR4', 'VALE3'],
    'Setor': ['Financeiro', 'Varejo', 'Petróleo e Gás', 'Materiais Básicos'],
    'RMSE': [1.2539, 0.4704, 10.4532, 0.3902],
    'MAE': [0.5394, 0.3631, 9.3375, 0.3053]
})

#%% In[42] Plotou-se o Gráfico de RMSE (RF)

plt.figure(figsize=(8, 5))
sns.barplot(data=resultados_rf, x='Ativo', y='RMSE', palette='magma')
plt.title('Erro Quadrático Médio (RMSE) por Ativo – Random Forest')
plt.ylabel('RMSE')
plt.xlabel('Ativo')
plt.grid(axis='y')
plt.tight_layout()
plt.savefig('grafico_rf_rmse.png', dpi=300)
plt.show()

#%% In[43] Plotou-se o Gráfico de MAE (RF)

plt.figure(figsize=(8, 5))
sns.barplot(data=resultados_rf, x='Ativo', y='MAE', palette='viridis')
plt.title('Erro Médio Absoluto (MAE) por Ativo – Random Forest')
plt.ylabel('MAE')
plt.xlabel('Ativo')
plt.grid(axis='y')
plt.tight_layout()
plt.savefig('grafico_rf_mae.png', dpi=300)
plt.show()

#%%########################################################################
# Início da Aplicação das Aruiteturas de Redes Neurais [RNAs]
# Arquitetura de Redes Neurais Recorrentes - Long Short Term Memory [LSTM]
###########################################################################

#% In[44]: Definiu-se a função de preparação da série temporal para LSTM

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

#%% In[45]: Definiu-se a função para rodar NRAs - modelo LSTM e GRU

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

#%% In[46]: Definiu-se a função para plotar LSTM

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
    
#%% In[47]: Rodou-se os modelos LSTM para cada ativo

# Buscou-se os resultados para o modelo LSTM
print("\n===== Resultados com LSTM =====")
res_lstm_itub4 = treinar_modelo_rna("ITUB4", itub4, tipo='LSTM')
y_test_itub4_lstm, y_pred_itub4_lstm, rmse_itub4_lstm, mae_itub4_lstm = treinar_modelo_rna("ITUB4", itub4, tipo='LSTM')
y_test_mglu3_lstm, y_pred_mglu3_lstm, rmse_mglu3_lstm, mae_mglu3_lstm = treinar_modelo_rna("MGLU3", mglu3, tipo='LSTM')
y_test_petr4_lstm, y_pred_petr4_lstm, rmse_petr4_lstm, mae_petr4_lstm = treinar_modelo_rna("PETR4", petr4, tipo='LSTM')
y_test_vale3_lstm, y_pred_vale3_lstm, rmse_vale3_lstm, mae_vale3_lstm = treinar_modelo_rna("VALE3", vale3, tipo='LSTM')

#%% In[48]: Plotou-se o melhor gráfico LSTM por Ativo

# ITUB4
plot_real_vs_pred(y_test_itub4_lstm, y_pred_itub4_lstm, 'ITUB4', 'ITUB4', 'LSTM', rmse=rmse_itub4_lstm, mae=mae_itub4_lstm)

# MGLU3
plot_real_vs_pred(y_test_mglu3_lstm, y_pred_mglu3_lstm, 'MGLU3', 'MGLU3', 'LSTM', rmse=rmse_mglu3_lstm, mae=mae_mglu3_lstm)

# PETR4
plot_real_vs_pred(y_test_petr4_lstm, y_pred_petr4_lstm, 'PETR4', 'PETR4', 'LSTM', rmse=rmse_petr4_lstm, mae=mae_petr4_lstm)

# VALE3
plot_real_vs_pred(y_test_vale3_lstm, y_pred_vale3_lstm, 'VALE3', 'VALE3', 'LSTM', rmse=rmse_vale3_lstm, mae=mae_vale3_lstm)

#%% In[49] Armazenou-se os resultados com os dados de desempenho do modelo LSTM

resultados_lstm = pd.DataFrame({
    'Ativo': ['ITUB4', 'MGLU3', 'PETR4', 'VALE3'],
    'Setor': ['Financeiro', 'Varejo', 'Petróleo e Gás', 'Materiais Básicos'],
    'RMSE': [rmse_itub4_lstm, rmse_mglu3_lstm, rmse_petr4_lstm, rmse_vale3_lstm],
    'MAE': [mae_itub4_lstm, mae_mglu3_lstm, mae_petr4_lstm, mae_vale3_lstm]
})

#%% In[50] Plotou-se o Gráfico de RMSE (LSTM)

plt.figure(figsize=(8, 5))
sns.barplot(data=resultados_lstm, x='Ativo', y='RMSE', palette='magma')
plt.title('Erro Quadrático Médio (RMSE) por Ativo – LSTM')
plt.ylabel('RMSE')
plt.xlabel('Ativo')
plt.grid(axis='y')
plt.tight_layout()
plt.savefig('grafico_lstm_rmse.png', dpi=300)
plt.show()

#%% In[51] Plotou-se o Gráfico de MAE (LSTM)

plt.figure(figsize=(8, 5))
sns.barplot(data=resultados_lstm, x='Ativo', y='MAE', palette='viridis')
plt.title('Erro Médio Absoluto (MAE) por Ativo – LSTM')
plt.ylabel('MAE')
plt.xlabel('Ativo')
plt.grid(axis='y')
plt.tight_layout()
plt.savefig('grafico_lstm_mae.png', dpi=300)
plt.show()

#%%####################################################################
# Arquitetura de Redes Neurais Recorrentes - Gated Recurrent Unit [GRU]
#######################################################################

#% In[52] Iniciou-se o Treinamento GRU por ativo

print("\n===== Resultados com GRU =====")
y_test_itub4_gru,  y_pred_itub4_gru,  rmse_itub4_gru,  mae_itub4_gru  = treinar_modelo_rna("ITUB4", itub4,  tipo='GRU')
y_test_mglu3_gru,  y_pred_mglu3_gru,  rmse_mglu3_gru,  mae_mglu3_gru  = treinar_modelo_rna("MGLU3", mglu3,  tipo='GRU')
y_test_petr4_gru,  y_pred_petr4_gru,  rmse_petr4_gru,  mae_petr4_gru  = treinar_modelo_rna("PETR4", petr4,  tipo='GRU')
y_test_vale3_gru,  y_pred_vale3_gru,  rmse_vale3_gru,  mae_vale3_gru  = treinar_modelo_rna("VALE3", vale3,  tipo='GRU')

#%% In[53] Plotou-se o Gráficos Real vs Previsto (GRU)

# ITUB4
plot_real_vs_pred(y_test_itub4_gru, y_pred_itub4_gru, 'ITUB4', 'ITUB4', 'GRU', rmse=rmse_itub4_gru, mae=mae_itub4_gru)

# MGLU3
plot_real_vs_pred(y_test_mglu3_gru, y_pred_mglu3_gru, 'MGLU3', 'MGLU3', 'GRU', rmse=rmse_mglu3_gru, mae=mae_mglu3_gru)

# PETR4
plot_real_vs_pred(y_test_petr4_gru, y_pred_petr4_gru, 'PETR4', 'PETR4', 'GRU', rmse=rmse_petr4_gru, mae=mae_petr4_gru)

# VALE3
plot_real_vs_pred(y_test_vale3_gru, y_pred_vale3_gru, 'VALE3', 'VALE3', 'GRU', rmse=rmse_vale3_gru, mae=mae_vale3_gru)

#%% In[54] Construi-se a Tabela de desempenho (GRU)

resultados_gru = pd.DataFrame({
    'Ativo': ['ITUB4', 'MGLU3', 'PETR4', 'VALE3'],
    'Setor': ['Financeiro', 'Varejo', 'Petróleo e Gás', 'Materiais Básicos'],
    'RMSE': [rmse_itub4_gru, rmse_mglu3_gru, rmse_petr4_gru, rmse_vale3_gru],
    'MAE' : [mae_itub4_gru, mae_mglu3_gru, mae_petr4_gru, mae_vale3_gru]
})
print(resultados_gru.round(4))

#%% In[55] Plotou-se o Gráfico RMSE (GRU)

plt.figure(figsize=(8,5))
sns.barplot(data=resultados_gru, x='Ativo', y='RMSE', palette='magma')
plt.title('Erro Quadrático Médio (RMSE) por Ativo – GRU')
plt.ylabel('RMSE'); plt.xlabel('Ativo')
plt.grid(axis='y'); plt.tight_layout()
plt.savefig('grafico_gru_rmse.png', dpi=300)
plt.show()


#%% In[56] Plotou-se o Gráfico MAE (GRU)

plt.figure(figsize=(8,5))
sns.barplot(data=resultados_gru, x='Ativo', y='MAE', palette='viridis')
plt.title('Erro Médio Absoluto (MAE) por Ativo – GRU')
plt.ylabel('MAE'); plt.xlabel('Ativo')
plt.grid(axis='y'); plt.tight_layout()
plt.savefig('grafico_gru_mae.png', dpi=300)
plt.show()

#%% In[57] Comparou-se as métricas de erro (RMSE e MAE) dos modelos LSTM vs GRU, por ativo 

# Plotou-se um Gráfico único, a fim de contrastar as duas arquiteturas.
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

#%%####################################################################
#                            F  I  M                                  #
#######################################################################

