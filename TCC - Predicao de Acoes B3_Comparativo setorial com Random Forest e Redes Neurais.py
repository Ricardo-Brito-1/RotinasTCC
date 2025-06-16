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
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Para escalonamento e divisão dos dados
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

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
import pandas as pd
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

#%% In[18]: Instalou-se a biblioteca para função auto_arima - Identificação automática para modelo ARIMA

!pip install pmdarima
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error, mean_squared_error

#%% In[19]: Aplicou-se a função auto_arima para identificação automática de parâmetros do modelo ARIMA (auto_arima)

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

#%% In[20]: Aplicou-se a função auto_arima para cada papel:

auto_arima_previsao(itub4, "ITUB4")
auto_arima_previsao(mglu3, "MGLU3")
auto_arima_previsao(petr4, "PETR4")
auto_arima_previsao(vale3, "VALE3")

#%%
###############################################################################
###############################################################################
#          *** R E S U L T A D O S   P R E L I M I N  A R E S ***
###############################################################################
###############################################################################