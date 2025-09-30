# Predição de Ações da B3

# Objetivo
Este projeto compara modelos de predição na previsão do preço de fechamento diário de quatro ativos da B3, de setores distintos da economia. A metodologia inclui um benchmark com ARIMA e modelos de Machine Learning [ML], nomeadamente: Random Forest [RF] e os modelos de Redes Neurais Recorrentes Long Short-Term Memory [LSTM] e Gated Recurrent Units [GRU], sendo esta última utilizada especificamente para fins de comparação de desempenho com a LSTM. Os ativos são:
    • ITUB4 do setor Financeiro;
    • MGLU3 do setor Varejo; 
    • PETR4 do setor Petróleo, Gás e Biocombustíveis; e
    • VALE3 do setor Materiais Básicos.

# Período analisado: 02/01/2015 a 30/05/2025
Divisão temporal: *Treino (2015–2023-04-23)* e **Teste (2023-04-24–2025-05-30)**  
Frequência: Business Day (pregões da B3)

# Métricas de avaliação: **RMSE** e **MAE** (em R$).

# Etapas do Pipeline

1. Coleta e Pré-processamento
- Download via [`yfinance`](https://pypi.org/project/yfinance/).
- Ajuste da frequência para *Business Day (‘B’)*, garantindo alinhamento aos pregões da B3.
- Remoção de dias sem negociação (sem imputação artificial).
- Estatísticas descritivas: média, desvio padrão, coeficiente de variação (CV).
- Gráficos:  
    • Volatilidade  
    • Volume Médio
    • Coeficiente de Variação

2. Indicadores Técnicos
- Implementação de EMA12, EMA26 e MACD com linha de sinal; e 
- Todos incluídos como features nos modelos de ML.

3. Teste de Estacionariedade nas séries de treino
- Aplicação do ADF (Dickey-Fuller Aumentado) em nível e em primeira diferença; e
- Todas as séries foram classificadas como I(1); 

4. Modelagem ARIMA
- Seleção automática de parâmetros via auto_arima();
- Previsões multi-step sem realimentação para manter equidade com os modelos de ML;
- Diagnóstico dos resíduos:
    •  Ljung–Box (para verificar a existência de autocorrelação); e
    •  Kolmogorov–Smirnov (para verificar a normalidade dos resíduos).
- Resultados consolidados:  

5. RF
- Features: EMA12, EMA26, MACD, Sinal_MACD, Open, High, Low e Volume;
- Ajuste manual de hiperparâmetros (n_estimators, max_depth, min_samples_split);
- Melhor desempenho para MGLU3 e VALE3; e
- Apresentou-se os gráficos.
 

6. RNAs - Redes Neurais Recorrentes [RNNs]

# LSTM
- Preparação com janelas deslizantes de 60 dias e MinMaxScaler;
- Arquitetura: 1 camada LSTM (50 neurônios) + Dense(1), Adam (0.001), 50 épocas; e
- Melhor desempenho para ITUB4 e PETR4; e 
- Apresentou-se os gráficos. 

# GRU
- Mesma arquitetura da LSTM, substituindo a camada recorrente; e
- Competitiva em MGLU3 e PETR4, inferior em ITUB4 e VALE3.  


# Resultados Comparativos

    • ARIMA: limitado em horizontes longos, serviu como benchmark inicial.
    • RF: melhor em MGLU3 e VALE3.  
    • LSTM: superior em ITUB4 e especialmente em PETR4.
    • GRU: alternativa competitiva, superando a LSTM em MGLU3 e PETR4, mas inferior em ITUB4 e VALE3. 

# Comparativos consolidados:
- Gráfico comparativo de RMSE; e
- Gráfico comparativo de MAE

## Observações:
- A pesquisa foi desenvolvida como parte do MBA em Data Science & Analytics – EAD, da USP/ESALQ; e
- Disclaimer: este estudo tem finalidade acadêmica e não constitui recomendação de investimento.
