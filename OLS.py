import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Dataset construction
estoque_medio_produtos = [
    {"Mês": "Jan 07", "Ano": 2007, "Estoque Médio de Produtos (%)": 67},
    {"Mês": "Fev 07", "Ano": 2007, "Estoque Médio de Produtos (%)": 78},
    {"Mês": "Mar 07", "Ano": 2007, "Estoque Médio de Produtos (%)": 100},
    {"Mês": "Abr 07", "Ano": 2007, "Estoque Médio de Produtos (%)": 88},
    {"Mês": "Mai 07", "Ano": 2007, "Estoque Médio de Produtos (%)": 61},
    {"Mês": "Jun 07", "Ano": 2007, "Estoque Médio de Produtos (%)": 72},
    {"Mês": "Jul 07", "Ano": 2007, "Estoque Médio de Produtos (%)": 62},
    {"Mês": "Ago 07", "Ano": 2007, "Estoque Médio de Produtos (%)": 59},
    {"Mês": "Set 07", "Ano": 2007, "Estoque Médio de Produtos (%)": 71},
    {"Mês": "Out 07", "Ano": 2007, "Estoque Médio de Produtos (%)": 71},
    {"Mês": "Nov 07", "Ano": 2007, "Estoque Médio de Produtos (%)": 78},
    {"Mês": "Dez 07", "Ano": 2007, "Estoque Médio de Produtos (%)": 86}
]

acuracidade_previsao = [
    {"Mês": "Abr 07", "Ano": 2007, "Acuracidade de Previsão (%)": 14},
    {"Mês": "Mai 07", "Ano": 2007, "Acuracidade de Previsão (%)": 39},
    {"Mês": "Jun 07", "Ano": 2007, "Acuracidade de Previsão (%)": 19},
    {"Mês": "Jul 07", "Ano": 2007, "Acuracidade de Previsão (%)": 30},
    {"Mês": "Ago 07", "Ano": 2007, "Acuracidade de Previsão (%)": 67},
    {"Mês": "Set 07", "Ano": 2007, "Acuracidade de Previsão (%)": 64},
    {"Mês": "Out 07", "Ano": 2007, "Acuracidade de Previsão (%)": 38},
    {"Mês": "Nov 07", "Ano": 2007, "Acuracidade de Previsão (%)": 34},
    {"Mês": "Dez 07", "Ano": 2007, "Acuracidade de Previsão (%)": 36}
]

sellin_sellout = [
    {"Mês": "Fev 05", "Ano": 2005, "Sell-in (unidades)": 800000, "Sell-out (unidades)": 580000},
    {"Mês": "Mar 05", "Ano": 2005, "Sell-in (unidades)": 900000, "Sell-out (unidades)": 620000},
    {"Mês": "Abr 05", "Ano": 2005, "Sell-in (unidades)": 620000, "Sell-out (unidades)": 620000},
    {"Mês": "Mai 05", "Ano": 2005, "Sell-in (unidades)": 500000, "Sell-out (unidades)": 600000},
    {"Mês": "Jun 05", "Ano": 2005, "Sell-in (unidades)": 650000, "Sell-out (unidades)": 620000},
    {"Mês": "Jul 05", "Ano": 2005, "Sell-in (unidades)": 750000, "Sell-out (unidades)": 620000},
    {"Mês": "Ago 05", "Ano": 2005, "Sell-in (unidades)": 800000, "Sell-out (unidades)": 630000},
    {"Mês": "Set 05", "Ano": 2005, "Sell-in (unidades)": 770000, "Sell-out (unidades)": 750000},
    {"Mês": "Out 05", "Ano": 2005, "Sell-in (unidades)": 820000, "Sell-out (unidades)": 620000},
    {"Mês": "Nov 05", "Ano": 2005, "Sell-in (unidades)": 800000, "Sell-out (unidades)": 630000},
    {"Mês": "Dez 05", "Ano": 2005, "Sell-in (unidades)": 780000, "Sell-out (unidades)": 630000},
    {"Mês": "Jan 06", "Ano": 2006, "Sell-in (unidades)": 640000, "Sell-out (unidades)": 630000},
    {"Mês": "Fev 06", "Ano": 2006, "Sell-in (unidades)": 540000, "Sell-out (unidades)": 620000},
    {"Mês": "Mar 06", "Ano": 2006, "Sell-in (unidades)": 750000, "Sell-out (unidades)": 570000},
    {"Mês": "Abr 06", "Ano": 2006, "Sell-in (unidades)": 700000, "Sell-out (unidades)": 600000},
    {"Mês": "Mai 06", "Ano": 2006, "Sell-in (unidades)": 400000, "Sell-out (unidades)": 580000},
    {"Mês": "Jun 06", "Ano": 2006, "Sell-in (unidades)": 600000, "Sell-out (unidades)": 600000},
    {"Mês": "Jul 06", "Ano": 2006, "Sell-in (unidades)": 650000, "Sell-out (unidades)": 600000},
    {"Mês": "Ago 06", "Ano": 2006, "Sell-in (unidades)": 720000, "Sell-out (unidades)": 670000},
    {"Mês": "Set 06", "Ano": 2006, "Sell-in (unidades)": 800000, "Sell-out (unidades)": 720000},
    {"Mês": "Out 06", "Ano": 2006, "Sell-in (unidades)": 780000, "Sell-out (unidades)": 630000},
    {"Mês": "Nov 06", "Ano": 2006, "Sell-in (unidades)": 700000, "Sell-out (unidades)": 630000},
    {"Mês": "Dez 06", "Ano": 2006, "Sell-in (unidades)": 830000, "Sell-out (unidades)": 600000},
    {"Mês": "Jan 07", "Ano": 2007, "Sell-in (unidades)": 1000000, "Sell-out (unidades)": 580000},
    {"Mês": "Fev 07", "Ano": 2007, "Sell-in (unidades)": 400000, "Sell-out (unidades)": 550000},
    {"Mês": "Mar 07", "Ano": 2007, "Sell-in (unidades)": 630000, "Sell-out (unidades)": 520000},
    {"Mês": "Abr 07", "Ano": 2007, "Sell-in (unidades)": 450000, "Sell-out (unidades)": 560000},
    {"Mês": "Mai 07", "Ano": 2007, "Sell-in (unidades)": 300000, "Sell-out (unidades)": 540000},
    {"Mês": "Jun 07", "Ano": 2007, "Sell-in (unidades)": 400000, "Sell-out (unidades)": 600000},
    {"Mês": "Jul 07", "Ano": 2007, "Sell-in (unidades)": 720000, "Sell-out (unidades)": 580000},
    {"Mês": "Ago 07", "Ano": 2007, "Sell-in (unidades)": 600000, "Sell-out (unidades)": 600000},
    {"Mês": "Set 07", "Ano": 2007, "Sell-in (unidades)": 700000, "Sell-out (unidades)": 580000},
    {"Mês": "Out 07", "Ano": 2007, "Sell-in (unidades)": 560000, "Sell-out (unidades)": 600000},
    {"Mês": "Nov 07", "Ano": 2007, "Sell-in (unidades)": 700000, "Sell-out (unidades)": 580000},
    {"Mês": "Dez 07", "Ano": 2007, "Sell-in (unidades)": 730000, "Sell-out (unidades)": 585000}
]


estoque_faturamento_liquido_mensal = [
    {"Mês": "Abr 06", "Ano": 2006, "Estoque (US$)": 1500000, "Faturamento (US$)": 600000},
    {"Mês": "Mai 06", "Ano": 2006, "Estoque (US$)": 1520000, "Faturamento (US$)": 1200000},
    {"Mês": "Jun 06", "Ano": 2006, "Estoque (US$)": 1470000, "Faturamento (US$)": 1400000},
    {"Mês": "Jul 06", "Ano": 2006, "Estoque (US$)": 1460000, "Faturamento (US$)": 1560000},
    {"Mês": "Ago 06", "Ano": 2006, "Estoque (US$)": 1480000, "Faturamento (US$)": 1700000},
    {"Mês": "Set 06", "Ano": 2006, "Estoque (US$)": 1360000, "Faturamento (US$)": 1680000},
    {"Mês": "Out 06", "Ano": 2006, "Estoque (US$)": 1400000, "Faturamento (US$)": 1600000},
    {"Mês": "Nov 06", "Ano": 2006, "Estoque (US$)": 1400000, "Faturamento (US$)": 1900000},
    {"Mês": "Dez 06", "Ano": 2006, "Estoque (US$)": 1300000, "Faturamento (US$)": 2300000},
    {"Mês": "Jan 07", "Ano": 2007, "Estoque (US$)": 1520000, "Faturamento (US$)": 800000},
    {"Mês": "Fev 07", "Ano": 2007, "Estoque (US$)": 1500000, "Faturamento (US$)": 1480000},
    {"Mês": "Mar 07", "Ano": 2007, "Estoque (US$)": 1400000, "Faturamento (US$)": 2800000},
    {"Mês": "Abr 07", "Ano": 2007, "Estoque (US$)": 1480000, "Faturamento (US$)": 600000},
    {"Mês": "Mai 07", "Ano": 2007, "Estoque (US$)": 1700000, "Faturamento (US$)": 700000},
    {"Mês": "Jun 07", "Ano": 2007, "Estoque (US$)": 1600000, "Faturamento (US$)": 2200000},
    {"Mês": "Jul 07", "Ano": 2007, "Estoque (US$)": 1500000, "Faturamento (US$)": 1650000},
    {"Mês": "Ago 07", "Ano": 2007, "Estoque (US$)": 1550000, "Faturamento (US$)": 1800000},
    {"Mês": "Set 07", "Ano": 2007, "Estoque (US$)": 1500000, "Faturamento (US$)": 2250000},
    {"Mês": "Out 07", "Ano": 2007, "Estoque (US$)": 2300000, "Faturamento (US$)": 1500000},
    {"Mês": "Nov 07", "Ano": 2007, "Estoque (US$)": 2800000, "Faturamento (US$)": 2100000},
    {"Mês": "Dez 07", "Ano": 2007, "Estoque (US$)": 2600000, "Faturamento (US$)": 2100000}
]


market_share=[
    {"Competidor": "Stiefel", "2005 (%)": 10.00, "2006 (%)": 9.80, "2007 (%)": 8.80},
    {"Competidor": "Mantecorp I Q Farm", "2005 (%)": 6.40, "2006 (%)": 6.50, "2007 (%)": 6.70},
    {"Competidor": "Galderma", "2005 (%)": 7.70, "2006 (%)": 7.00, "2007 (%)": 6.20},
    {"Competidor": "La Roche Posay", "2005 (%)": 4.00, "2006 (%)": 5.00, "2007 (%)": 5.90},
    {"Competidor": "Medley", "2005 (%)": 4.00, "2006 (%)": 4.60, "2007 (%)": 5.30},
    {"Competidor": "Bayer Schering PH", "2005 (%)": 5.60, "2006 (%)": 5.20, "2007 (%)": 4.60},
    {"Competidor": "Eurofarma", "2005 (%)": 4.00, "2006 (%)": 4.40, "2007 (%)": 4.60},
    {"Competidor": "Procter & Gamble", "2005 (%)": 4.40, "2006 (%)": 4.60, "2007 (%)": 4.50},
    {"Competidor": "Roche", "2005 (%)": 3.90, "2006 (%)": 4.20, "2007 (%)": 4.30},
    {"Competidor": "EMS Pharma", "2005 (%)": 2.90, "2006 (%)": 3.70, "2007 (%)": 4.10}
]

vendas_por_semana = [
    {"Semana": "1ª Semana", "Ano": 2006, "Vendas (%)": 5},
    {"Semana": "1ª Semana", "Ano": 2007, "Vendas (%)": 3},
    {"Semana": "2ª Semana", "Ano": 2006, "Vendas (%)": 23},
    {"Semana": "2ª Semana", "Ano": 2007, "Vendas (%)": 17},
    {"Semana": "3ª Semana", "Ano": 2006, "Vendas (%)": 29},
    {"Semana": "3ª Semana", "Ano": 2007, "Vendas (%)": 33},
    {"Semana": "4ª Semana", "Ano": 2006, "Vendas (%)": 33},
    {"Semana": "4ª Semana", "Ano": 2007, "Vendas (%)": 47}
]

# Months in portuguese to english
def traduzir_mes(mes):
    meses = {
        'Jan': 'Jan', 'Fev': 'Feb', 'Mar': 'Mar', 'Abr': 'Apr', 'Mai': 'May', 'Jun': 'Jun',
        'Jul': 'Jul', 'Ago': 'Aug', 'Set': 'Sep', 'Out': 'Oct', 'Nov': 'Nov', 'Dez': 'Dec'
    }
    for pt, en in meses.items():
        if pt in mes:
            return mes.replace(pt, en)
    return mes

# Create dataframes
df_estoque_medio = pd.DataFrame(estoque_medio_produtos)
df_acuracidade = pd.DataFrame(acuracidade_previsao)
df_sell = pd.DataFrame(sellin_sellout)
df_estoque_fat = pd.DataFrame(estoque_faturamento_liquido_mensal)


for df in [df_estoque_medio, df_acuracidade, df_sell, df_estoque_fat]:
    df['Mês'] = df['Mês'] + ' ' + df['Ano'].astype(str)
    df['Mês'] = df['Mês'].apply(traduzir_mes)
    df['Mês'] = pd.to_datetime(df['Mês'])

df_total = pd.merge(df_sell, df_estoque_medio, on=['Mês', 'Ano'], how='left')
df_total = pd.merge(df_total, df_acuracidade, on=['Mês', 'Ano'], how='left')
df_total = pd.merge(df_total, df_estoque_fat, on=['Mês', 'Ano'], how='left')

cols_fill = ['Estoque Médio de Produtos (%)', 'Acuracidade de Previsão (%)', 
             'Estoque (US$)', 'Faturamento (US$)']
df_total[cols_fill] = df_total[cols_fill].fillna(df_total[cols_fill].mean())

df_total['Tendencia'] = range(1, len(df_total) + 1)
df_total['Mes'] = df_total['Mês'].dt.month
df_total = pd.get_dummies(df_total, columns=['Mes'], prefix='Mes', drop_first=True)

# Select the relevant data that impacts the dependent variable
features = [
    'Sell-in (unidades)',  # Variável de entrada
    'Tendencia',  # Variável de tendência temporal
    'Mes_2', 'Mes_3', 'Mes_4', 'Mes_5', 'Mes_6',  # Variáveis dummy para meses
    'Mes_7', 'Mes_8', 'Mes_9', 'Mes_10', 'Mes_11', 'Mes_12'
]

# Divide train and test (Last 6 months for tests)
train = df_total[:-6]
test = df_total[-6:]

X_train = train[features]
y_train = train['Sell-out (unidades)']
X_test = test[features]
y_test = test['Sell-out (unidades)']

X_train = X_train.fillna(X_train.mean())
y_train = y_train.fillna(y_train.mean())

X_train = X_train.apply(pd.to_numeric, errors='coerce')
y_train = pd.to_numeric(y_train, errors='coerce')

X_train = X_train.astype(int)
X_test = X_test.astype(int)

X_train = sm.add_constant(X_train)
X_test = sm.add_constant(X_test)

# Adjusting the OLS model
modelo = sm.OLS(y_train, X_train)
resultados = modelo.fit()

# Results
print(resultados.summary())
print("\n" + "=" * 60)
print("Diagnóstico do Modelo:")

from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Autocolação
dw = durbin_watson(resultados.resid)
print(f"\nDurbin-Watson: {dw:.2f}")

# Heterocedasticidade
_, pval, _, _ = het_breuschpagan(resultados.resid, resultados.model.exog)
print(f"p-valor Heterocedasticidade: {pval:.4f}")

# Multicolinearidade
vif_data = pd.DataFrame()
vif_data["Variável"] = X_train.columns
vif_data["VIF"] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
print("\nVIF:\n", vif_data)

# Metrics and prediction
y_pred = resultados.predict(X_test)
print("\n" + "=" * 60)
print("Métricas de Desempenho:")
print(f"MAE: {mean_absolute_error(y_test, y_pred):,.2f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):,.2f}")
print(f"R²: {resultados.rsquared_adj:.3f}")

# Plot results
plt.figure(figsize=(14, 6))
plt.plot(train['Mês'], y_train, label='Treino', marker='o')
plt.plot(test['Mês'], y_test, label='Real', marker='o', color='orange')
plt.plot(test['Mês'], y_pred, label='Previsão OLS', linestyle='--', marker='x', color='red')
plt.title('Previsão de Demanda com Modelo OLS (Todas as Variáveis)', fontsize=14)
plt.xlabel('Data', fontsize=12)
plt.ylabel('Unidades Vendidas', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()