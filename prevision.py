# -*- coding: utf-8 -*-
"""CarRegressor.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1dnTlC-feL_OtsQjMEgvRXVgoI6btqZug

## Biblioteca
"""
# pip install pandas scikit-learn matplotlib joblib
'''
  Importação das bibliotecas
'''
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import joblib

"""##Treinamento"""

'''
  Importação do DataSet
'''
data = pd.read_csv('./car_regressor/data/car_data.csv')
'''
  Conversão em DataFrame
'''
df = pd.DataFrame(data)
'''
  Realizar um crosstable com os valores categóricos para representarem 0 ou 1
'''
df = pd.get_dummies(df, columns=['tipo_transmissao', 'tipo_vendedor', 'tipo_combustivel'], drop_first=True)
'''
  Cria um novo atributo de porcentagem de preço de venda com as colunas 'preco_venda' e 'preco_atual'
  obs: ação realizada por desconhecimento da grandeza do campo.
'''
df['pct_preco_venda'] = ((df['preco_venda'] - df['preco_atual']) / df['preco_atual']) * 100
'''
  Remoção das colunas inrrelevantes por conta da alteração
'''
df = df.drop(columns=['preco_atual', 'preco_venda'])
'''
  Definição dos atributos de treinamento (x) e de teste (y).
  Distribuição dos dados para teeinamento e teste
'''
x = df[['ano',
        'kms_rodados',
        'n_donos',
        'tipo_transmissao_Manual',
        'tipo_vendedor_Revendedor',
        'tipo_combustivel_GasNatural',
        'tipo_combustivel_Gasolina']]
y = df[['pct_preco_venda']]
x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.2, random_state=42)
'''
  Realização de varios tipos de treinamentos para obter o com melhor resultados
'''
models = {
    "Ridge Regression": Ridge(),
    "Lasso Regression": Lasso(),
    "Decision Tree Regressor": DecisionTreeRegressor(),
    "Random Forest Regressor": RandomForestRegressor(),
    "Gradient Boosting Regressor": GradientBoostingRegressor()
}
'''
  Definição de parâmetros de otimização
'''
param_grid_ridge = {'alpha': [0.1, 1, 10, 100, 1000]}
param_grid_lasso = {'alpha': [0.1, 1, 10, 100, 1000]}
'''
  Dicionário para armazenar os resultados
'''
results = {}
best_model = None
best_r2_score = -float('inf')
'''
  Treinar e avaliar os modelos
'''
for name, model in models.items():
    print(f"Treinando o modelo: {name}")
    if name == "Ridge Regression":
        grid_search = GridSearchCV(model, param_grid_ridge, cv=5)
        grid_search.fit(x_treino, y_treino)
        best_model_found = grid_search.best_estimator_
    elif name == "Lasso Regression":
        grid_search = GridSearchCV(model, param_grid_lasso, cv=5)
        grid_search.fit(x_treino, y_treino)
        best_model_found = grid_search.best_estimator_
    else:
        '''
          Manter o treinamento convencional para os demais
        '''
        best_model_found = model
        best_model_found.fit(x_treino, y_treino)

    '''
      Realizar previsões
    '''
    y_pred = best_model_found.predict(x_teste)
    '''
      Métricas de avaliação do modelo
    '''
    r2 = r2_score(y_teste, y_pred)
    mae = mean_absolute_error(y_teste, y_pred)
    mse = mean_squared_error(y_teste, y_pred)
    rmse = mean_squared_error(y_teste, y_pred, squared=False)
    '''
      Apresentação dos resultados
    '''
    results[name] = {'R2 Score': r2, 'MAE': mae, 'MSE': mse, 'RMSE': rmse}
    '''
      Apresentação gráfica da regressão de ambos os modelos
    '''
    # plt.figure(figsize=(10, 6))
    # plt.plot(range(y_pred.shape[0]), y_pred, 'r--', label='Preço previsto')
    # plt.plot(range(y_teste.shape[0]), y_teste, 'g--', label='Preço real')
    # plt.legend()
    # plt.xlabel('Índice')
    # plt.ylabel('Preços')
    # plt.title(f'{name} - Comparação entre Preço Previsto e Real')
    # plt.show()
    '''
      Verificar o modelo com o melhor resultado
    '''
    if r2 > best_r2_score:
        best_r2_score = r2
        best_model = best_model_found
        best_model_name = name
'''
  Exibição das metricas...
'''
for name, metrics in results.items():
    print(f"\n{name}:")
    for metric, value in metrics.items():
        print(f"{metric}: {value}")
'''
  Salvamento do modelo
'''
if best_model is not None:
    print(f"\nO melhor modelo é: {best_model_name} com R2 Score: {best_r2_score}")
    joblib.dump(best_model, './car_regressor/model/best_model_car_regressor.pkl')

"""##Testando a previsão do melhor modelo"""

'''
  Carregamento do melhor modelo salvo
'''
# modelo_car_regressor = joblib.load('./car_regressor/model/best_model_car_regressor.pkl')
'''
  Inserção dos dados de input de teste...
'''
# ano = int(input("Digite o ano do carro: "))
# preco_atual = float(input("Digite o preço atual do carro: "))
# kms_rodados = int(input("Digite os quilômetros rodados: "))
# n_donos = int(input("Digite o número de donos anteriores: "))
# tipo_transmissao_manual = int(input("Transmissão manual (1 para sim, 0 para não): "))
# tipo_vendedor_revendedor = int(input("Vendedor revendedor (1 para sim, 0 para não): "))
# combustivel_gas_natural = int(input("Combustível gás natural (1 para sim, 0 para não): "))
# combustivel_gasolina = int(input("Combustível gasolina (1 para sim, 0 para não): "))
'''
  Criando um DataFrame com as respostas dos input
'''
# dados = pd.DataFrame({
#     'ano': [ano],
#     'kms_rodados': [kms_rodados],
#     'n_donos': [n_donos],
#     'tipo_transmissao_Manual': [tipo_transmissao_manual],
#     'tipo_vendedor_Revendedor': [tipo_vendedor_revendedor],
#     'tipo_combustivel_GasNatural': [combustivel_gas_natural],
#     'tipo_combustivel_Gasolina': [combustivel_gasolina]
# })
'''
  Realizar a previsão
'''
# pct_preco_venda = modelo_car_regressor.predict(dados)[0]
'''
  Realizar o calculo do preço atual menos a porcentagem de desconto pela desvalorização do carro
'''
# preco_venda = preco_atual * (1 + pct_preco_venda / 100)
'''
  Exibição de resultado
'''
# print(f'A porcentagem prevista é: {pct_preco_venda:.2f}%')
# print(f'O preço previsto de venda do carro é: R${preco_venda:.2f}')
