import pandas as pd
from prevision import train_test

file_path = 'C:/Users/luiga/Documents/python-final-project/car_regressor/data/car_data.csv'

try:

    data = pd.read_csv(file_path, sep=',', encoding='utf-8')
    
    print("Dados carregados com sucesso!")
    print(data.head())

    params_Ridge = {
    'alpha': [1.0],  # Regularização padrão. Pode variar de 0 (sem regularização) até valores muito altos (>1000) para maior penalização.
    'fit_intercept': [True],  # True para ajustar o intercepto, False para forçá-lo a 0. Útil para dados já centralizados.
    'solver': ['auto'],  # Escolha automática do solver. Opções: ['auto', 'svd', 'cholesky', 'saga', 'lsqr', etc.].
    }
    params_Lasso = {
    'alpha': [1.0],  # Regularização padrão. Pode variar de 0 (sem regularização) a valores altos (>1000). 
    'fit_intercept': [True],  # True para ajustar o intercepto; False para forçá-lo a 0.
    'selection': ['cyclic'],  # Método para atualizar coeficientes: 'cyclic' ou 'random'. Random é mais eficiente em grandes dados.
    }
    params_DecisionTree = {
    'criterion': ['squared_error'],  # Função de erro: ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'].
    'max_depth': [None],  # Sem limite por padrão. Pode variar de 1 (árvore rasa) a valores maiores (ex.: 30) para mais profundidade.
    'min_samples_split': [2],  # Mínimo de amostras para dividir. Pode variar de 2 a valores maiores (ex.: 10) para limitar divisões.
    'min_samples_leaf': [1],  # Mínimo de amostras em cada folha. Pode variar de 1 a valores maiores (ex.: 5) para evitar overfitting.
    'max_features': [None]  # Quantas features considerar: None (todas), 'sqrt' (raiz quadrada), 'log2' ou número inteiro específico.
    }
    params_RandomForest = {
        'n_estimators': [100],  # Número de árvores. Pode variar de 10 (rápido) a valores altos (ex.: 1000) para maior precisão.
        'criterion': ['squared_error'],  # Função de erro: ['squared_error', 'absolute_error', 'poisson'].
        'max_depth': [None],  # Sem limite de profundidade. Pode variar de 1 a valores altos (ex.: 50).
        'min_samples_split': [2],  # Mínimo de amostras para dividir. Pode variar de 2 a valores maiores (ex.: 20).
        'min_samples_leaf': [1],  # Mínimo de amostras por folha. Valores maiores (>1) reduzem overfitting.
        'max_features': ['sqrt'],  # Quantidade de features usadas: ['sqrt', 'log2', None, número inteiro].
        'bootstrap': [True]  # Usa amostragem com reposição (True) ou sem (False). Sem reposição pode melhorar diversidade em dados pequenos.
    }
    params_GradientBoosting = {
        'n_estimators': [100],  # Número de árvores. Pode variar de 10 a valores altos (ex.: 500). Aumentar pode reduzir underfitting.
        'learning_rate': [0.1],  # Taxa de aprendizado. Pode variar de valores baixos (ex.: 0.01) a altos (ex.: 0.5). Taxas menores são mais precisas.
        'max_depth': [3],  # Profundidade máxima de cada árvore. Pode variar de 1 (simples) a 10 (mais complexa).
        'min_samples_split': [2],  # Mínimo de amostras para divisão. Valores altos (ex.: 10) limitam divisões e reduzem overfitting.
        'min_samples_leaf': [1],  # Mínimo de amostras em cada folha. Pode variar de 1 a valores maiores (ex.: 5).
        'subsample': [1.0],  # Proporção das amostras usadas em cada árvore. Pode variar de 0.5 a 1.0 para introduzir aleatoriedade.
        'loss': ['squared_error']  # Função de perda: ['squared_error', 'absolute_error', 'huber', 'quantile'].
    }

    sucesso = train_test(data, params_Ridge, params_Lasso, params_DecisionTree, params_RandomForest, params_GradientBoosting)
    
    if sucesso:
        print("Treinamento dos modelos realizado com sucesso!")
    else:
        print("Erro no treinamento dos modelos.")

except FileNotFoundError:
    print(f"Erro: O arquivo '{file_path}' não foi encontrado.")
except pd.errors.ParserError:
    print(f"Erro ao analisar o arquivo. Verifique o formato do arquivo '{file_path}'.")
except Exception as e:
    print(f"Erro inesperado: {e}")