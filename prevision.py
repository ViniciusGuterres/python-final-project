import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import joblib
import seaborn as sns

def train_test(data, 
               params_Ridge, 
               params_Lasso, 
               params_DecisionTree, 
               params_RandomForest, 
               params_GradientBoosting):
    '''
        Tratamento do dataset - criando o atributo 'pct_preco_venda' 
        com a porcentagem da desvalorização e eliminação dos atributos
        'preco_atual' e 'preco_venda'. O 'pct_preco_venda' substituiu 
        essa coluna.
    '''
    df = pd.DataFrame(data)
    df = pd.get_dummies(df, columns=['tipo_transmissao', 'tipo_vendedor', 'tipo_combustivel'], drop_first=True)
    df['pct_preco_venda'] = ((df['preco_venda'] - df['preco_atual']) / df['preco_atual']) * 100
    df = df.drop(columns=['preco_atual', 'preco_venda'])
    '''
        Divisão dos atributos de treinamento e teste
    '''
    x = df[['ano', 'kms_rodados', 'n_donos', 'tipo_transmissao_Manual', 'tipo_vendedor_Revendedor', 'tipo_combustivel_GasNatural', 'tipo_combustivel_Gasolina']]
    y = df[['pct_preco_venda']]
    x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.2, random_state=42)
    '''
        preparação dos modelos com os seu parâmetros
    '''
    models = {
        "Ridge Regression": (Ridge(), params_Ridge),
        "Lasso Regression": (Lasso(), params_Lasso),
        "Decision Tree Regressor": (DecisionTreeRegressor(), params_DecisionTree),
        "Random Forest Regressor": (RandomForestRegressor(), params_RandomForest),
        "Gradient Boosting Regressor": (GradientBoostingRegressor(), params_GradientBoosting)
    }
    '''
        variavel para armazenas os dados dos treinamentos e uma variavel contador
    '''
    results = []
    model_id = 1
    '''
        Loop para realizar os treinamentos e todo o processo de armazenamento dos dados
    '''
    for name, (model, params) in models.items():
        print(f"Treinando o modelo: {name}")
        if params:
            grid_search = GridSearchCV(model, params, cv=5)
            grid_search.fit(x_treino, y_treino)
            best_model_found = grid_search.best_estimator_
        else:
            best_model_found = model
            best_model_found.fit(x_treino, y_treino)
        '''
            Salvar cada modelo individualmente
        '''
        model_path = f'./car_regressor/model/{name.replace(" ", "_")}_model_{model_id}.pkl'
        joblib.dump(best_model_found, model_path)
        '''
            Avaliação do modelo
        '''
        y_pred = best_model_found.predict(x_teste)
        r2 = r2_score(y_teste, y_pred)
        mae = mean_absolute_error(y_teste, y_pred)
        mse = mean_squared_error(y_teste, y_pred)
        rmse = mean_squared_error(y_teste, y_pred, squared=False)
        '''
            Armazenamento dos resultados
        '''
        results.append({
            'ID': model_id,
            'Model': name,
            'R2 Score': r2,
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'Saved_Model_Path': model_path
        })
        '''
            Salvar o gráfico de comparação
        '''
        plt.figure(figsize=(10, 6))
        plt.plot(range(y_pred.shape[0]), y_pred, 'r--', label='Preço previsto')
        plt.plot(range(y_teste.shape[0]), y_teste, 'g--', label='Preço real')
        plt.legend()
        plt.xlabel('Índice')
        plt.ylabel('Preços')
        plt.title(f'{name} - Comparação entre Preço Previsto e Real')
        plt.savefig(r'C:\Users\luiga\Documents\python-final-project\car_regressor\img_train\{}_{}_comparacao_preco.png'.format(model_id,name.replace(" ", "_")))
        '''
            Salvar o gráfico de importância
        '''
        if hasattr(best_model_found, "feature_importances_"):
            importances = best_model_found.feature_importances_
            feature_names = x.columns
            sorted_indices = importances.argsort()[::-1]
            sorted_importances = importances[sorted_indices]
            sorted_feature_names = feature_names[sorted_indices]
            importances_df = pd.DataFrame({'Feature': sorted_feature_names, 'Importance': sorted_importances})
            plt.figure(figsize=(12, 8))
            sns.barplot(
                x='Importance',
                y='Feature',
                data=importances_df,
                palette='viridis'
            )
            plt.title(f'{name} - Importância das Características')
            plt.xlabel('Importância')
            plt.ylabel('Características')
            plt.grid(True)
            plt.savefig(r'C:\Users\luiga\Documents\python-final-project\car_regressor\img_train\{}{}_importancia_atributos.png'.format(model_id,name.replace(" ", "_")))
        '''
            Incrementar o ID para o próximo modelo
        '''
        model_id += 1
    '''
        Salvar os resultados em um arquivo CSV
    '''
    results_df = pd.DataFrame(results)
    results_df.to_csv(r'C:\Users\luiga\Documents\python-final-project\car_regressor\metrics\metrics.csv', index=False)

    return results
