from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import pygal
from pygal.style import Style
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import os

def ML_function(data, 
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
    y = df['pct_preco_venda']  # Mudança para Series 1D
    x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.2, random_state=42)
    '''
        Preparação dos modelos com os seus parâmetros
    '''
    models = {
        "Ridge Regression": (Ridge(), params_Ridge),
        "Lasso Regression": (Lasso(), params_Lasso),
        "Decision Tree Regressor": (DecisionTreeRegressor(), params_DecisionTree),
        "Random Forest Regressor": (RandomForestRegressor(), params_RandomForest),
        "Gradient Boosting Regressor": (GradientBoostingRegressor(), params_GradientBoosting)
    }
    '''
        Variável para armazenar os dados dos treinamentos e uma variável contador
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
        model_path = f'./prevision/model/{model_id}_{name.replace(" ", "_")}.pkl'
        joblib.dump(best_model_found, model_path)
        '''
            Avaliação do modelo
        '''
        y_pred = best_model_found.predict(x_teste)
        y_pred_series = pd.Series(y_pred)  # Garantir que y_pred seja uma Series
        r2 = r2_score(y_teste, y_pred_series)
        mae = mean_absolute_error(y_teste, y_pred_series)
        mse = mean_squared_error(y_teste, y_pred_series)
        rmse = mse**0.5
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
            Gráfico de comparação usando Pygal
        '''
        comparison_chart = pygal.Line(style=Style(colors=['#FF5733', '#33FF57']))
        comparison_chart.title = f'{name} - Comparação entre Preço Previsto e Real'
        comparison_chart.x_labels = [str(i) for i in range(1, len(y_teste) + 1)]
        comparison_chart.add('Preço Real', y_teste.tolist())
        comparison_chart.add('Preço Previsto', y_pred_series.tolist())
        comparison_chart.render_to_file(f'./app/static/img_train/{model_id}_{name.replace(" ", "_")}/{model_id}_{name.replace(" ", "_")}_comparacao_preco.svg')
        '''
            Gráfico de importância das características usando Pygal
        '''
        if hasattr(best_model_found, "feature_importances_"):
            importances = best_model_found.feature_importances_
            feature_names = x.columns
            sorted_indices = importances.argsort()[::-1]
            sorted_importances = importances[sorted_indices]
            sorted_feature_names = feature_names[sorted_indices]
            importance_chart = pygal.HorizontalBar(style=Style(colors=['#33FF57', '#FF5733']))
            importance_chart.title = f'{name} - Importância das Características'
            for i in range(len(sorted_importances)):
                importance_chart.add(sorted_feature_names[i], sorted_importances[i])
            importance_chart.render_to_file(f'./app/static/img_train/{model_id}_{name.replace(" ", "_")}/{model_id}_{name.replace(" ", "_")}_importancia_atributos.svg')
        '''
            Incrementar o ID para o próximo modelo
        '''
        model_id += 1
    '''
        Salvar os resultados em um arquivo CSV
    '''
    results_df = pd.DataFrame(results)
    results_df.to_csv('./prevision/metrics/metrics.csv', index=False)

    return results

def DA_function(data):
    output_path = './app/static/img_analysis'

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    df = pd.DataFrame(data)
    '''
      Gráfico de linha: Dimensão=ano, Medida=AVG(preco_venda), AVG(preco_atual)
    '''
    avg_prices_by_year = df.groupby('ano')[['preco_venda', 'preco_atual']].mean()
    line_chart = pygal.Line(style=Style(colors=['#FF5733', '#33FF57']))
    line_chart.title = 'Média de Preço por Ano'
    line_chart.x_labels = avg_prices_by_year.index.astype(str)
    line_chart.add('Preço de Venda Médio', avg_prices_by_year['preco_venda'].tolist())
    line_chart.add('Preço Atual Médio', avg_prices_by_year['preco_atual'].tolist())
    line_chart.render_to_file(os.path.join(output_path, 'media_preco_por_ano.svg'))
    '''
      Gráfico de barra: Dimensão=tipo_combustivel, Medida=COUNT(tipo_combustivel)
    '''
    fuel_type_counts = df['tipo_combustivel'].value_counts()
    bar_chart_fuel = pygal.Bar(style=Style(colors=['#33FF57', '#FF5733']))
    bar_chart_fuel.title = 'Quantidade por Tipo de Combustível'
    for fuel_type, count in fuel_type_counts.items():
        bar_chart_fuel.add(fuel_type, count)
    bar_chart_fuel.render_to_file(os.path.join(output_path, 'quantidade_tipo_combustivel.svg'))
    '''
      Gráfico de barra: Dimensão=tipo_transmissao, Medida=COUNT(tipo_transmissao)
    '''
    transmission_type_counts = df['tipo_transmissao'].value_counts()
    bar_chart_transmission = pygal.Bar(style=Style(colors=['#33FF57', '#FF5733']))
    bar_chart_transmission.title = 'Quantidade por Tipo de Transmissão'
    for transmission_type, count in transmission_type_counts.items():
        bar_chart_transmission.add(transmission_type, count)
    bar_chart_transmission.render_to_file(os.path.join(output_path, 'quantidade_tipo_transmissao.svg'))
    '''
      KPI: Quantidade de carros
    '''
    total_cars = len(df)
    kpi_chart = pygal.SolidGauge(inner_radius=0.7, style=Style(colors=['#FF5733']))
    kpi_chart.title = 'Total de Carros no Dataset'
    kpi_chart.add('Total', [{'value': total_cars, 'max_value': total_cars}])
    kpi_chart.render_to_file(os.path.join(output_path, 'total_carros_kpi.svg'))
    
    print(f'Gráficos criados e salvos em: {output_path}')
