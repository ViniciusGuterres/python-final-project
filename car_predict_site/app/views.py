import os
import csv
import pandas as pd
from django.shortcuts import render, redirect
from django.core.exceptions import ValidationError
from django.conf import settings
from .forms import UploadCSVForm, MLParameterForm, CarPredictionForm
from prevision.prevision import ML_function, DA_function
import joblib

def index(request):
    if request.method == 'POST':
        form = UploadCSVForm(request.POST, request.FILES)
        if form.is_valid():
            csv_file = request.FILES['file']
            if not csv_file.name.endswith('.csv'):
                return render(request, 'index.html', {'form': form, 'error': 'Por favor, envie um arquivo CSV.'})

            try:
                csv_file.seek(0)
                decoded_file = csv_file.read().decode('utf-8').splitlines()
                csv_file.seek(0)
                reader = csv.DictReader(decoded_file)
                expected_headers = [
                    "nome_carro", "ano", "preco_venda", "preco_atual", "kms_rodados",
                    "tipo_combustivel", "tipo_vendedor", "tipo_transmissao", "n_donos"
                ]

                if reader.fieldnames != expected_headers:
                    print('chegou error')
                    raise ValidationError('O arquivo CSV deve conter os cabeçalhos esperados.')

                save_path = os.path.join(settings.MEDIA_ROOT, 'uploads', csv_file.name)
                with open(save_path, 'wb') as f:
                    for chunk in csv_file.chunks():
                        f.write(chunk)

                return redirect('configure_parameters')
            except Exception as e:
                return render(request, 'index.html', {'form': form, 'error': str(e)})
    else:
        form = UploadCSVForm()

    return render(request, 'index.html', {'form': form})

def car_prediction(request):
    resultado = None
    if request.method == "POST":
        form = CarPredictionForm(request.POST)
        if form.is_valid():

            modelo_escolhido = form.cleaned_data['modelo']
            ano = form.cleaned_data['ano']
            preco_atual = form.cleaned_data['preco_atual']
            kms_rodados = form.cleaned_data['kms_rodados']
            n_donos = form.cleaned_data['n_donos']
            tipo_transmissao_manual = int(form.cleaned_data['tipo_transmissao_manual'])
            tipo_vendedor_revendedor = int(form.cleaned_data['tipo_vendedor_revendedor'])
            combustivel_gas_natural = int(form.cleaned_data['combustivel_gas_natural'])
            combustivel_gasolina = int(form.cleaned_data['combustivel_gasolina'])

            model_path = f"prevision/model/{modelo_escolhido}.pkl"
            modelo = joblib.load(model_path)

            dados = pd.DataFrame({
                'ano': [ano],
                'kms_rodados': [kms_rodados],
                'n_donos': [n_donos],
                'tipo_transmissao_Manual': [tipo_transmissao_manual],
                'tipo_vendedor_Revendedor': [tipo_vendedor_revendedor],
                'tipo_combustivel_GasNatural': [combustivel_gas_natural],
                'tipo_combustivel_Gasolina': [combustivel_gasolina]
            })

            pct_preco_venda = modelo.predict(dados)[0]
            preco_venda = preco_atual * (1 + pct_preco_venda / 100)

            resultado = {
                'pct_preco_venda': pct_preco_venda,
                'preco_venda': preco_venda,
            }
    else:
        form = CarPredictionForm()

    return render(request, 'car_prediction.html', {'form': form, 'resultado': resultado})

def dashboard(request):
    try:
        file_path = './media/uploads/car_data.csv'
        data = pd.read_csv(file_path, sep=',', encoding='utf-8')
    except FileNotFoundError:
       return redirect('index') 
    
    DA_function(data)
    
    return render(request, 'dashboard.html')

def configure_parameters(request):
    parameter_form = MLParameterForm()

    if request.method == 'POST':
        parameter_form = MLParameterForm(request.POST)
        if parameter_form.is_valid():
            try:
                data = pd.read_csv('./media/uploads/car_data.csv', sep=',', encoding='utf-8')

                params = parameter_form.cleaned_data

                params_Ridge = {
                    'alpha': [params['ridge_alpha']],
                    'fit_intercept': [params['ridge_fit_intercept']],
                    'solver': ['auto'],
                }
                params_Lasso = {
                    'alpha': [params['lasso_alpha']],
                    'fit_intercept': [params['lasso_fit_intercept']],
                    'selection': ['cyclic'],
                }
                params_DecisionTree = {
                    'criterion': ['squared_error'],
                    'max_depth': [params['decision_tree_max_depth']],
                    'min_samples_split': [params['decision_tree_min_samples_split']],
                    'min_samples_leaf': [params['decision_tree_min_samples_leaf']],
                    'max_features': [params['decision_tree_max_features']],
                }
                params_RandomForest = {
                    'n_estimators': [params['random_forest_n_estimators']],
                    'criterion': ['squared_error'],
                    'max_depth': [params['random_forest_max_depth']],
                    'min_samples_split': [params['random_forest_min_samples_split']],
                    'min_samples_leaf': [params['random_forest_min_samples_leaf']],
                    'max_features': ['sqrt'],
                    'bootstrap': [True],
                }
                params_GradientBoosting = {
                    'n_estimators': [params['gradient_boosting_n_estimators']],
                    'learning_rate': [params['gradient_boosting_learning_rate']],
                    'max_depth': [params['gradient_boosting_max_depth']],
                    'min_samples_split': [params['gradient_boosting_min_samples_split']],
                    'min_samples_leaf': [params['gradient_boosting_min_samples_leaf']],
                    'subsample': [1.0],
                    'loss': ['squared_error'],
                }

                sucesso = ML_function(
                    data,
                    params_Ridge,
                    params_Lasso,
                    params_DecisionTree,
                    params_RandomForest,
                    params_GradientBoosting
                )

                if sucesso:
                    return redirect('dashboard')
                else:
                    return redirect('dashboard')

            except Exception as e:
                return redirect('dashboard')


    return render(request, 'configure_parameters.html', {'parameter_form': parameter_form})
