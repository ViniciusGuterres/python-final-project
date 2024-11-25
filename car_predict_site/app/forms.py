from django import forms

class UploadCSVForm(forms.Form):
    file = forms.FileField(label="Selecione o arquivo CSV", widget=forms.ClearableFileInput(attrs={'accept': '.csv'}))

class MLParameterForm(forms.Form):
    # Ridge Parameters
    ridge_alpha = forms.FloatField(label='Ridge Alpha', required=True)
    ridge_fit_intercept = forms.BooleanField(label='Ridge Fit Intercept', required=False, initial=True)
    decision_tree_max_depth = forms.IntegerField(label='Decision Tree Max Depth', required=False, initial=None)
    
    decision_tree_min_samples_split = forms.IntegerField(label='Decision Tree Min Samples Split', required=True, initial=2)

    ridge_solver = forms.ChoiceField(label="Ridge Solver", choices=[
        ('auto', 'Auto'), ('svd', 'SVD'), ('cholesky', 'Cholesky'), ('saga', 'SAGA'), ('lsqr', 'LSQR')
    ], initial='auto')
    
    # Lasso Parameters
    lasso_alpha = forms.FloatField(label="Lasso Alpha", initial=1.0)
    lasso_fit_intercept = forms.BooleanField(label="Lasso Fit Intercept", initial=True, required=False)
    lasso_selection = forms.ChoiceField(label="Lasso Selection", choices=[
        ('cyclic', 'Cyclic'), ('random', 'Random')
    ], initial='cyclic')


class CarPredictionForm(forms.Form):
    MODELO_CHOICES = [
        ('1_Ridge_Regression', 'Ridge'),
        ('2_Lasso_Regression', 'Lasso'),
        ('3_Decision_Tree_Regressor', 'Decision Tree'),
        ('4_Random_Forest_Regressor', 'Random Forest'),
        ('5_Gradient_Boosting_Regressor', 'Gradient Boosting'),
    ]
    modelo = forms.ChoiceField(choices=MODELO_CHOICES, label="Escolha o modelo")
    ano = forms.IntegerField(label="Ano do carro", min_value=1900, max_value=2024)
    preco_atual = forms.FloatField(label="Preço atual do carro", min_value=0)
    kms_rodados = forms.IntegerField(label="Quilômetros rodados", min_value=0)
    n_donos = forms.IntegerField(label="Número de donos anteriores", min_value=0)
    tipo_transmissao_manual = forms.BooleanField(
        label="Transmissão manual", required=False
    )
    tipo_vendedor_revendedor = forms.BooleanField(
        label="Vendedor revendedor", required=False
    )
    combustivel_gas_natural = forms.BooleanField(
        label="Combustível gás natural", required=False
    )
    combustivel_gasolina = forms.BooleanField(
        label="Combustível gasolina", required=False
    )