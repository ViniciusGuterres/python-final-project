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