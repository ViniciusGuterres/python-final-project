from django import forms

class UploadCSVForm(forms.Form):
    file = forms.FileField(label="Selecione o arquivo CSV", widget=forms.ClearableFileInput(attrs={'accept': '.csv'}))
