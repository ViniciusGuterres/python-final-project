import os
import csv
from django.shortcuts import render, redirect
from django.core.exceptions import ValidationError
from django.conf import settings
from .forms import UploadCSVForm, MLParameterForm

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

def dashboard(request):
    print('dashboard')
    return render(request, 'dashboard.html')

def configure_parameters(request):
    parameter_form = MLParameterForm()

    if request.method == 'POST':  # Handle parameter submission
        parameter_form = MLParameterForm(request.POST)
        if parameter_form.is_valid():
            parameters = parameter_form.cleaned_data
            # Process parameters (e.g., save for analysis or redirect)
            return redirect('dashboard')  # Redirect to the dashboard or another page

    # Always render the form if it's not a POST request or if the form is invalid
    return render(request, 'configure_parameters.html', {'parameter_form': parameter_form})