import os
import csv
from django.shortcuts import render, redirect
from django.core.exceptions import ValidationError
from django.conf import settings
from .forms import UploadCSVForm

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

                return redirect('dashboard')
            except Exception as e:
                return render(request, 'index.html', {'form': form, 'error': str(e)})
    else:
        form = UploadCSVForm()

    return render(request, 'index.html', {'form': form})


def dashboard(request):
    print('dashboard')
    return render(request, 'dashboard.html')
