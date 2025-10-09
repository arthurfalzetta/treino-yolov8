FROM continuumio/miniconda3

# Criar ambiente
RUN conda create -n py311 python=3.11
SHELL ["conda", "run", "-n", "py311", "/bin/bash", "-c"]

# Instalar ifcopenshell via conda-forge
RUN conda install -c conda-forge ifcopenshell boto3 python-dotenv -y

# Copiar código
COPY app.py /app/app.py
WORKDIR /app

# Rodar Lambda Python runtime
CMD ["app.handler"]
