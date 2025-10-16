# FROM python:3.10-slim

# RUN apt-get update && apt-get install -y \
#     git build-essential python3-dev libgl1 libglib2.0-0 \
#     && rm -rf /var/lib/apt/lists/*

# WORKDIR /app
# COPY . /app

# RUN pip install --no-cache-dir -r requirements.txt \
#     && pip install flask

# ENV PYTHONUNBUFFERED=1

# # Use string simples para o Render
# CMD python servidor.py

# Imagem base própria da Lambda
# Usa o runtime oficial da AWS Lambda para Python 3.10
# Base oficial da AWS Lambda para Python 3.10
FROM python:3.10-slim

# Instala dependências do sistema (necessárias para ifcopenshell e cv2)
RUN apt-get update && apt-get install -y \
    git build-essential python3-dev libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

# Instala dependências Python
RUN pip install --no-cache-dir -r requirements.txt

# Instala o runtime da Lambda (para rodar como container Lambda)
RUN pip install awslambdaric

# Define o handler principal (ex: app.handler)
CMD ["python3", "-m", "awslambdaric", "app.handler"]



