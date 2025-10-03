import os
import csv
import json
import ifcopenshell
import boto3
from inference_sdk import InferenceHTTPClient
from dotenv import load_dotenv
import os

load_dotenv() # Carrega variáveis de ambiente do arquivo .env

# === 1. Configurações S3 ===
S3_ENDPOINT = os.getenv("S3_ENDPOINT")
S3_BUCKET = os.getenv("S3_BUCKET")
IFC_KEY = os.getenv("IFC_KEY")
IMAGES_PREFIX = os.getenv("IMAGES_PREFIX")
S3_KEY=os.getenv("S3_KEY")
S3_SECRET=os.getenv("S3_SECRET")
ROBOFLOW_KEY=os.getenv("ROBOFLOW_KEY")
# boto3 client para S3 compatível
s3 = boto3.client(
    "s3",
    endpoint_url=S3_ENDPOINT,
    aws_access_key_id=S3_KEY,
    aws_secret_access_key=S3_SECRET
)

# === 2. Baixar IFC do S3 ===
ifc_obj = s3.get_object(Bucket=S3_BUCKET, Key=IFC_KEY)
ifc_data = ifc_obj["Body"].read()

# ifcopenshell não abre bytes diretamente, então vamos salvar temporariamente na /tmp do Lambda
ifc_path = "/tmp/teste.ifc"
with open(ifc_path, "wb") as f:
    f.write(ifc_data)

ifc_file = ifcopenshell.open(ifc_path)
colunas_ifc = ifc_file.by_type("IfcColumn")
total_colunas_ifc = len(colunas_ifc)

# === 3. Inicializar cliente Roboflow ===
client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=ROBOFLOW_KEY
)

# === 4. Listar imagens no S3 ===
image_objs = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=IMAGES_PREFIX)
image_keys = sorted([obj['Key'] for obj in image_objs.get('Contents', []) if obj['Key'].lower().endswith(('.jpg','.png','.jpeg'))])

# Baixar imagens para /tmp e criar paths
image_paths = []
for key in image_keys:
    img_obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
    img_data = img_obj["Body"].read()
    tmp_path = f"/tmp/{os.path.basename(key)}"
    with open(tmp_path, "wb") as f:
        f.write(img_data)
    image_paths.append(tmp_path)

print(f"🔎 Processando {len(image_paths)} imagens de uma vez...")

# === 5. Rodar workflow Roboflow ===
colunas_detectadas_list = []

try:
    results = client.run_workflow(
        workspace_name="pi-eeksi",
        workflow_id="detect-count-and-visualize",
        images={"image": image_paths},
        use_cache=True
    )

    if isinstance(results, str):
        results = json.loads(results)
    elif isinstance(results, list):
        for i in range(len(results)):
            if isinstance(results[i], str):
                results[i] = json.loads(results[i])

except Exception as e:
    print(f"⚠️ Erro geral na API: {e}")
    results = [{}] * len(image_paths)

# === 6. Criar CSV no /tmp ===
csv_file = "/tmp/resultado_colunas.csv"
with open(csv_file, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["caminho_imagem", "colunas_detectadas", "percentual_ifc"])

    for img_path, r in zip(image_paths, results):
        if not isinstance(r, dict):
            r = {}
        colunas_detectadas = r.get("count_objects", 0)
        percentual = (colunas_detectadas / total_colunas_ifc) * 100 if total_colunas_ifc > 0 else 0
        writer.writerow([img_path, colunas_detectadas, round(percentual, 2)])
        print(f"👉 {img_path} -> {colunas_detectadas} colunas detectadas ({round(percentual,2)}%)")

# === 7. Subir CSV de volta pro S3 ===
s3.upload_file(csv_file, S3_BUCKET, "resultados/resultado_colunas.csv")

print("\n✅ Processamento concluído! CSV salvo em S3 em 'resultados/resultado_colunas.csv'")
