import datetime
import os
import csv
import json
import ifcopenshell
import boto3
from inference_sdk import InferenceHTTPClient
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from concurrent.futures import ThreadPoolExecutor
from supabase import create_client, Client


load_dotenv()
app = Flask(__name__)

@app.route("/run", methods=["POST"])
def run():
    inicio = datetime.datetime.now()
    try:
        data = request.get_json(force=True)
        IFC_KEY = data.get("ifc_key")
        IMAGES_PREFIX = data.get("images_prefix")
        IMAGE_KEYS = data.get("image_keys")  # lista opcional de keys

        if not IFC_KEY:
            return jsonify({"error": "Parâmetro 'ifc_key' obrigatório"}), 400
        if not IMAGES_PREFIX and not IMAGE_KEYS:
            return jsonify({"error": "Informe 'images_prefix' ou 'image_keys'"}), 400

        # Credenciais
        S3_ENDPOINT = os.getenv("S3_ENDPOINT")
        S3_BUCKET = os.getenv("S3_BUCKET")
        S3_KEY = os.getenv("S3_KEY")
        S3_SECRET = os.getenv("S3_SECRET")
        ROBOFLOW_KEY = os.getenv("ROBOFLOW_KEY")

        s3 = boto3.client(
            "s3",
            endpoint_url=S3_ENDPOINT,
            aws_access_key_id=S3_KEY,
            aws_secret_access_key=S3_SECRET
        )

        # === Baixar IFC ===
        ifc_obj = s3.get_object(Bucket=S3_BUCKET, Key=IFC_KEY)
        ifc_path = "/tmp/temp.ifc"
        with open(ifc_path, "wb") as f:
            f.write(ifc_obj["Body"].read())

        ifc_file = ifcopenshell.open(ifc_path)
        colunas_ifc = ifc_file.by_type("IfcColumn")
        total_colunas_ifc = len(colunas_ifc)
        print(f"🏗️ Total de colunas no IFC: {total_colunas_ifc}")

        # Roboflow client
        client = InferenceHTTPClient(
            api_url="https://serverless.roboflow.com",
            api_key=ROBOFLOW_KEY
        )

        # === Definir keys de imagens ===
        if IMAGE_KEYS:
            image_keys = IMAGE_KEYS
        else:
            objs = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=IMAGES_PREFIX)
            image_keys = sorted([
                obj['Key'] for obj in objs.get('Contents', [])
                if obj['Key'].lower().endswith(('.jpg', '.jpeg', '.png'))
            ])
        print(f"🖼️ {len(image_keys)} imagens selecionadas")

        # Baixar imagens para /tmp em paralelo
        def download_image(key):
            obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
            tmp_path = f"/tmp/{os.path.basename(key)}"
            with open(tmp_path, "wb") as f:
                f.write(obj["Body"].read())
            return tmp_path

        with ThreadPoolExecutor(max_workers=8) as executor:
            image_paths = list(executor.map(download_image, image_keys))

        print(f"🔎 Processando {len(image_paths)} imagens no Roboflow...")

        # Rodar workflow com todos os paths
        try:
            results = client.run_workflow(
                workspace_name="pi-eeksi",
                workflow_id="detect-count-and-visualize",
                images={"image": image_paths},
                use_cache=True
            )

            # Normalizar resultados
            if isinstance(results, str):
                results = json.loads(results)
            elif isinstance(results, list):
                for i in range(len(results)):
                    if isinstance(results[i], str):
                        results[i] = json.loads(results[i])
        except Exception as e:
            print(f"⚠️ Erro Roboflow: {e}")
            results = [{}] * len(image_paths)

        # Criar CSV
        csv_path = "/tmp/resultado_colunas.csv"
        soma_percentuais = 0  # Acumulador para a soma das porcentagens
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["imagem", "colunas_detectadas", "percentual_ifc"])
            for img_path, r in zip(image_paths, results):
                if not isinstance(r, dict):
                    r = {}
                colunas_detectadas = r.get("count_objects", 0)
                percentual = (colunas_detectadas / total_colunas_ifc) * 100 if total_colunas_ifc > 0 else 0
                soma_percentuais += percentual  # Soma os percentuais
                writer.writerow([img_path, colunas_detectadas, round(percentual, 2)])
                print(f"📊 {img_path}: {colunas_detectadas} colunas ({round(percentual,2)}%)")

        # Subir CSV S3
        s3.upload_file(csv_path, S3_BUCKET, "resultados/resultado_colunas.csv")

        duracao = datetime.datetime.now() - inicio
        print(f"🚀 Concluído em {duracao}")
        soma_percentuais_float = float(soma_percentuais)

        resultado_json = {
            "mensagem": "Processamento concluído",
            "csv": "resultados/resultado_colunas.csv",
            "duracao": str(duracao),
            "soma_percentuais": soma_percentuais_float
        }

        return jsonify(resultado_json)

    except Exception as e:
        print(f"❌ Erro geral: {e}")
        return jsonify({"erro": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
