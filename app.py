import datetime, os, csv, json, ifcopenshell, boto3
from inference_sdk import InferenceHTTPClient
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor

load_dotenv()  # Carrega variáveis do .env

def handler(event, context):
    agora = datetime.datetime.now()
    try:
        # === 1. Configurações S3 ===
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

        # === 2. Baixar IFC dinâmico ===
        ifc_key = event.get("ifc_key")
        if not ifc_key:
            raise ValueError("ifc_key não informado no body")
        ifc_obj = s3.get_object(Bucket=S3_BUCKET, Key=ifc_key)
        ifc_path = "/tmp/teste.ifc"
        with open(ifc_path, "wb") as f:
            f.write(ifc_obj["Body"].read())

        ifc_file = ifcopenshell.open(ifc_path)
        colunas_ifc = ifc_file.by_type("IfcColumn")
        total_colunas_ifc = len(colunas_ifc)

        # === 3. Inicializar cliente Roboflow ===
        client = InferenceHTTPClient(
            api_url="https://serverless.roboflow.com",
            api_key=ROBOFLOW_KEY
        )

        # === 4. Baixar imagens dinâmicas ===
        image_keys = event.get("image_keys")  # lista de paths no S3
        if not image_keys:
            images_prefix = event.get("images_prefix")
            if not images_prefix:
                raise ValueError("image_keys ou images_prefix devem ser informados")
            objs = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=images_prefix)
            image_keys = sorted([obj['Key'] for obj in objs.get('Contents', []) 
                                 if obj['Key'].lower().endswith(('.jpg','.png','.jpeg'))])

        # Função para baixar uma imagem
        def download_image(key):
            img_obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
            tmp_path = f"/tmp/{os.path.basename(key)}"
            with open(tmp_path, "wb") as f:
                f.write(img_obj["Body"].read())
            return tmp_path

        # Baixar em paralelo
        with ThreadPoolExecutor(max_workers=8) as executor:
            image_paths = list(executor.map(download_image, image_keys))

        print(f"🔎 Processando {len(image_paths)} imagens de uma vez...")

        # === 5. Rodar workflow Roboflow ===
        try:
            results = client.run_workflow(
                workspace_name="pi-eeksi",
                workflow_id="detect-count-and-visualize",
                images={"image": image_paths},
                use_cache=True
            )
            results = [json.loads(r) if isinstance(r, str) else r for r in results]
        except Exception as e:
            print(f"⚠️ Erro na API Roboflow: {e}")
            results = [{}] * len(image_paths)

        # === 6. Criar CSV ===
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

        # === 7. Subir CSV pro S3 ===
        s3.upload_file(csv_file, S3_BUCKET, "resultados/resultado_colunas.csv")

        fim = datetime.datetime.now()
        duracao = fim - agora
        print(f"⏱ Duração total: {duracao}")

        return {
            "statusCode": 200,
            "body": json.dumps({"message": "Processamento concluído", "csv": "resultados/resultado_colunas.csv"})
        }

    except Exception as e:
        print(f"❌ Erro geral: {e}")
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }
