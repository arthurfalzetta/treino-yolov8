import os
import csv
import ifcopenshell
from ultralytics import YOLO

# === 1. Carregar IFC e contar colunas ===
ifc_file = ifcopenshell.open("teste.ifc")
colunas_ifc = ifc_file.by_type("IfcColumn")
total_colunas_ifc = len(colunas_ifc)

# === 2. Configurações ===
image_folder = "imagens"
output_folder = "resultados/deteccao"
csv_file = "resultados/resultado_colunas.csv"

model = YOLO("meu_modelo_yolo/content/runs/detect/train/weights/best.pt")

os.makedirs(output_folder, exist_ok=True)

# Lista todas as imagens na pasta (ordenadas)
images = sorted([f for f in os.listdir(image_folder) if f.lower().endswith((".jpg", ".png", ".jpeg"))])

with open(csv_file, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["imagem", "colunas_detectadas", "percentual_ifc", "resultado_imagem"])

    for img_name in images:
        image_path = os.path.join(image_folder, img_name)
        
        # Rodar YOLO e salvar diretamente na pasta de saída, sobrescrevendo se necessário
        results = model(image_path, save=True, project=output_folder, exist_ok=True)
        
        # Filtrar apenas classe "column"
        detected_classes = results[0].boxes.cls.cpu().numpy().astype(int)
        class_names = results[0].names
        colunas_detectadas = sum(1 for c in detected_classes if class_names[c] == "column")
        
        # Percentual em relação ao IFC
        percentual = (colunas_detectadas / total_colunas_ifc) * 100 if total_colunas_ifc > 0 else 0
        
        # Caminho da imagem resultante (YOLO salva na mesma pasta com mesmo nome da original)
        result_img_path = os.path.join(output_folder, img_name)
        
        writer.writerow([img_name, colunas_detectadas, round(percentual, 2), result_img_path])
        print(f"{img_name} -> {colunas_detectadas} colunas detectadas ({round(percentual,2)}%), imagem salva em {result_img_path}")

print("\n✅ Processamento concluído! CSV salvo em:", csv_file)
