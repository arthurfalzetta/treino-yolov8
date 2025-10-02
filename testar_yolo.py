import ifcopenshell
from ultralytics import YOLO

# === 1. Carregar IFC e contar colunas ===
ifc_file = ifcopenshell.open("teste.ifc")
colunas_ifc = ifc_file.by_type("IfcColumn")
total_colunas_ifc = len(colunas_ifc)

# === 2. Rodar YOLO na foto ===
image_path = "canteiro-de-obras.jpg"
model = YOLO("meu_modelo_yolo/content/runs/detect/train/weights/best.pt")

results = model(image_path, save=True, project="resultados", name="deteccao", exist_ok=True)

# === 3. Filtrar apenas classe "column" ===
detected_classes = results[0].boxes.cls.cpu().numpy().astype(int)
class_names = results[0].names
colunas_detectadas = sum(1 for c in detected_classes if class_names[c] == "column")

# === 4. Comparação ===
print("\n📊 Comparação simples:")
print(f" - Colunas no BIM (IFC): {total_colunas_ifc}")
print(f" - Colunas detectadas pelo YOLO: {colunas_detectadas}")
print("🖼️ Resultado salvo em: resultados/deteccao/predict/")

