from ultralytics import YOLO
import os
import pandas as pd

model = YOLO("runs/detect/train/weights/best.pt")

input_folder = "imagens_teste"

dados = []

for file in os.listdir(input_folder):
    if file.endswith((".jpg", ".png", ".jpeg")):
        path = os.path.join(input_folder, file)

        results = model(path, save=True, project="resultados", name="deteccao", exists_ok=True)

        for r in results:
            contagem = {"coluna": 0, "viga": 0, "laje": 0}
            for c in r.boxes.cls:
                elem = model.names[int(c)]
                contagem[elem] += 1

            dados.append({
                "imagem": file,
                "colunas_detectadas": contagem["coluna"],
                "vigas_detectadas": contagem["viga"],
                "lajes_detectadas": contagem["laje"]
            })

df = pd.Dataframe(dados)
df.to_csv("relatorio_detectado.csv", index=False)

print("Relatório salvo em relatorio_detectado.csv")