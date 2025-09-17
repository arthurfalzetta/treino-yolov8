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
            contagem = {"column": 0, "beam": 0, "slab": 0, "person": 0, "wall": 0, "crane": 0, "lift car": 0, "ring clurch": 0, "stair": 0, "trailer": 0, "upright": 0}
            for c in r.boxes.cls:
                elem = model.names[int(c)]
                contagem[elem] += 1

            dados.append({
                "imagem": file,
                "colunas_detectadas": contagem["column"],
                "vigas_detectadas": contagem["beam"],
                "lajes_detectadas": contagem["slab"],
                "pessoas_detectadas": contagem["person"],
                "paredes_detectadas": contagem["wall"],
                "guindastes_detectados": contagem["crane"],
                "elevadores_detectados": contagem["lift car"],
                "escadas_detectadas": contagem["stair"],
                "carretas_detectadas": contagem["trailer"],
                "estruturas_verticais_detectadas": contagem["upright"],
                "estruturas_anel_detectadas": contagem["ring clurch"]
            })

df = pd.Dataframe(dados)
df.to_csv("relatorio_detectado.csv", index=False)

print("Relatório salvo em relatorio_detectado.csv")