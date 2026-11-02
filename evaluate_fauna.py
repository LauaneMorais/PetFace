"""
Benchmark Completo da FaunaBR (PetFace ECCV 2024)
=================================================
Executa a avaliação de biometria facial para as espécies brasileiras:
- Onça-Pintada (cat.pt vs unified.pt)
- Capivara (guineapig.pt vs unified.pt)
- Arara-Vermelha (parakeet.pt vs unified.pt)

Extrai os embeddings com alta velocidade e gera a tabela comparativa de AUC.
"""

import os
import sys
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import pandas as pd
import numpy as np

# Adicionar src ao path para carregar a arquitetura ResNet-50
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from backbones.resnet import r50

# Transformação padrão oficial do PetFace
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

def compute_auc_score(labels, scores):
    y_true = np.asarray(labels)
    y_score = np.maximum(np.asarray(scores), 0)
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    u = sum(np.sum(p > neg) + 0.5 * np.sum(p == neg) for p in pos)
    return (u / (len(pos) * len(neg))) * 100

def load_backbone(weight_path, device):
    model = r50()
    state = torch.load(weight_path, map_location="cpu")
    model.load_state_dict(state["state_dict_backbone"])
    model.to(device)
    model.eval()
    return model

def extract_embeddings_for_species(model, species_dir, device):
    embeddings = {}
    files = sorted([f for f in os.listdir(species_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    with torch.no_grad():
        for f in files:
            p = os.path.join(species_dir, f)
            img = Image.open(p).convert("RGB")
            t = transform(img).unsqueeze(0).to(device)
            emb = F.normalize(model(t)).cpu().squeeze(0)
            embeddings[f] = emb
    return embeddings

def evaluate_pair_split(embeddings, csv_path, species):
    df = pd.read_csv(csv_path)
    sims = []
    labels = df["label"].tolist()

    for _, row in df.iterrows():
        # Obter nome do arquivo
        f1 = os.path.basename(row["filename1"])
        f2 = os.path.basename(row["filename2"])
        e1 = embeddings[f1]
        e2 = embeddings[f2]
        sim = torch.dot(e1, e2).item()
        sims.append(sim)

    df_out = df.copy()
    df_out["sim"] = sims
    auc = compute_auc_score(labels, sims)
    
    pos_sims = [s for s, l in zip(sims, labels) if l == 1]
    neg_sims = [s for s, l in zip(sims, labels) if l == 0]
    mean_pos = np.mean(pos_sims) if pos_sims else 0
    mean_neg = np.mean(neg_sims) if neg_sims else 0

    return auc, mean_pos, mean_neg, df_out

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Executando no dispositivo: {device.upper()}")
    print("=" * 75)

    experiments = [
        # (Espécie, Nome do Modelo, Caminho dos pesos)
        ("onca", "cat (Especialista)", "pretrained/arcface/cat.pt"),
        ("onca", "unified (Generalista)", "pretrained/arcface/unified.pt"),
        ("capivara", "guineapig (Especialista)", "pretrained/arcface/guineapig.pt"),
        ("capivara", "unified (Generalista)", "pretrained/arcface/unified.pt"),
        ("arara", "parakeet (Especialista)", "pretrained/arcface/parakeet.pt"),
        ("arara", "unified (Generalista)", "pretrained/arcface/unified.pt"),
    ]

    results_table = []
    os.makedirs(os.path.join("results", "faunaBR"), exist_ok=True)

    # Cache de modelos para não recarregar várias vezes
    loaded_models = {}

    for species, model_desc, weight_path in experiments:
        if not os.path.exists(weight_path):
            print(f"Pesos não encontrados: {weight_path}")
            continue

        if weight_path not in loaded_models:
            print(f"Carregando pesos: {weight_path}...")
            loaded_models[weight_path] = load_backbone(weight_path, device)
        model = loaded_models[weight_path]

        species_dir = os.path.join("data", "faunaBR", "aligned_images", species)
        csv_split = os.path.join("data", "faunaBR", "split", species, "verification.csv")

        # Extrair embeddings das fotos únicas da espécie
        embeddings = extract_embeddings_for_species(model, species_dir, device)

        # Avaliar pares
        auc, mean_pos, mean_neg, df_res = evaluate_pair_split(embeddings, csv_split, species)

        # Salvar CSV compatível com o formato original do PetFace
        out_name = f"{species}_{os.path.splitext(os.path.basename(weight_path))[0]}.csv"
        out_csv = os.path.join("results", "faunaBR", out_name)
        df_res.to_csv(out_csv, index=False)

        results_table.append({
            "Espécie": species.capitalize(),
            "Modelo": model_desc,
            "AUC (%)": f"{auc:.2f}%",
            "Sim Média (Mesmo ID)": f"{mean_pos:.4f}",
            "Sim Média (IDs Dif)": f"{mean_neg:.4f}",
            "Delta Sim": f"{(mean_pos - mean_neg):+.4f}",
        })

    print("\n" + "=" * 75)
    print("RESULTADOS FINAIS DE BIOMETRIA - FAUNA BRASILEIRA (PetFace ECCV 2024)")
    print("=" * 75)
    summary_df = pd.DataFrame(results_table)
    print(summary_df.to_string(index=False))
    print("=" * 75)
    print("Todos os resultados individuais foram salvos em: results/faunaBR/")

if __name__ == "__main__":
    main()

