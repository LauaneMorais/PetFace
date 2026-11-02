"""
Gera os arquivos de teste (verification.csv) para a FaunaBR
===========================================================
Cria os pares de teste para cada espécie:
- Pares Positivos (label = 1): fotos do mesmo indivíduo
- Pares Negativos (label = 0): fotos de indivíduos diferentes da mesma espécie
Salva em:
- data/faunaBR/split/onca/verification.csv
- data/faunaBR/split/capivara/verification.csv
- data/faunaBR/split/arara/verification.csv
"""

import os
import re
import itertools
import pandas as pd

BASE_ALIGNED = os.path.join("data", "faunaBR", "aligned_images")
BASE_SPLIT = os.path.join("data", "faunaBR", "split")

def get_individual_name(filename):
    # Extrai o nome do indivíduo antes do número da foto
    # Ex: onca_apache_01.jpg -> onca_apache
    # Ex: onca_gabi_04_cedula.jpg -> onca_gabi
    m = re.match(r"^(.*?)_\d+.*\.jpg$", filename, re.IGNORECASE)
    if m:
        return m.group(1)
    return os.path.splitext(filename)[0]

def generate_verification_csv(species):
    species_dir = os.path.join(BASE_ALIGNED, species)
    if not os.path.isdir(species_dir):
        print(f"Diretório não encontrado: {species_dir}")
        return

    # Listar fotos
    files = sorted([f for f in os.listdir(species_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    
    # Agrupar por indivíduo
    ind_map = {}
    for f in files:
        ind = get_individual_name(f)
        ind_map.setdefault(ind, []).append(f)

    print(f"\n[{species.upper()}] Encontrados {len(ind_map)} indivíduos e {len(files)} fotos:")
    for ind, flist in ind_map.items():
        print(f"   • {ind}: {len(flist)} fotos")

    pairs = []
    
    # 1. Gerar pares POSITIVOS (mesmo indivíduo, label = 1)
    for ind, flist in ind_map.items():
        for f1, f2 in itertools.combinations(flist, 2):
            rel_p1 = f"{species}/{f1}"
            rel_p2 = f"{species}/{f2}"
            pairs.append({"filename1": rel_p1, "filename2": rel_p2, "label": 1})

    num_pos = len(pairs)

    # 2. Gerar pares NEGATIVOS (indivíduos diferentes da mesma espécie, label = 0)
    ind_keys = list(ind_map.keys())
    neg_count = 0
    for i in range(len(ind_keys)):
        for j in range(i + 1, len(ind_keys)):
            ind1 = ind_keys[i]
            ind2 = ind_keys[j]
            for f1 in ind_map[ind1]:
                for f2 in ind_map[ind2]:
                    rel_p1 = f"{species}/{f1}"
                    rel_p2 = f"{species}/{f2}"
                    pairs.append({"filename1": rel_p1, "filename2": rel_p2, "label": 0})
                    neg_count += 1

    df = pd.DataFrame(pairs)
    out_dir = os.path.join(BASE_SPLIT, species)
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "verification.csv")
    df.to_csv(out_file, index=False)

    print(f"   -> Salvo em: {out_file}")
    print(f"   -> Total de Pares: {len(df)} (Positivos: {num_pos} | Negativos: {neg_count})")

def main():
    print("=" * 60)
    print("Gerando arquivos verification.csv para FaunaBR...")
    print("=" * 60)
    for sp in ["onca", "capivara", "arara"]:
        generate_verification_csv(sp)
    print("\n" + "=" * 60)
    print("Concluído com sucesso!")
    print("=" * 60)

if __name__ == "__main__":
    main()

