"""
Gera Gráficos Científicos com Matplotlib para o Benchmark FaunaBR
Produz 4 gráficos em alta resolução (300 DPI) para relatórios, slides e artigos:
1. grafico1_comparativo_auc.png       -> Barras comparando Especialista vs. Generalista
2. grafico2_curvas_roc.png             -> Curvas ROC detalhadas para as 3 espécies
3. grafico3_distribuicao_similaridade.png -> Histogramas e densidade de Pares Positivos vs Negativos
4. painel_completo_faunabr.png         -> Dashboard consolidado 2x2 com todas as análises
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join("results", "faunaBR")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# Configuração visual elegante do Matplotlib
plt.rcParams.update({
    "font.sans-serif": ["Segoe UI", "Arial", "DejaVu Sans"],
    "font.family": "sans-serif",
    "axes.edgecolor": "#CCCCCC",
    "axes.linewidth": 1.0,
    "grid.color": "#E5E5E5",
    "grid.linestyle": "--",
    "grid.alpha": 0.7,
})

def compute_roc_curve(y_true, y_score):
    """Calcula FPR e TPR para a curva ROC sem depender de bibliotecas externas."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    desc_score_indices = np.argsort(y_score)[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    distinct_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_indices, y_true.size - 1]

    tps = np.cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps

    tps = np.r_[0, tps]
    fps = np.r_[0, fps]

    fpr = fps / fps[-1]
    tpr = tps / tps[-1]
    auc = np.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1]) / 2.0) * 100
    return fpr, tpr, auc

# Carregar dados
data = {
    "onca": {
        "cat": pd.read_csv(os.path.join(RESULTS_DIR, "onca_cat.csv")),
        "unified": pd.read_csv(os.path.join(RESULTS_DIR, "onca_unified.csv")),
    },
    "capivara": {
        "guineapig": pd.read_csv(os.path.join(RESULTS_DIR, "capivara_guineapig.csv")),
        "unified": pd.read_csv(os.path.join(RESULTS_DIR, "capivara_unified.csv")),
    },
    "arara": {
        "parakeet": pd.read_csv(os.path.join(RESULTS_DIR, "arara_parakeet.csv")),
        "unified": pd.read_csv(os.path.join(RESULTS_DIR, "arara_unified.csv")),
    },
}

# 
# GRÁFICO 1: Comparativo de AUC (Especialista vs. Generalista)
def plot_auc_comparison(save_path=None):
    species_names = ["Arara-Vermelha", "Capivara", "Onça-Pintada"]
    keys = ["arara", "capivara", "onca"]
    spec_names = ["parakeet", "guineapig", "cat"]
    
    auc_specialist = [compute_roc_curve(data[k][s]["label"], data[k][s]["sim"])[2] for k, s in zip(keys, spec_names)]
    auc_unified = [compute_roc_curve(data[k]["unified"]["label"], data[k]["unified"]["sim"])[2] for k in keys]

    x = np.arange(len(species_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
    rects1 = ax.bar(x - width/2, auc_specialist, width, label="Modelo Especialista da Família", color="#1E88E5", edgecolor="#0D47A1", alpha=0.9, zorder=3)
    rects2 = ax.bar(x + width/2, auc_unified, width, label="Modelo Generalista (Unified 13 Famílias)", color="#7E57C2", edgecolor="#4A148C", alpha=0.9, zorder=3)

    ax.set_ylabel("ROC-AUC (%)", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_title("Comparativo de Biometria: Especialista vs. Generalista (FaunaBR)", fontsize=13, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(species_names, fontsize=11, fontweight="bold")
    ax.set_ylim(0, 108)
    ax.axhline(50, color="#E53935", linestyle=":", linewidth=1.5, label="Nível Aleatório (Baseline 50%)")
    ax.grid(axis="y", zorder=0)
    ax.legend(frameon=True, facecolor="white", edgecolor="#DDD", loc="lower right", fontsize=10)

    # Rótulos nas barras
    def autolabel(rects, deltas=None):
        for idx, rect in enumerate(rects):
            h = rect.get_height()
            ax.annotate(f"{h:.2f}%",
                        xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", va="bottom", fontsize=10, fontweight="bold", color="#222")

    autolabel(rects1)
    autolabel(rects2)

    # Anotações destacando os insights
    ax.annotate("+7.14% ganho com\nfiltro de psitacídeo", xy=(x[0] - width/2, auc_specialist[0]), xytext=(x[0] - width/2 - 0.25, 102),
                arrowprops=dict(arrowstyle="->", color="#1E88E5", lw=1.5), fontsize=8.5, color="#0D47A1", fontweight="bold", ha="center")
    
    ax.annotate("Generalista vence por\ndiversidade de pelagens", xy=(x[2] + width/2, auc_unified[2]), xytext=(x[2] + width/2 + 0.15, 80),
                arrowprops=dict(arrowstyle="->", color="#7E57C2", lw=1.5), fontsize=8.5, color="#4A148C", fontweight="bold", ha="center")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Salvo: {save_path}")
    plt.close()

# GRÁFICO 2: Curvas ROC (Receiver Operating Characteristic)
def plot_roc_curves(save_path=None):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=300, sharey=True)
    configs = [
        ("Arara-Vermelha", "arara", "parakeet", "Especialista (Parakeet)", "#D32F2F"),
        ("Capivara", "capivara", "guineapig", "Especialista (Guinea Pig)", "#388E3C"),
        ("Onça-Pintada", "onca", "cat", "Especialista (Cat)", "#F57C00"),
    ]

    for ax, (name, sp, spec_key, spec_label, col) in zip(axes, configs):
        # Especialista
        df_spec = data[sp][spec_key]
        fpr_s, tpr_s, auc_s = compute_roc_curve(df_spec["label"], df_spec["sim"])
        ax.plot(fpr_s, tpr_s, color=col, linewidth=2.4, label=f"{spec_label}\nAUC = {auc_s:.2f}%")

        # Generalista
        df_uni = data[sp]["unified"]
        fpr_u, tpr_u, auc_u = compute_roc_curve(df_uni["label"], df_uni["sim"])
        ax.plot(fpr_u, tpr_u, color="#5E35B1", linewidth=2.0, linestyle="--", label=f"Generalista (Unified)\nAUC = {auc_u:.2f}%")

        # Linha diagonal aleatória
        ax.plot([0, 1], [0, 1], color="#999999", linestyle=":", linewidth=1.2, label="Aleatório (AUC=50%)")

        ax.set_title(f"{name}", fontsize=12, fontweight="bold", pad=10)
        ax.set_xlabel("Taxa de Falsos Positivos (FPR)", fontsize=10, labelpad=8)
        if ax == axes[0]:
            ax.set_ylabel("Taxa de Verdadeiros Positivos (TPR)", fontsize=10, labelpad=8)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.04)
        ax.grid(True, zorder=0)
        ax.legend(frameon=True, facecolor="white", edgecolor="#E0E0E0", loc="lower right", fontsize=8.5)

    fig.suptitle("Curvas ROC por Espécie - Benchmark FaunaBR (PetFace ArcFace)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Salvo: {save_path}")
    plt.close()

# GRÁFICO 3: Distribuição das Similaridades (Pares Positivos vs. Negativos)
def plot_similarity_distributions(save_path=None):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), dpi=300, sharey=False)
    configs = [
        ("Arara-Vermelha", "arara", "parakeet", "Melhor Modelo: Parakeet"),
        ("Capivara", "capivara", "guineapig", "Melhor Modelo: Guinea Pig"),
        ("Onça-Pintada", "onca", "unified", "Melhor Modelo: Unified"),
    ]

    for ax, (name, sp, best_key, subtitle) in zip(axes, configs):
        df = data[sp][best_key]
        pos = df[df["label"] == 1]["sim"]
        neg = df[df["label"] == 0]["sim"]

        # Histogramas transparentes
        bins = np.linspace(min(df["sim"]) - 0.05, max(df["sim"]) + 0.05, 18)
        ax.hist(neg, bins=bins, color="#E53935", alpha=0.55, edgecolor="#B71C1C", density=True, label=f"IDs Diferentes (N={len(neg)})", zorder=2)
        ax.hist(pos, bins=bins, color="#2E7D32", alpha=0.65, edgecolor="#1B5E20", density=True, label=f"Mesmo ID (N={len(pos)})", zorder=2)

        # Linhas verticais com a média
        mean_neg = neg.mean()
        mean_pos = pos.mean()
        ax.axvline(mean_neg, color="#B71C1C", linestyle="--", linewidth=1.8, label=f"Média Neg: {mean_neg:.2f}")
        ax.axvline(mean_pos, color="#1B5E20", linestyle="--", linewidth=1.8, label=f"Média Pos: {mean_pos:.2f}")

        # Seta do Delta
        ax.annotate(f"Separação\nΔ = +{(mean_pos - mean_neg):.2f}",
                    xy=((mean_neg + mean_pos) / 2, ax.get_ylim()[1] * 0.75 if ax.get_ylim()[1] > 0 else 2),
                    xytext=((mean_neg + mean_pos) / 2, 4.0),
                    ha="center", fontsize=9, fontweight="bold", color="#0D47A1",
                    bbox=dict(boxstyle="round,pad=0.3", fc="#E3F2FD", ec="#90CAF9"))

        ax.set_title(f"{name}\n({subtitle})", fontsize=11, fontweight="bold", pad=8)
        ax.set_xlabel("Similaridade de Cosseno (Dot Product)", fontsize=10, labelpad=8)
        if ax == axes[0]:
            ax.set_ylabel("Densidade de Probabilidade", fontsize=10, labelpad=8)
        ax.set_xlim(0.1, 1.0)
        ax.grid(True, zorder=0)
        ax.legend(frameon=True, facecolor="white", edgecolor="#E0E0E0", loc="upper left", fontsize=8.0)

    fig.suptitle("Distribuição das Similaridades: Pares Genuínos (Positivos) vs. Impostores (Negativos)", fontsize=13, fontweight="bold", y=1.03)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Salvo: {save_path}")
    plt.close()

# GRÁFICO 4: Dashboard Integrado (Painel Completo 2x2)
def plot_full_dashboard(save_path=None):
    fig = plt.figure(figsize=(16, 11), dpi=300)
    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.22)

    # 1. Comparativo de AUC
    ax1 = fig.add_subplot(gs[0, 0])
    species_names = ["Arara-Vermelha", "Capivara", "Onça-Pintada"]
    keys = ["arara", "capivara", "onca"]
    spec_names = ["parakeet", "guineapig", "cat"]
    auc_specialist = [compute_roc_curve(data[k][s]["label"], data[k][s]["sim"])[2] for k, s in zip(keys, spec_names)]
    auc_unified = [compute_roc_curve(data[k]["unified"]["label"], data[k]["unified"]["sim"])[2] for k in keys]
    x = np.arange(len(species_names))
    w = 0.35
    r1 = ax1.bar(x - w/2, auc_specialist, w, label="Especialista", color="#1E88E5", edgecolor="#0D47A1", zorder=3)
    r2 = ax1.bar(x + w/2, auc_unified, w, label="Generalista (Unified)", color="#7E57C2", edgecolor="#4A148C", zorder=3)
    ax1.set_ylabel("ROC-AUC (%)", fontweight="bold")
    ax1.set_title("(A) Desempenho Global de Verificação (ROC-AUC)", fontweight="bold", fontsize=11, pad=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(species_names, fontweight="bold")
    ax1.set_ylim(0, 108)
    ax1.grid(axis="y", zorder=0)
    ax1.legend(loc="lower right", fontsize=9)
    for rects in [r1, r2]:
        for r in rects:
            h = r.get_height()
            ax1.annotate(f"{h:.1f}%", xy=(r.get_x() + r.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8.5, fontweight="bold")

    # 2. Margem de Separação (Delta)
    ax2 = fig.add_subplot(gs[0, 1])
    deltas = []
    labels_delta = []
    colors_delta = ["#D32F2F", "#388E3C", "#F57C00"]
    best_pairs = [("arara", "parakeet", "Arara (Parakeet)"), ("capivara", "guineapig", "Capivara (GuineaPig)"), ("onca", "unified", "Onça (Unified)")]
    for sp, m, lbl in best_pairs:
        df = data[sp][m]
        p = df[df["label"] == 1]["sim"].mean()
        n = df[df["label"] == 0]["sim"].mean()
        deltas.append(p - n)
        labels_delta.append(lbl)
    
    bars = ax2.bar(labels_delta, deltas, color=colors_delta, edgecolor="#333", width=0.5, zorder=3)
    ax2.set_ylabel("Margem de Separação (Δ Sim)", fontweight="bold")
    ax2.set_title("(B) Margem Biométrica de Separação (Média Pos - Neg)", fontweight="bold", fontsize=11, pad=10)
    ax2.set_ylim(0, 0.40)
    ax2.grid(axis="y", zorder=0)
    for b in bars:
        h = b.get_height()
        ax2.annotate(f"+{h:.4f}", xy=(b.get_x() + b.get_width() / 2, h), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=9, fontweight="bold")

    # 3. Curvas ROC Combinadas
    ax3 = fig.add_subplot(gs[1, 0])
    for sp, m, lbl, col in [("arara", "parakeet", "Arara (93.98%)", "#D32F2F"), ("capivara", "guineapig", "Capivara (87.96%)", "#388E3C"), ("onca", "unified", "Onça (69.32%)", "#F57C00")]:
        fpr, tpr, _ = compute_roc_curve(data[sp][m]["label"], data[sp][m]["sim"])
        ax3.plot(fpr, tpr, color=col, lw=2.2, label=lbl)
    ax3.plot([0, 1], [0, 1], ":", color="#888", label="Aleatório (50%)")
    ax3.set_xlabel("Taxa de Falsos Positivos (FPR)", fontweight="bold")
    ax3.set_ylabel("Taxa de Verdadeiros Positivos (TPR)", fontweight="bold")
    ax3.set_title("(C) Curvas ROC dos Melhores Modelos", fontweight="bold", fontsize=11, pad=10)
    ax3.legend(loc="lower right", fontsize=8.5)
    ax3.grid(True, zorder=0)

    # 4. Boxplot Comparativo de Similaridade
    ax4 = fig.add_subplot(gs[1, 1])
    box_data = []
    box_labels = []
    colors_box = []
    for sp, m, short in [("arara", "parakeet", "Arara"), ("capivara", "guineapig", "Capivara"), ("onca", "unified", "Onça")]:
        df = data[sp][m]
        box_data.append(df[df["label"] == 1]["sim"])
        box_labels.append(f"{short}\n(Mesmo ID)")
        colors_box.append("#C8E6C9") # Verde claro
        box_data.append(df[df["label"] == 0]["sim"])
        box_labels.append(f"{short}\n(IDs Dif)")
        colors_box.append("#FFCDD2") # Vermelho claro

    bp = ax4.boxplot(box_data, tick_labels=box_labels, patch_artist=True, medianprops=dict(color="#111", lw=1.5))
    for patch, c in zip(bp["boxes"], colors_box):
        patch.set_facecolor(c)
        patch.set_edgecolor("#555")

    ax4.set_ylabel("Similaridade de Cosseno", fontweight="bold")
    ax4.set_title("(D) Dispersão de Similaridades: Positivos (Verde) vs. Negativos (Vermelho)", fontweight="bold", fontsize=11, pad=10)
    ax4.grid(axis="y", zorder=0)

    fig.suptitle("Relatório Gráfico Integrado - Extensão FaunaBR (PetFace ECCV 2024)", fontsize=15, fontweight="bold", y=0.98)
    if save_path:
        plt.savefig(save_path)
        print(f"Salvo: {save_path}")
    plt.close()

def main():
    print("=" * 60)
    print("Gerando gráficos científicos com Matplotlib...")
    print("=" * 60)
    plot_auc_comparison(os.path.join(PLOTS_DIR, "grafico1_comparativo_auc.png"))
    plot_roc_curves(os.path.join(PLOTS_DIR, "grafico2_curvas_roc.png"))
    plot_similarity_distributions(os.path.join(PLOTS_DIR, "grafico3_distribuicao_similaridade.png"))
    plot_full_dashboard(os.path.join(PLOTS_DIR, "painel_completo_faunabr.png"))
    print("=" * 60)
    print(f"Todos os gráficos foram salvos com sucesso em: {PLOTS_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    main()

