# ==============================================================
# Taller 2 IA - Gráficos de la Fase 1 (revisión de datos)
# Lee train_dataset.csv y guarda las figuras en la carpeta figuras/
# ==============================================================

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------------------------------------
# 1. Rutas y carga de datos
# --------------------------------------------------------------
BASE = Path(__file__).resolve().parent
DATOS = BASE / "datos"
FIGURAS = BASE / "figuras"
FIGURAS.mkdir(exist_ok=True)

train = pd.read_csv(DATOS / "train_dataset.csv")

# --------------------------------------------------------------
# 2. Estilo general de las gráficas
# --------------------------------------------------------------
AZUL = "#2a78d6"        # color principal
AZUL_CLARO = "#cde2fb"  # relleno de las cajas
NARANJA = "#eb6834"     # valores imposibles (-1)
GRIS = "#52514e"        # ejes y textos secundarios
GRILLA = "#e5e4e0"

plt.rcParams.update({
    "font.size": 9,
    "axes.edgecolor": GRIS,
    "xtick.color": GRIS,
    "ytick.color": GRIS,
    "axes.titleweight": "bold",
})

# Variables numéricas y su nombre en español para los títulos
variables = {
    "age": "Edad (años)",
    "salary": "Salario",
    "credit_score": "Puntaje crediticio",
    "years_in_current_job": "Años en el trabajo actual",
    "years_of_education": "Años de educación",
    "number_of_children": "Número de hijos",
}

# --------------------------------------------------------------
# 3. Porcentaje de atípicos con el criterio IQR
# --------------------------------------------------------------
def porcentaje_atipicos(serie):
    q1 = serie.quantile(0.25)
    q3 = serie.quantile(0.75)
    iqr = q3 - q1
    lim_inf = q1 - 1.5 * iqr
    lim_sup = q3 + 1.5 * iqr
    atipicos = (serie < lim_inf) | (serie > lim_sup)
    return atipicos.mean() * 100

# --------------------------------------------------------------
# 4. Figura 1: diagramas de caja de las variables numéricas
# --------------------------------------------------------------
fig, ejes = plt.subplots(2, 3, figsize=(7.2, 4.6))

for ax, col in zip(ejes.ravel(), variables):
    datos = train[col]

    # Diagrama de caja horizontal
    ax.boxplot(
        datos, vert=False, widths=0.5, patch_artist=True,
        boxprops=dict(facecolor=AZUL_CLARO, edgecolor=AZUL),
        medianprops=dict(color=AZUL, linewidth=2),
        whiskerprops=dict(color=AZUL),
        capprops=dict(color=AZUL),
        flierprops=dict(marker="o", markersize=3, markerfacecolor=AZUL,
                        markeredgecolor="none", alpha=0.35),
    )

    # Marcar en naranja los valores imposibles (negativos)
    negativos = datos[datos < 0]
    if len(negativos) > 0:
        ax.scatter(negativos, [1] * len(negativos), s=36, color=NARANJA,
                   edgecolor="white", zorder=3, label="Valor imposible (-1)")
        ax.legend(loc="upper right", fontsize=7, frameon=False)

    # Título y porcentaje de atípicos
    ax.set_title(variables[col], fontsize=9)
    ax.text(0.02, 0.06, f"Atípicos IQR: {porcentaje_atipicos(datos):.2f} %",
            transform=ax.transAxes, fontsize=7.5, color=GRIS)

    # Limpieza visual
    ax.set_yticks([])
    ax.grid(axis="x", color=GRILLA, linewidth=0.6)
    ax.set_axisbelow(True)
    for borde in ["top", "right", "left"]:
        ax.spines[borde].set_visible(False)

fig.tight_layout()
fig.savefig(FIGURAS / "fig_fase1_atipicos.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# --------------------------------------------------------------
# 5. Figura 2: distribución del número de tarjetas de crédito
# --------------------------------------------------------------
conteo = train["number_of_credit_cards"].value_counts().sort_index()

fig, ax = plt.subplots(figsize=(4.2, 2.6))
barras = ax.bar(conteo.index, conteo.values, width=0.6, color=AZUL)

# Porcentaje encima de cada barra
for barra, valor in zip(barras, conteo.values):
    porcentaje = valor / len(train) * 100
    ax.text(barra.get_x() + barra.get_width() / 2, valor + 300,
            f"{porcentaje:.1f} %", ha="center", fontsize=7.5)

ax.set_xlabel("Número de tarjetas de crédito")
ax.set_ylabel("Clientes")
ax.set_ylim(0, conteo.max() * 1.15)
ax.grid(axis="y", color=GRILLA, linewidth=0.6)
ax.set_axisbelow(True)
for borde in ["top", "right"]:
    ax.spines[borde].set_visible(False)

fig.tight_layout()
fig.savefig(FIGURAS / "fig_fase1_tarjetas.png", dpi=300, bbox_inches="tight")
plt.close(fig)

print("Figuras guardadas en:", FIGURAS)