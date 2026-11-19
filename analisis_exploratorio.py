# ==============================================================
# Taller 2 IA - Fase 2: Análisis exploratorio de datos (EDA)
# Persona A
# Lee train_dataset.csv, imprime los hallazgos y guarda
# figuras (figuras/) y tablas (resultados/)
# ==============================================================

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import chi2_contingency, mannwhitneyu

# --------------------------------------------------------------
# 0. Rutas, carga y estilo
# --------------------------------------------------------------
BASE = Path(__file__).resolve().parent
DATOS = BASE / "datos"
FIGURAS = BASE / "figuras"
RESULTADOS = BASE / "resultados"
FIGURAS.mkdir(exist_ok=True)
RESULTADOS.mkdir(exist_ok=True)

train = pd.read_csv(DATOS / "train_dataset.csv")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

# Variable objetivo como 0/1 (solo para el análisis; el mapeo
# definitivo con if/else se hace en la Fase 3)
y = train["defaulted_loan"].astype(int)
TASA_GLOBAL = y.mean()

AZUL = "#2a78d6"      # clase 0: pagó
NARANJA = "#eb6834"   # clase 1: no pagó (moroso)
GRIS = "#52514e"
GRILLA = "#e5e4e0"
plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": GRIS, "xtick.color": GRIS,
    "ytick.color": GRIS, "axes.titleweight": "bold",
})

def limpiar_ejes(ax, eje_grilla="y"):
    ax.grid(axis=eje_grilla, color=GRILLA, linewidth=0.6)
    ax.set_axisbelow(True)
    for borde in ["top", "right"]:
        ax.spines[borde].set_visible(False)

numericas = {
    "age": "Edad",
    "salary": "Salario",
    "credit_score": "Puntaje crediticio",
    "years_in_current_job": "Años en el trabajo actual",
    "years_of_education": "Años de educación",
    "number_of_children": "Número de hijos",
}
categoricas = {
    "has_criminal_records": "Antecedentes penales",
    "is_married": "Casado(a)",
    "smoker": "Fumador(a)",
    "email_domain": "Dominio de correo",
    "credit_card_provider": "Franquicia de la tarjeta",
    "number_of_credit_cards": "Número de tarjetas",
}

# --------------------------------------------------------------
# 2.1 Distribución de la variable objetivo (desbalance)
# --------------------------------------------------------------
print("=" * 60)
print("PASO 2.1 - DISTRIBUCIÓN DE LA VARIABLE OBJETIVO")
print("=" * 60)
conteo = y.value_counts().sort_index()
print(f"Pagó (0):    {conteo[0]:>6}  ({conteo[0] / len(y) * 100:.2f} %)")
print(f"No pagó (1): {conteo[1]:>6}  ({conteo[1] / len(y) * 100:.2f} %)")
print(f"Razón de desbalance: {conteo[0] / conteo[1]:.2f} : 1")

fig, ax = plt.subplots(figsize=(3.6, 2.6))
barras = ax.bar(["Pagó (0)", "No pagó (1)"], conteo.values,
                color=[AZUL, NARANJA], width=0.55)
for barra, valor in zip(barras, conteo.values):
    ax.text(barra.get_x() + barra.get_width() / 2, valor + 400,
            f"{valor:,} ({valor / len(y) * 100:.1f} %)".replace(",", "."),
            ha="center", fontsize=8)
ax.set_ylabel("Clientes")
ax.set_ylim(0, conteo.max() * 1.15)
limpiar_ejes(ax)
fig.tight_layout()
fig.savefig(FIGURAS / "fig_fase2_clases.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# --------------------------------------------------------------
# 2.2 Variables numéricas: comparación entre clases
# --------------------------------------------------------------
print("\n" + "=" * 60)
print("PASO 2.2 - VARIABLES NUMÉRICAS POR CLASE")
print("=" * 60)
filas = []
for col, nombre in numericas.items():
    pago = train.loc[y == 0, col]
    moroso = train.loc[y == 1, col]
    p = mannwhitneyu(pago, moroso).pvalue
    filas.append({
        "variable": col,
        "media_pago": round(pago.mean(), 2),
        "media_moroso": round(moroso.mean(), 2),
        "mediana_pago": pago.median(),
        "mediana_moroso": moroso.median(),
        "p_mann_whitney": p,
        "corr_con_impago": round(np.corrcoef(train[col], y)[0, 1], 3),
    })
tabla_num = pd.DataFrame(filas).set_index("variable")
print(tabla_num)

# Figura: histogramas superpuestos por clase
fig, ejes = plt.subplots(2, 3, figsize=(7.2, 4.6))
for ax, (col, nombre) in zip(ejes.ravel(), numericas.items()):
    discreta = train[col].nunique() <= 10
    if discreta:
        bins = np.arange(train[col].min() - 0.5, train[col].max() + 1.5, 1)
    else:
        bins = 30
    ax.hist(train.loc[y == 0, col], bins=bins, density=True, color=AZUL,
            alpha=0.55, label="Pagó")
    ax.hist(train.loc[y == 1, col], bins=bins, density=True, color=NARANJA,
            alpha=0.55, label="No pagó")
    ax.set_title(nombre, fontsize=9)
    ax.set_yticks([])
    limpiar_ejes(ax, "x")
    ax.spines["left"].set_visible(False)
ejes[0, 0].legend(fontsize=7, frameon=False, loc="upper left")
fig.tight_layout()
fig.savefig(FIGURAS / "fig_fase2_histogramas.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# --------------------------------------------------------------
# 2.3 Variables categóricas y binarias: tasa de impago
# --------------------------------------------------------------
print("\n" + "=" * 60)
print("PASO 2.3 - TASA DE IMPAGO POR CATEGORÍA")
print("=" * 60)
filas = []
for col in list(categoricas) + ["number_of_children"]:
    tabla = pd.crosstab(train[col], y)
    chi2, p, _, _ = chi2_contingency(tabla)
    v_cramer = np.sqrt(chi2 / (tabla.values.sum() * (min(tabla.shape) - 1)))
    filas.append({"variable": col, "p_chi2": p, "v_cramer": round(v_cramer, 3)})
    print(f"\n{col}  (chi2 p = {p:.2e}, V de Cramér = {v_cramer:.3f})")
    print(train.groupby(col)["defaulted_loan"]
          .agg(tasa_impago="mean", clientes="count").round(3))
tabla_cat = pd.DataFrame(filas).set_index("variable")

# Variables de texto con muchas categorías o fechas
fecha = pd.to_datetime(train["application_date"])
tasa_anio = train.groupby(fecha.dt.year)["defaulted_loan"].mean()
print("\nTasa de impago por año de solicitud:")
print(tasa_anio.round(3))
tasas_empleo = train.groupby("current_job")["defaulted_loan"].agg(["mean", "count"])
ruido_esperado = np.sqrt(TASA_GLOBAL * (1 - TASA_GLOBAL) / tasas_empleo["count"].mean())
print(f"\ncurrent_job: {len(tasas_empleo)} ocupaciones, "
      f"desv. de la tasa entre ocupaciones = {tasas_empleo['mean'].std():.3f} "
      f"(ruido esperado por azar = {ruido_esperado:.3f})")

# Figura: tasa de impago por categoría
fig, ejes = plt.subplots(2, 3, figsize=(7.2, 4.6))
for ax, col in zip(ejes.ravel(), ["number_of_children", "has_criminal_records",
                                  "is_married", "smoker", "email_domain",
                                  "number_of_credit_cards"]):
    tasas = train.groupby(col)["defaulted_loan"].mean() * 100
    etiquetas = [str(e).replace(".com", "") for e in tasas.index]
    ax.bar(etiquetas, tasas.values, color=AZUL, width=0.6)
    ax.axhline(TASA_GLOBAL * 100, color=GRIS, linestyle="--", linewidth=1)
    nombre = categoricas.get(col, "Número de hijos")
    ax.set_title(nombre, fontsize=9)
    ax.set_ylim(0, 105)
    ax.tick_params(axis="x", labelsize=7.5)
    limpiar_ejes(ax)
for ax in ejes[:, 0]:
    ax.set_ylabel("Tasa de impago (%)")
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.text(0.5, 0.985, f"Línea punteada: tasa promedio de impago ({TASA_GLOBAL * 100:.1f} %)",
         ha="center", va="top", fontsize=8, color=GRIS)
fig.savefig(FIGURAS / "fig_fase2_tasas_categoricas.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# --------------------------------------------------------------
# 2.4 Matriz de correlación
# --------------------------------------------------------------
print("\n" + "=" * 60)
print("PASO 2.4 - MATRIZ DE CORRELACIÓN")
print("=" * 60)
cols_corr = list(numericas) + ["number_of_credit_cards", "has_criminal_records"]
matriz = train[cols_corr].astype(float).assign(impago=y).corr()
print(matriz.round(2))

etiquetas_corr = [numericas.get(c, categoricas.get(c, c)) for c in cols_corr] + ["Impago"]
mapa = LinearSegmentedColormap.from_list(
    "azul_rojo", ["#104281", "#2a78d6", "#f0efec", "#e34948", "#a32f2f"])
fig, ax = plt.subplots(figsize=(6.2, 5.2))
im = ax.imshow(matriz.values, cmap=mapa, vmin=-1, vmax=1)
ax.set_xticks(range(len(etiquetas_corr)))
ax.set_yticks(range(len(etiquetas_corr)))
ax.set_xticklabels(etiquetas_corr, rotation=45, ha="right", fontsize=7.5)
ax.set_yticklabels(etiquetas_corr, fontsize=7.5)
for i in range(len(matriz)):
    for j in range(len(matriz)):
        valor = matriz.values[i, j]
        ax.text(j, i, f"{valor:.2f}", ha="center", va="center", fontsize=6.5,
                color="white" if abs(valor) > 0.6 else "#0b0b0b")
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Correlación de Pearson")
fig.tight_layout()
fig.savefig(FIGURAS / "fig_fase2_correlacion.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# --------------------------------------------------------------
# 2.5 Valor de información (IV): ranking de variables
# --------------------------------------------------------------
print("\n" + "=" * 60)
print("PASO 2.5 - VALOR DE INFORMACIÓN (IV)")
print("=" * 60)

def valor_informacion(x, objetivo):
    """IV = sum((%buenos - %malos) * ln(%buenos / %malos)) por grupo."""
    tabla = pd.DataFrame({"x": x, "y": objetivo}).groupby("x", observed=True)["y"]
    malos = tabla.sum()
    buenos = tabla.count() - malos
    p_malos = (malos + 0.5) / (malos.sum() + 0.5)     # +0.5 evita divisiones por 0
    p_buenos = (buenos + 0.5) / (buenos.sum() + 0.5)
    return ((p_buenos - p_malos) * np.log(p_buenos / p_malos)).sum()

iv = {}
for col in numericas:
    if train[col].nunique() > 10:
        grupos = pd.qcut(train[col], 10, duplicates="drop")   # deciles
    else:
        grupos = train[col]
    iv[col] = valor_informacion(grupos, y)
for col in categoricas:
    iv[col] = valor_informacion(train[col], y)
iv["application_year"] = valor_informacion(fecha.dt.year, y)

def fuerza(valor):
    if valor < 0.02:
        return "Sin poder predictivo"
    if valor < 0.1:
        return "Débil"
    if valor < 0.3:
        return "Medio"
    return "Fuerte"

tabla_iv = (pd.Series(iv, name="IV").sort_values(ascending=False).round(4)
            .to_frame().assign(fuerza=lambda d: d["IV"].map(fuerza)))
print(tabla_iv)

# Tabla resumen para el informe
resumen = tabla_iv.join(tabla_num[["p_mann_whitney", "corr_con_impago"]]) \
                  .join(tabla_cat)
resumen.to_csv(RESULTADOS / "eda_relacion_impago.csv")

# Figura: ranking IV
nombres_iv = {**numericas, **categoricas, "application_year": "Año de solicitud"}
orden = tabla_iv.sort_values("IV")
fig, ax = plt.subplots(figsize=(5.6, 3.6))
ax.barh([nombres_iv[c] for c in orden.index], orden["IV"],
        color=[AZUL if v >= 0.02 else "#9ec5f4" for v in orden["IV"]], height=0.6)
# Umbrales usuales del IV: 0.02 (débil), 0.1 (medio), 0.3 (fuerte)
for umbral in [0.02, 0.1, 0.3]:
    ax.axvline(umbral, color=GRIS, linestyle="--", linewidth=0.8)
for x, texto in [(0.06, "Débil"), (0.2, "Medio"), (0.55, "Fuerte")]:
    ax.text(x, len(orden) - 0.3, texto, fontsize=7, color=GRIS, ha="center")
ax.set_ylim(-0.6, len(orden) + 0.2)
for i, v in enumerate(orden["IV"]):
    ax.text(max(v, 0.02) + 0.01, i, f"{v:.3f}", va="center", fontsize=7)
ax.set_xlabel("Valor de información (IV)")
ax.tick_params(axis="y", labelsize=7.5)
limpiar_ejes(ax, "x")
fig.tight_layout()
fig.savefig(FIGURAS / "fig_fase2_iv.png", dpi=300, bbox_inches="tight")
plt.close(fig)

print("\nFiguras guardadas en:", FIGURAS)
print("Tabla resumen guardada en:", RESULTADOS / "eda_relacion_impago.csv")