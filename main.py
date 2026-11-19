# ==============================================================
# Taller 2 IA - Fase 1: Inspección inicial de los datos
# Persona A
# ==============================================================

from pathlib import Path
import pandas as pd
import numpy as np

# --------------------------------------------------------------
# Paso 1.1 - Rutas del proyecto (relativas a la carpeta de main.py)
# --------------------------------------------------------------
BASE = Path(__file__).resolve().parent
DATOS = BASE / "datos"
RESULTADOS = BASE / "resultados"
FIGURAS = BASE / "figuras"
RESULTADOS.mkdir(exist_ok=True)
FIGURAS.mkdir(exist_ok=True)

# Mostrar todas las columnas al imprimir tablas
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

# --------------------------------------------------------------
# Paso 1.2 - Cargar datos y verificar dimensiones
# Esperado: train (36000, 23) y test (24000, 22)
# --------------------------------------------------------------
train = pd.read_csv(DATOS / "train_dataset.csv")
test = pd.read_csv(DATOS / "test_dataset.csv")

print("=" * 60)
print("PASO 1.2 - DIMENSIONES")
print("=" * 60)
print("Train:", train.shape)
print("Test: ", test.shape)
print("\nPrimeras filas de train:")
print(train.head())

# --------------------------------------------------------------
# Paso 1.3 - Diccionario de datos
# --------------------------------------------------------------
def diccionario_datos(df):
    return pd.DataFrame({
        "tipo": df.dtypes.astype(str),
        "valores_unicos": df.nunique(),
        "nulos": df.isna().sum(),
        "pct_nulos": (df.isna().mean() * 100).round(2),
        "ejemplo": df.iloc[0],
    })

dicc_train = diccionario_datos(train)
dicc_train.to_csv(RESULTADOS / "diccionario_datos.csv")

print("\n" + "=" * 60)
print("PASO 1.3 - DICCIONARIO DE DATOS (train)")
print("=" * 60)
print(dicc_train)

# --------------------------------------------------------------
# Paso 1.4 - Clasificar columnas
# --------------------------------------------------------------
objetivo = "defaulted_loan"
id_col = "application_id"

predictoras = train.drop(columns=[objetivo, id_col])
num_cols = predictoras.select_dtypes(include="number").columns.tolist()
cat_cols = predictoras.select_dtypes(exclude="number").columns.tolist()

print("\n" + "=" * 60)
print("PASO 1.4 - CLASIFICACIÓN DE COLUMNAS")
print("=" * 60)
print(f"Numéricas ({len(num_cols)}):", num_cols)
print(f"Categóricas ({len(cat_cols)}):", cat_cols)

# --------------------------------------------------------------
# Paso 1.5 - Duplicados e IDs
# --------------------------------------------------------------
print("\n" + "=" * 60)
print("PASO 1.5 - DUPLICADOS E IDs")
print("=" * 60)
print("Filas duplicadas (train):", train.duplicated().sum())
print("Duplicadas ignorando el ID:", train.drop(columns=id_col).duplicated().sum())
print("IDs repetidos train:", train[id_col].duplicated().sum())
print("IDs repetidos test: ", test[id_col].duplicated().sum())
print("IDs presentes en ambos:", len(set(train[id_col]) & set(test[id_col])))

# --------------------------------------------------------------
# Paso 1.6 - Coherencia entre train y test
# --------------------------------------------------------------
print("\n" + "=" * 60)
print("PASO 1.6 - COHERENCIA TRAIN vs TEST")
print("=" * 60)
print("Columnas solo en train:", set(train.columns) - set(test.columns))
print("Columnas solo en test: ", set(test.columns) - set(train.columns))

print("\nCategorías que aparecen en test pero no en train:")
hay_nuevas = False
for c in cat_cols:
    nuevas = set(test[c].dropna().unique()) - set(train[c].dropna().unique())
    if nuevas:
        hay_nuevas = True
        ejemplos = list(nuevas)[:3]
        print(f"  {c}: {len(nuevas)} valores nuevos (ej.: {ejemplos})")
if not hay_nuevas:
    print("  Ninguna")

print("\nNulos en test:")
nulos_test = test.isna().sum()
print(nulos_test[nulos_test > 0] if (nulos_test > 0).any() else "  Ninguno")

# --------------------------------------------------------------
# Paso 1.7 - Variable objetivo
# --------------------------------------------------------------
print("\n" + "=" * 60)
print("PASO 1.7 - VARIABLE OBJETIVO")
print("=" * 60)
print("Tipo de dato:", train[objetivo].dtype)
print("\nConteo:")
print(train[objetivo].value_counts(dropna=False))
print("\nProporción:")
print(train[objetivo].value_counts(normalize=True).round(4))

# --------------------------------------------------------------
# Paso 1.8 - Rangos, atípicos (IQR) y coherencia lógica
# --------------------------------------------------------------
print("\n" + "=" * 60)
print("PASO 1.8 - RANGOS Y ATÍPICOS")
print("=" * 60)
cont_cols = ["age", "salary", "credit_score", "years_in_current_job",
             "years_of_education", "number_of_children", "number_of_credit_cards"]

resumen = train[cont_cols].describe().T.round(2)
q1 = train[cont_cols].quantile(0.25)
q3 = train[cont_cols].quantile(0.75)
iqr = q3 - q1
lim_inf = q1 - 1.5 * iqr
lim_sup = q3 + 1.5 * iqr
resumen["lim_inf_IQR"] = lim_inf.round(2)
resumen["lim_sup_IQR"] = lim_sup.round(2)
resumen["n_atipicos"] = ((train[cont_cols] < lim_inf) | (train[cont_cols] > lim_sup)).sum()
resumen["pct_atipicos"] = (resumen["n_atipicos"] / len(train) * 100).round(2)
resumen.to_csv(RESULTADOS / "rangos_atipicos.csv")
print(resumen)

print("\nCoherencia lógica:")
print("  Edad < 18:", (train["age"] < 18).sum())
print("  Años en trabajo actual > edad - 14:", (train["years_in_current_job"] > train["age"] - 14).sum())
print("  Años de educación > edad:", (train["years_of_education"] > train["age"]).sum())
print("  Salario <= 0:", (train["salary"] <= 0).sum())
print("  Valores negativos por columna:")
print((train[cont_cols] < 0).sum()[lambda x: x > 0])

print("\nRangos en test (para comparar con train):")
print(test[cont_cols].describe().T[["min", "max", "mean"]].round(2))