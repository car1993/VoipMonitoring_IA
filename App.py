import os
import datetime
import pandas as pd
import joblib
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter.scrolledtext import ScrolledText
from sklearn.ensemble import IsolationForest

# ═════════════════════════════════════════
# CONFIGURACIÓN DE SENSIBILIDAD
# ═════════════════════════════════════════
UMBRAL_ASR = 70
UMBRAL_ABR = 70
UMBRAL_ACD = 200

PISO_ASR = 20    # Promedio mínimo para reportar VARIACIÓN % de ASR
PISO_ABR = 5     # Promedio mínimo para reportar VARIACIÓN % de ABR (caída a 0 siempre se reporta)
PISO_ACD = 0.3   # Promedio mínimo para reportar VARIACIÓN % de ACD

UMBRAL_INTENTOS_X    = 10
UMBRAL_INTENTOS_CERO = 500

CARPETA_MODELOS = "modelos_voip"
os.makedirs(CARPETA_MODELOS, exist_ok=True)

# ═════════════════════════════════════════
# HELPERS
# ═════════════════════════════════════════
def nombre_modelo(area):
    return os.path.join(CARPETA_MODELOS, area.replace(" ", "_").replace(":", "") + ".pkl")

def parsear_fechas(serie):
    anio_actual = datetime.datetime.now().year
    def convertir(valor):
        try:
            dt = pd.to_datetime(str(valor), format="%b %d, %H:%M", errors="coerce")
            if pd.isna(dt):
                return pd.NaT
            return dt.replace(year=anio_actual)
        except:
            return pd.NaT
    return serie.apply(convertir)

def limpiar_numero(serie):
    """
    Convierte una columna a número.
    Celda vacía, NaN, guión o cualquier texto no numérico → 0
    """
    return (
        serie
        .astype(str)
        .str.strip()
        .str.replace(",", ".")
        .str.replace("-", "0")
        .pipe(pd.to_numeric, errors="coerce")
        .fillna(0)   # NaN (celda vacía en Excel) → 0
    )

def evaluar_intentos(actual, promedio, dias):
    if actual == 0 and promedio >= UMBRAL_INTENTOS_CERO:
        return f"0 intentos registrados (promedio historico: {promedio:,.0f} en {dias} dias)"
    if promedio == 0 or actual == 0:
        return None
    ratio = max(actual, promedio) / min(actual, promedio)
    if ratio >= UMBRAL_INTENTOS_X:
        diferencia = abs(actual - promedio)
        direccion  = "disminucion" if actual < promedio else "aumento"
        return f"{direccion} de {diferencia:,.0f} intentos vs promedio {promedio:,.0f} ({dias} dias)"
    return None

def evaluar_metrica(col, actual, historico):
    umbrales = {"ACD": UMBRAL_ACD, "ASR": UMBRAL_ASR, "ABR": UMBRAL_ABR}
    pisos    = {"ACD": PISO_ACD,   "ASR": PISO_ASR,   "ABR": PISO_ABR}
    promedio = historico[col].mean()
    dias     = len(historico)

    # Caída a 0: siempre reportar si el histórico tenía algún valor real
    # El piso NO aplica aquí — un ABR que cae a 0 siempre es crítico
    if actual == 0 and promedio > 0:
        return f"{col} en 0 (promedio historico: {promedio:.2f} en {dias} dias)"

    if promedio == 0 or actual == 0:
        return None

    # Piso mínimo: evita falsas alarmas en variaciones % de valores pequeños
    if promedio < pisos[col]:
        return None

    variacion = abs((actual - promedio) / promedio) * 100
    if variacion >= umbrales[col]:
        direccion = "disminucion" if actual < promedio else "aumento"
        return f"{direccion} de {col}: {actual:.2f} vs promedio {promedio:.2f} ({variacion:.0f}%, {dias} dias)"
    return None

# ═════════════════════════════════════════
# FUNCIÓN PRINCIPAL
# ═════════════════════════════════════════
def leer_archivo_excel():

    ruta_archivo = filedialog.askopenfilename(
        title="Seleccionar reporte Excel",
        filetypes=[("Excel files", "*.xlsx *.xls")]
    )
    if not ruta_archivo:
        return

    try:
        # ── LEER ────────────────────────────
        df = pd.read_excel(ruta_archivo, header=1, keep_default_na=False)
        df.columns = ["area", "fecha", "intentos", "ACD", "ASR", "ABR"]

        # Eliminar filas donde area o fecha están vacías
        df = df[df["area"].astype(str).str.strip() != ""]
        df = df[df["fecha"].astype(str).str.strip() != ""]

        # Convertir métricas: vacío/NaN/guión → 0
        for col in ["intentos", "ACD", "ASR", "ABR"]:
            df[col] = limpiar_numero(df[col])

        df["fecha"] = parsear_fechas(df["fecha"].astype(str))
        df = df[df["fecha"].notna()]

        # ── DETECTAR DÍA MÁS RECIENTE ───────
        fecha_hoy = df["fecha"].max().normalize()
        df_hoy  = df[df["fecha"].dt.normalize() == fecha_hoy]
        df_hist = df[df["fecha"].dt.normalize() <  fecha_hoy]

        # ── INICIO REPORTE ───────────────────
        cuadro_resultados.delete(1.0, tk.END)
        cuadro_resultados.insert(tk.END, "== ANALISIS DE DATOS VOIP ==\n\n")
        cuadro_resultados.insert(tk.END, f"Fecha analizada : {fecha_hoy.strftime('%b %d, %Y')}\n")
        cuadro_resultados.insert(tk.END, f"Dias historicos : {df_hist['fecha'].dt.normalize().nunique()}\n\n")

        columnas_ia = ["intentos", "ACD", "ASR", "ABR"]
        alertas_encontradas = False

        for area in df_hoy["area"].unique():

            registros_hoy = df_hoy[df_hoy["area"] == area]
            if registros_hoy.empty:
                continue
            ultimo = registros_hoy.iloc[0]

            historico = df_hist[df_hist["area"] == area].sort_values("fecha")
            if len(historico) < 2:
                continue

            # ── IA: cargar o crear modelo ────
            ruta_modelo = nombre_modelo(area)
            if os.path.exists(ruta_modelo):
                modelo_ia = joblib.load(ruta_modelo)
            else:
                modelo_ia = IsolationForest(contamination=0.1, random_state=42)

            modelo_ia.fit(historico[columnas_ia])
            joblib.dump(modelo_ia, ruta_modelo)

            prediccion  = modelo_ia.predict(ultimo[columnas_ia].values.reshape(1, -1))
            es_anomalia = prediccion[0] == -1

            # ── EVALUAR MOTIVOS ──────────────
            motivos = []

            msg = evaluar_intentos(ultimo["intentos"], historico["intentos"].mean(), len(historico))
            if msg:
                motivos.append(f"Se observa {msg}")

            for col in ["ACD", "ASR", "ABR"]:
                msg = evaluar_metrica(col, ultimo[col], historico)
                if msg:
                    motivos.append(f"Se observa {msg}")

            if not es_anomalia and not motivos:
                continue

            alertas_encontradas = True
            estado = "CRITICO" if motivos else "REVISAR (IA detecto patron inusual)"

            cuadro_resultados.insert(tk.END, f"{'─'*55}\n")
            cuadro_resultados.insert(tk.END, f"AREA   : {area}\n")
            cuadro_resultados.insert(tk.END, f"FECHA  : {ultimo['fecha'].strftime('%b %d %H:%M')}\n")
            cuadro_resultados.insert(tk.END, f"ESTADO : {estado}\n")

            if motivos:
                cuadro_resultados.insert(tk.END, "MOTIVOS:\n")
                for m in motivos:
                    cuadro_resultados.insert(tk.END, f"  ! {m}\n")

            cuadro_resultados.insert(tk.END, "\n")

        if not alertas_encontradas:
            cuadro_resultados.insert(tk.END, "Sin anomalias. Todas las areas en comportamiento normal.\n")

        cuadro_resultados.insert(tk.END, "\nAnalisis completado correctamente.")

    except Exception as e:
        messagebox.showerror("Error", f"Ocurrio un error:\n\n{e}")


# ═════════════════════════════════════════
# INTERFAZ GRÁFICA
# ═════════════════════════════════════════
ventana = tk.Tk()
ventana.title("Analisis de Datos VoIP")
ventana.geometry("1000x700")
ventana.configure(bg="#1e1e1e")

tk.Label(ventana, text="Analisis de Datos VoIP", font=("Arial", 24, "bold"),
         bg="#1e1e1e", fg="#00ff00").pack(pady=20)

tk.Button(ventana, text="Cargar Archivo Excel", font=("Arial", 14),
          bg="#00ff00", fg="#1e1e1e", command=leer_archivo_excel).pack(pady=10)

cuadro_resultados = ScrolledText(ventana, width=120, height=30,
                                  font=("Courier New", 12), bg="#1e1e1e",
                                  fg="#00ff00", insertbackground="white")
cuadro_resultados.pack(padx=20, pady=20)

tk.Label(ventana, text="PROYECTO: Actividad 4. Creacion de un modelo de inteligencia artificial avanzada",
         font=("Arial", 10), bg="#1e1e1e", fg="#00ff00").pack()

ventana.mainloop()