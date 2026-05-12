import os
import io
import datetime
import pandas as pd
import joblib
import secrets
from flask import (
    Flask, render_template, request, redirect, url_for,
    flash, session, jsonify
)
from authlib.integrations.flask_client import OAuth
from werkzeug.utils import secure_filename
from sklearn.ensemble import IsolationForest

# ═════════════════════════════════════════
# CONFIGURACIÓN
# ═════════════════════════════════════════
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", secrets.token_hex(32))

# OAuth — reemplazar con tus credenciales reales
GOOGLE_CLIENT_ID     = os.getenv("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "")
GITHUB_CLIENT_ID     = os.getenv("GITHUB_CLIENT_ID", "")
GITHUB_CLIENT_SECRET = os.getenv("GITHUB_CLIENT_SECRET", "")

oauth = OAuth(app)

oauth.register(
    name="google",
    client_id=GOOGLE_CLIENT_ID,
    client_secret=GOOGLE_CLIENT_SECRET,
    server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
    client_kwargs={"scope": "openid email profile"},
)

oauth.register(
    name="github",
    client_id=GITHUB_CLIENT_ID,
    client_secret=GITHUB_CLIENT_SECRET,
    access_token_url="https://github.com/login/oauth/access_token",
    authorize_url="https://github.com/login/oauth/authorize",
    api_base_url="https://api.github.com/",
    client_kwargs={"scope": "user:email"},
)

# Configuración de subida
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
ALLOWED_EXTENSIONS = {".xlsx", ".xls"}
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ═════════════════════════════════════════
# AJUSTES DE ANÁLISIS (desde App.py)
# ═════════════════════════════════════════
UMBRAL_ASR = 70
UMBRAL_ABR = 70
UMBRAL_ACD = 200

PISO_ASR = 20
PISO_ABR = 5
PISO_ACD = 0.3

UMBRAL_INTENTOS_X = 10
UMBRAL_INTENTOS_CERO = 500

CARPETA_MODELOS = os.path.join(os.path.dirname(__file__), "modelos_voip")
os.makedirs(CARPETA_MODELOS, exist_ok=True)


# ═════════════════════════════════════════
# HELPERS (copiados de App.py)
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
    return (
        serie.astype(str).str.strip()
        .str.replace(",", ".")
        .str.replace("-", "0")
        .pipe(pd.to_numeric, errors="coerce")
        .fillna(0)
    )


def evaluar_intentos(actual, promedio, dias):
    if actual == 0 and promedio >= UMBRAL_INTENTOS_CERO:
        return f"0 intentos registrados (promedio historico: {promedio:,.0f} en {dias} dias)"
    if promedio == 0 or actual == 0:
        return None
    ratio = max(actual, promedio) / min(actual, promedio)
    if ratio >= UMBRAL_INTENTOS_X:
        diferencia = abs(actual - promedio)
        direccion = "disminucion" if actual < promedio else "aumento"
        return f"{direccion} de {diferencia:,.0f} intentos vs promedio {promedio:,.0f} ({dias} dias)"
    return None


def evaluar_metrica(col, actual, historico):
    umbrales = {"ACD": UMBRAL_ACD, "ASR": UMBRAL_ASR, "ABR": UMBRAL_ABR}
    pisos = {"ACD": PISO_ACD, "ASR": PISO_ASR, "ABR": PISO_ABR}
    promedio = historico[col].mean()
    dias = len(historico)

    if actual == 0 and promedio > 0:
        return f"{col} en 0 (promedio historico: {promedio:.2f} en {dias} dias)"
    if promedio == 0 or actual == 0:
        return None
    if promedio < pisos[col]:
        return None

    variacion = abs((actual - promedio) / promedio) * 100
    if variacion >= umbrales[col]:
        direccion = "disminucion" if actual < promedio else "aumento"
        return f"{direccion} de {col}: {actual:.2f} vs promedio {promedio:.2f} ({variacion:.0f}%, {dias} dias)"
    return None


def analizar_excel(ruta_archivo):
    """Ejecuta el análisis completo y devuelve resultados estructurados."""
    df = pd.read_excel(ruta_archivo, header=1, keep_default_na=False)
    df.columns = ["area", "fecha", "intentos", "ACD", "ASR", "ABR"]

    df = df[df["area"].astype(str).str.strip() != ""]
    df = df[df["fecha"].astype(str).str.strip() != ""]

    for col in ["intentos", "ACD", "ASR", "ABR"]:
        df[col] = limpiar_numero(df[col])

    df["fecha"] = parsear_fechas(df["fecha"].astype(str))
    df = df[df["fecha"].notna()]

    fecha_hoy = df["fecha"].max().normalize()
    df_hoy = df[df["fecha"].dt.normalize() == fecha_hoy]
    df_hist = df[df["fecha"].dt.normalize() < fecha_hoy]

    columnas_ia = ["intentos", "ACD", "ASR", "ABR"]
    resultados = []
    total_areas = 0

    for area in df_hoy["area"].unique():
        registros_hoy = df_hoy[df_hoy["area"] == area]
        if registros_hoy.empty:
            continue
        ultimo = registros_hoy.iloc[0]

        historico = df_hist[df_hist["area"] == area].sort_values("fecha")
        if len(historico) < 2:
            continue

        total_areas += 1

        ruta_modelo = nombre_modelo(area)
        if os.path.exists(ruta_modelo):
            modelo_ia = joblib.load(ruta_modelo)
        else:
            modelo_ia = IsolationForest(contamination=0.1, random_state=42)

        modelo_ia.fit(historico[columnas_ia])
        joblib.dump(modelo_ia, ruta_modelo)

        prediccion = modelo_ia.predict(ultimo[columnas_ia].values.reshape(1, -1))
        es_anomalia = prediccion[0] == -1

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

        estado = "CRITICO" if motivos else "REVISAR (IA detecto patron inusual)"

        resultados.append({
            "area": area,
            "fecha": ultimo["fecha"].strftime("%b %d %H:%M"),
            "estado": estado,
            "motivos": motivos,
            "valores": {
                "intentos": int(ultimo["intentos"]),
                "ACD": round(ultimo["ACD"], 2),
                "ASR": round(ultimo["ASR"], 2),
                "ABR": round(ultimo["ABR"], 2),
            },
        })

    return {
        "fecha_analizada": fecha_hoy.strftime("%b %d, %Y"),
        "dias_historicos": df_hist["fecha"].dt.normalize().nunique(),
        "total_areas": total_areas,
        "alertas": resultados,
        "sin_alertas": len(resultados) == 0,
    }


# ═════════════════════════════════════════
# RUTAS — AUTENTICACIÓN
# ═════════════════════════════════════════
@app.route("/")
def index():
    usuario = session.get("usuario")
    return render_template("login.html", usuario=usuario)


@app.route("/login/<proveedor>")
def login(proveedor):
    if proveedor not in ("google", "github"):
        flash("Proveedor no soportado", "danger")
        return redirect(url_for("index"))
    redirect_uri = url_for(f"authorize", proveedor=proveedor, _external=True)
    return oauth.create_client(proveedor).authorize_redirect(redirect_uri)


@app.route("/authorize/<proveedor>")
def authorize(proveedor):
    if proveedor not in ("google", "github"):
        flash("Proveedor no soportado", "danger")
        return redirect(url_for("index"))
    try:
        token = oauth.create_client(proveedor).authorize_access_token()
    except Exception as e:
        flash(f"Error de autenticacion: {str(e)}", "danger")
        return redirect(url_for("index"))

    if proveedor == "google":
        userinfo = token.get("userinfo")
        if not userinfo:
            userinfo = oauth.google.parse_id_token(token)
        session["usuario"] = {
            "nombre": userinfo.get("name", "Usuario Google"),
            "email": userinfo.get("email", ""),
            "foto": userinfo.get("picture", ""),
            "proveedor": "Google",
        }
    elif proveedor == "github":
        resp = oauth.github.get("user", token=token)
        userinfo = resp.json()
        session["usuario"] = {
            "nombre": userinfo.get("name") or userinfo.get("login", "Usuario GitHub"),
            "email": userinfo.get("email", ""),
            "foto": userinfo.get("avatar_url", ""),
            "proveedor": "GitHub",
        }

    flash(f"Inicio de sesion exitoso con {proveedor}", "success")
    return redirect(url_for("dashboard"))


@app.route("/logout")
def logout():
    session.pop("usuario", None)
    flash("Sesion cerrada", "info")
    return redirect(url_for("index"))


# ═════════════════════════════════════════
# RUTAS — DASHBOARD
# ═════════════════════════════════════════
@app.route("/dashboard")
def dashboard():
    usuario = session.get("usuario")
    if not usuario:
        flash("Debes iniciar sesion primero", "warning")
        return redirect(url_for("index"))
    return render_template("dashboard.html", usuario=usuario)


@app.route("/upload", methods=["POST"])
def upload():
    usuario = session.get("usuario")
    if not usuario:
        return jsonify({"error": "No autenticado"}), 401

    if "file" not in request.files:
        flash("No se selecciono ningun archivo", "danger")
        return redirect(url_for("dashboard"))

    file = request.files["file"]
    if file.filename == "":
        flash("No se selecciono ningun archivo", "danger")
        return redirect(url_for("dashboard"))

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        flash("Formato no soportado. Solo .xlsx o .xls", "danger")
        return redirect(url_for("dashboard"))

    filename = secure_filename(f"{secrets.token_hex(8)}_{file.filename}")
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    try:
        resultados = analizar_excel(filepath)
        os.remove(filepath)
        return render_template("resultados.html", usuario=usuario, datos=resultados)
    except Exception as e:
        if os.path.exists(filepath):
            os.remove(filepath)
        flash(f"Error al analizar el archivo: {str(e)}", "danger")
        return redirect(url_for("dashboard"))


# ═════════════════════════════════════════
# INICIAR
# ═════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 50)
    print("Servidor VoIP iniciado")
    print(f"  http://localhost:5000")
    print("=" * 50)
    app.run(debug=True, port=5000)
