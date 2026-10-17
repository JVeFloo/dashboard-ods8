# =============================================================================
# DASHBOARD v2 - Estudio del Desempleo y la Informalidad Laboral en México
# Proyecto Integrador IDIA · Universidad Politécnica de Querétaro · ODS 8
# =============================================================================
# Desplegado en Streamlit Community Cloud.
# Estructura esperada (todos los archivos en la raíz del repositorio):
#   dashboard_v2.py
#   requirements.txt
#   enoe_final.csv
#   features_clustering.csv
#   features_forecasting.csv
# =============================================================================

import os
import glob
import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

# -----------------------------------------------------------------------------
# CONFIGURACIÓN DE PÁGINA
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Mercado Laboral en México · ODS 8",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# PALETA DE COLORES DEL PROYECTO (misma del reporte y el póster)
# -----------------------------------------------------------------------------
COLOR_INDIGO       = "#11347B"   # principal
COLOR_INDIGO_DARK  = "#0B2358"
COLOR_ACCENT       = "#A6192E"   # rojo UPQ
COLOR_TEAL         = "#0E7C7B"
COLOR_ORANGE       = "#E07A5F"   # complemento cálido
COLOR_MUSTARD      = "#D4A72C"
COLOR_INK          = "#1E293B"
COLOR_INK_SOFT     = "#5A6B8C"
COLOR_BG_SOFT      = "#EEF1F7"
COLOR_CARD_BORDER  = "#D7DEEC"
COLOR_HOOK_BG      = "#FDF3F4"

# Paleta cíclica para gráficas categóricas
PALETA_CATEGORICA = [COLOR_INDIGO, COLOR_ACCENT, COLOR_TEAL, COLOR_ORANGE, COLOR_MUSTARD]

# Colormap índigo→rojo para mapas de calor / barras ordenadas
CMAP_PROYECTO = LinearSegmentedColormap.from_list(
    "proyecto", ["#DDE3F0", COLOR_INDIGO, COLOR_ACCENT], N=256
)

# -----------------------------------------------------------------------------
# TEMA GLOBAL (CSS): fuerza modo claro con paleta del proyecto
# -----------------------------------------------------------------------------
CSS_GLOBAL = f"""
<style>
    /* Fondo general y forzar modo claro */
    .stApp {{
        background: linear-gradient(180deg, {COLOR_BG_SOFT} 0%, #F7F9FC 100%);
        color: {COLOR_INK};
    }}
    [data-testid="stHeader"] {{
        background: transparent;
    }}
    /* Sidebar con degradado índigo */
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, {COLOR_INDIGO_DARK} 0%, {COLOR_INDIGO} 100%);
    }}
    [data-testid="stSidebar"] * {{
        color: white !important;
    }}
    [data-testid="stSidebar"] .stRadio label {{
        padding: 10px 12px;
        border-radius: 8px;
        margin-bottom: 4px;
        transition: background 0.2s;
    }}
    [data-testid="stSidebar"] .stRadio label:hover {{
        background: rgba(255,255,255,0.10);
    }}
    /* Selectbox y otros controles del sidebar en blanco */
    [data-testid="stSidebar"] .stSelectbox > div > div,
    [data-testid="stSidebar"] .stTextInput > div > div > input {{
        background: rgba(255,255,255,0.15) !important;
        color: white !important;
        border: 1px solid rgba(255,255,255,0.3) !important;
    }}
    /* Título principal */
    h1 {{
        color: {COLOR_INDIGO};
        font-weight: 700;
        border-bottom: 4px solid {COLOR_ACCENT};
        padding-bottom: 8px;
        margin-top: 0;
    }}
    h2 {{
        color: {COLOR_INDIGO};
        font-weight: 700;
        margin-top: 24px;
    }}
    h3 {{
        color: {COLOR_INK};
        font-weight: 600;
    }}
    /* Tarjetas KPI */
    .kpi-card {{
        background: white;
        border-radius: 14px;
        padding: 22px 24px;
        box-shadow: 0 4px 16px rgba(17,52,123,0.08);
        border-top: 5px solid {COLOR_INDIGO};
        text-align: center;
        height: 100%;
    }}
    .kpi-card.accent  {{ border-top-color: {COLOR_ACCENT}; }}
    .kpi-card.teal    {{ border-top-color: {COLOR_TEAL}; }}
    .kpi-card.mustard {{ border-top-color: {COLOR_MUSTARD}; }}
    .kpi-label {{
        font-size: 12px;
        font-weight: 700;
        color: {COLOR_INK_SOFT};
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 6px;
    }}
    .kpi-value {{
        font-size: 30px;
        font-weight: 800;
        color: {COLOR_INDIGO};
        line-height: 1.1;
    }}
    .kpi-value.accent  {{ color: {COLOR_ACCENT}; }}
    .kpi-value.teal    {{ color: {COLOR_TEAL}; }}
    .kpi-value.mustard {{ color: {COLOR_MUSTARD}; }}
    /* Cajas de contexto */
    .box-info {{
        background: white;
        border-left: 5px solid {COLOR_INDIGO};
        border-radius: 10px;
        padding: 16px 20px;
        margin: 12px 0;
        box-shadow: 0 2px 6px rgba(17,52,123,0.05);
    }}
    .box-hook {{
        background: {COLOR_HOOK_BG};
        border-left: 5px solid {COLOR_ACCENT};
        border-radius: 10px;
        padding: 16px 20px;
        margin: 12px 0;
        font-size: 15px;
    }}
    .box-success {{
        background: #E9F5F4;
        border-left: 5px solid {COLOR_TEAL};
        border-radius: 10px;
        padding: 16px 20px;
        margin: 12px 0;
    }}
    .box-warning {{
        background: #FFF7E6;
        border-left: 5px solid {COLOR_MUSTARD};
        border-radius: 10px;
        padding: 16px 20px;
        margin: 12px 0;
    }}
    /* Chips de cluster */
    .chip {{
        display: inline-block;
        padding: 6px 14px;
        border-radius: 20px;
        color: white;
        font-weight: 600;
        font-size: 13px;
        margin: 4px 4px 4px 0;
    }}
    /* Botones */
    .stButton > button,
    .stDownloadButton > button {{
        background: {COLOR_INDIGO} !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 10px 22px !important;
        font-weight: 600 !important;
        transition: all 0.2s !important;
    }}
    .stButton > button:hover,
    .stDownloadButton > button:hover {{
        background: {COLOR_INDIGO_DARK} !important;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(17,52,123,0.25);
    }}
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 6px;
        background: white;
        padding: 6px;
        border-radius: 10px;
    }}
    .stTabs [data-baseweb="tab"] {{
        border-radius: 8px;
        padding: 10px 20px;
        color: {COLOR_INK_SOFT};
        font-weight: 600;
    }}
    .stTabs [aria-selected="true"] {{
        background: {COLOR_INDIGO} !important;
        color: white !important;
    }}
    /* Métricas nativas de Streamlit */
    [data-testid="stMetric"] {{
        background: white;
        border-radius: 10px;
        padding: 14px;
        box-shadow: 0 2px 8px rgba(17,52,123,0.06);
        border-left: 4px solid {COLOR_INDIGO};
    }}
    [data-testid="stMetricLabel"] {{
        color: {COLOR_INK_SOFT};
        font-weight: 600;
    }}
    [data-testid="stMetricValue"] {{
        color: {COLOR_INDIGO};
        font-weight: 700;
    }}
    /* Expander */
    .streamlit-expanderHeader {{
        background: white;
        border-radius: 8px;
        font-weight: 600;
        color: {COLOR_INDIGO};
    }}
    /* Ocultar hamburguesa y footer */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
</style>
"""
st.markdown(CSS_GLOBAL, unsafe_allow_html=True)

# Estilo global de matplotlib para que todas las gráficas usen la paleta
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": COLOR_INK_SOFT,
    "axes.labelcolor": COLOR_INK,
    "axes.titlecolor": COLOR_INDIGO,
    "axes.titleweight": "bold",
    "axes.titlesize": 13,
    "xtick.color": COLOR_INK_SOFT,
    "ytick.color": COLOR_INK_SOFT,
    "grid.color": COLOR_BG_SOFT,
    "axes.grid": True,
    "axes.axisbelow": True,
    "font.size": 10,
})

# =============================================================================
# CARGA DE DATOS Y MODELOS (una sola vez, cacheado)
# =============================================================================
def _localizar(nombre, candidatos):
    for c in candidatos:
        if os.path.exists(c):
            return c
    for m in glob.glob(os.path.join(".", "**", nombre), recursive=True):
        if "Trash" not in m and ".ipynb_checkpoints" not in m:
            return m
    raise FileNotFoundError(
        f"No encontré '{nombre}'. Corre el dashboard desde la carpeta del proyecto."
    )

@st.cache_data
def cargar_datos():
    ruta = _localizar("enoe_final.csv", [
        "data/enoe_final.csv", "data/raw/enoe_final.csv", "src/enoe_final.csv",
    ])
    df = pd.read_csv(ruta, dtype={"clave_entidad": str, "id_indicador": str})
    df["valor"] = pd.to_numeric(df["valor"], errors="coerce")
    df["anio"] = pd.to_numeric(df["anio"], errors="coerce").astype("Int64")
    return df

@st.cache_data
def cargar_features():
    rc = _localizar("features_clustering.csv", [
        "data/processed/features_clustering.csv",
    ])
    rf = _localizar("features_forecasting.csv", [
        "data/processed/features_forecasting.csv",
    ])
    return pd.read_csv(rc), pd.read_csv(rf)

@st.cache_resource
def entrenar_modelos():
    clu, fore = cargar_features()

    # K-Means idéntico al del reporte
    X = clu[["tasa_desocupacion", "tasa_informalidad", "tasa_participacion"]]
    Xs = StandardScaler().fit_transform(X)
    km = KMeans(n_clusters=3, random_state=42, n_init=10)
    clu = clu.copy()
    clu["cluster"] = km.fit_predict(Xs)
    perfil = clu.groupby("cluster")[["tasa_desocupacion","tasa_informalidad","tasa_participacion"]].mean()

    # Etiquetas interpretables según el perfil real
    etiquetas = {}
    etiquetas[perfil["tasa_informalidad"].idxmax()] = "Sur informal"
    etiquetas[perfil["tasa_desocupacion"].idxmax()] = "Norte/Centro formal"
    restante = [c for c in perfil.index if c not in etiquetas][0]
    etiquetas[restante] = "Costero/turístico"
    clu["perfil"] = clu["cluster"].map(etiquetas)

    # Colores por cluster (mismos del póster)
    color_cluster = {
        "Norte/Centro formal": COLOR_INDIGO,
        "Sur informal":        COLOR_ACCENT,
        "Costero/turístico":   COLOR_TEAL,
    }
    clu["color"] = clu["perfil"].map(color_cluster)

    # Regresión lineal — evaluación cronológica
    F = ["lag_1","lag_2","lag_4","media_movil_4"]
    n_test = 12
    train = fore.iloc[:-n_test]
    reg_eval = LinearRegression().fit(train[F], train["tasa_desocupacion"])
    y_pred_test = reg_eval.predict(fore.iloc[-n_test:][F])
    mae = float(np.mean(np.abs(y_pred_test - fore.iloc[-n_test:]["tasa_desocupacion"])))

    # Modelo final entrenado con todos los datos
    reg_final = LinearRegression().fit(fore[F], fore["tasa_desocupacion"])
    return clu, perfil, etiquetas, color_cluster, reg_final, reg_eval, mae, fore

# =============================================================================
# HELPERS DE UI
# =============================================================================
def kpi(col, label, value, variante="indigo"):
    with col:
        st.markdown(
            f'<div class="kpi-card {variante}">'
            f'<div class="kpi-label">{label}</div>'
            f'<div class="kpi-value {variante}">{value}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

def box(kind, html):
    st.markdown(f'<div class="box-{kind}">{html}</div>', unsafe_allow_html=True)

def estilo_ejes(ax):
    for s in ["top","right"]:
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(COLOR_INK_SOFT)
    ax.spines["bottom"].set_color(COLOR_INK_SOFT)

# =============================================================================
# CARGA INICIAL
# =============================================================================
try:
    df_master = cargar_datos()
    clu, perfil, etiquetas, color_cluster, reg_final, reg_eval, mae_test, fore = entrenar_modelos()
except FileNotFoundError as e:
    st.error(f"⚠️ No se pudieron cargar los datos: {e}")
    st.stop()

# =============================================================================
# SIDEBAR: navegación y filtros
# =============================================================================
with st.sidebar:
    st.markdown(
        f"<div style='text-align:center; padding: 12px 0 8px 0;'>"
        f"<div style='font-size:32px'>📊</div>"
        f"<h3 style='color:white; margin:6px 0 4px 0; font-size:18px;'>Mercado Laboral</h3>"
        f"<div style='color:rgba(255,255,255,0.7); font-size:12px;'>México · ODS 8</div>"
        f"</div>",
        unsafe_allow_html=True,
    )
    st.markdown("<hr style='border-color:rgba(255,255,255,0.15); margin:12px 0;'>", unsafe_allow_html=True)

    seccion = st.radio(
        "Navegación",
        [
            "🏠 Vista General",
            "🗺️ Mapa Interactivo",
            "📍 Análisis por Estado",
            "📈 Tendencias Históricas",
            "🤖 Laboratorio de Predicción",
            "💡 Guía Interactiva ODS 8",
            "❓ Preguntas Frecuentes",
        ],
        label_visibility="collapsed",
    )

    st.markdown("<hr style='border-color:rgba(255,255,255,0.15); margin:16px 0;'>", unsafe_allow_html=True)
    st.markdown("**Filtros globales**")
    anios_disp = sorted(df_master["anio"].dropna().unique().tolist(), reverse=True)
    anio_global = st.selectbox("Año", anios_disp, index=0)
    trim_global = st.selectbox("Trimestre", [1,2,3,4], index=0)

    st.markdown("<hr style='border-color:rgba(255,255,255,0.15); margin:16px 0;'>", unsafe_allow_html=True)
    st.markdown(
        f"<div style='font-size:11px; color:rgba(255,255,255,0.6); text-align:center;'>"
        f"Universidad Politécnica<br>de Querétaro · IDIA 224</div>",
        unsafe_allow_html=True,
    )

# =============================================================================
# HELPER: obtener valor de un indicador
# =============================================================================
def val_indicador(df, patron, entidad=None, anio=None, trim=None):
    sub = df[df["nombre_indicador"].str.contains(patron, case=False, na=False, regex=True)]
    if entidad is not None: sub = sub[sub["entidad"] == entidad]
    if anio is not None:    sub = sub[sub["anio"] == anio]
    if trim is not None:    sub = sub[sub["trimestre"].astype(str) == str(trim)]
    return sub["valor"].mean() if not sub.empty else np.nan

# =============================================================================
# SECCIÓN 1: VISTA GENERAL
# =============================================================================
if seccion == "🏠 Vista General":
    st.markdown("# Desempleo e Informalidad en México")
    st.markdown(
        f"<div style='color:{COLOR_INK_SOFT}; font-size:16px; margin-top:-8px;'>"
        f"Periodo de observación: <b>{anio_global} · Trimestre {trim_global}</b></div>",
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    NAC = "Estados Unidos Mexicanos"
    pea    = val_indicador(df_master, r"Economicamente Activa \(PEA\)", NAC, anio_global, trim_global)
    ocup   = val_indicador(df_master, r"^Poblacion ocupada$", NAC, anio_global, trim_global)
    desoc  = val_indicador(df_master, r"^Poblacion desocupada$", NAC, anio_global, trim_global)
    td     = val_indicador(df_master, r"Tasa de Desocupacion", NAC, anio_global, trim_global)
    tinf   = val_indicador(df_master, r"Informalidad Laboral total", NAC, anio_global, trim_global)

    c1, c2, c3, c4, c5 = st.columns(5)
    kpi(c1, "PEA Nacional",       f"{int(pea):,}"   if pd.notna(pea)   else "N/D", "indigo")
    kpi(c2, "Ocupados",           f"{int(ocup):,}"  if pd.notna(ocup)  else "N/D", "teal")
    kpi(c3, "Desocupados",        f"{int(desoc):,}" if pd.notna(desoc) else "N/D", "accent")
    kpi(c4, "Tasa Desocupación",  f"{td:.2f}%"      if pd.notna(td)    else "N/D", "accent")
    kpi(c5, "Tasa Informalidad",  f"{tinf:.2f}%"    if pd.notna(tinf)  else "N/D", "mustard")

    # Ranking estatal de desocupación
    st.markdown("## 📊 Ranking Estatal de Desocupación")

    df_rank = df_master[
        (df_master["nombre_indicador"].str.contains("Tasa de Desocupacion", case=False, na=False)) &
        (df_master["nivel"] == "Estatal") &
        (df_master["anio"] == anio_global) &
        (df_master["trimestre"].astype(str) == str(trim_global))
    ].groupby("entidad")["valor"].mean().sort_values(ascending=False)

    if not df_rank.empty:
        fig, ax = plt.subplots(figsize=(13, 6))
        colores = CMAP_PROYECTO(np.linspace(0.95, 0.15, len(df_rank)))
        ax.bar(range(len(df_rank)), df_rank.values, color=colores, edgecolor="white", linewidth=0.5)
        ax.set_xticks(range(len(df_rank)))
        ax.set_xticklabels(df_rank.index, rotation=75, ha="right", fontsize=8)
        ax.set_ylabel("Tasa de desocupación (%)")
        ax.set_title(f"Desocupación por entidad · {anio_global}/T{trim_global}")
        # Anotar máximo y mínimo
        ax.annotate(f"↑ {df_rank.iloc[0]:.2f}%", xy=(0, df_rank.iloc[0]),
                    xytext=(0, 4), textcoords="offset points", ha="center",
                    fontsize=9, fontweight="bold", color=COLOR_ACCENT)
        ax.annotate(f"↓ {df_rank.iloc[-1]:.2f}%", xy=(len(df_rank)-1, df_rank.iloc[-1]),
                    xytext=(0, 4), textcoords="offset points", ha="center",
                    fontsize=9, fontweight="bold", color=COLOR_TEAL)
        estilo_ejes(ax)
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.info("No hay datos estatales para el filtro seleccionado.")

    # Lectura ODS 8
    if pd.notna(td) and pd.notna(tinf):
        estado_td = "históricamente bajo" if td < 3.5 else "en niveles moderados"
        estado_ti = "crítico (más del 50%)" if tinf > 50 else "en transición hacia la formalidad"
        box("hook",
            f"<b style='color:{COLOR_ACCENT};'>Lectura bajo el ODS 8:</b> "
            f"en <b>{anio_global}/T{trim_global}</b> México registra una desocupación "
            f"del <b>{td:.2f}%</b> ({estado_td}) junto con una informalidad del <b>{tinf:.2f}%</b> "
            f"({estado_ti}). Esta combinación ilustra la paradoja laboral mexicana: "
            f"sin seguro de desempleo, la población se refugia en el sector informal, por lo "
            f"que una tasa de desempleo baja no equivale a trabajo decente.")

    # Descargas
    st.markdown("## 📥 Descarga de datos")
    c1, c2 = st.columns(2)
    with c1:
        csv_full = df_master.to_csv(index=False).encode("utf-8-sig")
        st.download_button("📄 Dataset ENOE completo (CSV)", csv_full,
                          "enoe_completo.csv", "text/csv", use_container_width=True)
    with c2:
        csv_periodo = df_master[
            (df_master["anio"] == anio_global) &
            (df_master["trimestre"].astype(str) == str(trim_global))
        ].to_csv(index=False).encode("utf-8-sig")
        st.download_button(f"📄 Solo {anio_global}/T{trim_global} (CSV)",
                          csv_periodo, f"enoe_{anio_global}_T{trim_global}.csv",
                          "text/csv", use_container_width=True)


# =============================================================================
# SECCIÓN 2: MAPA INTERACTIVO
# =============================================================================
elif seccion == "🗺️ Mapa Interactivo":
    st.markdown("# Mapa Interactivo de Indicadores Laborales")

    box("info",
        f"<b style='color:{COLOR_INDIGO};'>Guía de lectura territorial (ODS 8):</b><br>"
        f"Este mapa muestra la distribución espacial del mercado laboral. Elige "
        f"un indicador y observa qué regiones están mejor o peor posicionadas. "
        f"Los tonos <b>oscuros</b> indican valores altos; los <b>claros</b>, valores bajos.")

    indicadores_map = {
        "Tasa de Desocupación":  "Tasa de Desocupacion",
        "Tasa de Informalidad":  "Informalidad Laboral total",
        "Tasa de Participación": "Tasa Neta de Participacion",
    }
    c1, c2 = st.columns([2, 3])
    with c1:
        ind_label = st.radio("Indicador a visualizar:", list(indicadores_map.keys()))
    with c2:
        st.markdown(f"<br><div style='color:{COLOR_INK_SOFT};'>"
                    f"Vista: <b>{anio_global}/T{trim_global}</b></div>", unsafe_allow_html=True)

    ind_patron = indicadores_map[ind_label]
    df_mapa = df_master[
        (df_master["nombre_indicador"].str.contains(ind_patron, case=False, na=False)) &
        (df_master["nivel"] == "Estatal") &
        (df_master["anio"] == anio_global) &
        (df_master["trimestre"].astype(str) == str(trim_global))
    ].groupby("entidad")["valor"].mean().reset_index()

    coords = {
        'Aguascalientes':[21.88,-102.29],'Baja California':[30.84,-115.28],
        'Baja California Sur':[26.04,-111.67],'Campeche':[19.83,-90.53],
        'Coahuila de Zaragoza':[27.06,-101.71],'Colima':[19.12,-104.01],
        'Chiapas':[16.76,-93.13],'Chihuahua':[28.63,-106.07],
        'Ciudad de Mexico':[19.43,-99.13],'Durango':[24.03,-104.65],
        'Guanajuato':[21.02,-101.26],'Guerrero':[17.44,-99.55],
        'Hidalgo':[20.09,-98.76],'Jalisco':[20.66,-103.35],
        'Mexico':[19.36,-99.64],'Michoacan de Ocampo':[19.57,-101.71],
        'Morelos':[18.68,-99.10],'Nayarit':[21.75,-104.85],
        'Nuevo Leon':[25.59,-100.05],'Oaxaca':[17.07,-96.73],
        'Puebla':[19.04,-98.21],'Queretaro':[20.59,-100.39],
        'Quintana Roo':[19.18,-88.48],'San Luis Potosi':[22.16,-100.99],
        'Sinaloa':[25.17,-107.48],'Sonora':[29.69,-110.66],
        'Tabasco':[17.84,-92.62],'Tamaulipas':[24.27,-98.84],
        'Tlaxcala':[19.32,-98.19],'Veracruz de Ignacio de la Llave':[19.17,-96.13],
        'Yucatan':[20.70,-89.09],'Zacatecas':[22.77,-102.58],
    }
    df_mapa["lat"] = df_mapa["entidad"].map(lambda x: coords.get(x, [np.nan, np.nan])[0])
    df_mapa["lon"] = df_mapa["entidad"].map(lambda x: coords.get(x, [np.nan, np.nan])[1])
    df_mapa = df_mapa.dropna(subset=["lat","lon"])

    if not df_mapa.empty:
        # Círculos proporcionales (más grandes = valor más alto)
        vmax = df_mapa["valor"].max()
        df_mapa["size"] = (df_mapa["valor"] / (vmax if vmax > 0 else 1)) * 55000 + 8000
        # Color según cluster (para dar contexto)
        estado_a_cluster = clu.set_index("entidad")["perfil"].to_dict()
        df_mapa["cluster"] = df_mapa["entidad"].map(estado_a_cluster).fillna("—")
        df_mapa["color"] = df_mapa["cluster"].map(color_cluster).fillna(COLOR_INK_SOFT)

        st.map(df_mapa, latitude="lat", longitude="lon", size="size", color="color")

        # Top 5 y bottom 5
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"##### 🔺 Top 5 más alto")
            top = df_mapa.nlargest(5, "valor")[["entidad","valor"]]
            for _, r in top.iterrows():
                st.markdown(
                    f"<div style='background:white; padding:8px 12px; margin:4px 0; "
                    f"border-radius:6px; border-left:3px solid {COLOR_ACCENT};'>"
                    f"<b>{r['entidad']}</b> — {r['valor']:.2f}%</div>",
                    unsafe_allow_html=True)
        with c2:
            st.markdown(f"##### 🔻 Top 5 más bajo")
            bot = df_mapa.nsmallest(5, "valor")[["entidad","valor"]]
            for _, r in bot.iterrows():
                st.markdown(
                    f"<div style='background:white; padding:8px 12px; margin:4px 0; "
                    f"border-radius:6px; border-left:3px solid {COLOR_TEAL};'>"
                    f"<b>{r['entidad']}</b> — {r['valor']:.2f}%</div>",
                    unsafe_allow_html=True)

        box("info",
            f"<b>Colores del mapa:</b> "
            f"<span class='chip' style='background:{COLOR_INDIGO};'>Norte/Centro formal</span> "
            f"<span class='chip' style='background:{COLOR_ACCENT};'>Sur informal</span> "
            f"<span class='chip' style='background:{COLOR_TEAL};'>Costero/turístico</span><br>"
            f"El tamaño del círculo refleja el valor del indicador seleccionado.")
    else:
        st.warning("No hay datos estatales para el indicador y periodo seleccionados.")


# =============================================================================
# SECCIÓN 3: ANÁLISIS POR ESTADO
# =============================================================================
elif seccion == "📍 Análisis por Estado":
    st.markdown("# Análisis por Entidad Federativa")

    estados = sorted(df_master[df_master["nivel"] == "Estatal"]["entidad"].unique())
    estado_sel = st.selectbox("Selecciona una entidad:", estados)

    td_est  = val_indicador(df_master, r"Tasa de Desocupacion", estado_sel, anio_global, trim_global)
    ti_est  = val_indicador(df_master, r"Informalidad Laboral total", estado_sel, anio_global, trim_global)
    tp_est  = val_indicador(df_master, r"Tasa Neta de Participacion", estado_sel, anio_global, trim_global)
    ocup_est = val_indicador(df_master, r"^Poblacion ocupada$", estado_sel, anio_global, trim_global)

    # Semáforo de vulnerabilidad
    if pd.notna(td_est) and pd.notna(ti_est):
        if td_est < 3.0 and ti_est > 65:
            box("hook",
                f"<h3 style='color:{COLOR_ACCENT}; margin:0 0 8px 0;'>🚨 Alerta de vulnerabilidad laboral</h3>"
                f"<b>{estado_sel}</b> presenta la paradoja en su forma más extrema: desocupación baja "
                f"({td_est:.2f}%) pero informalidad muy alta ({ti_est:.1f}%). Casi todo el empleo carece "
                f"de protecciones básicas del trabajo decente.")
        elif ti_est > 55:
            box("warning",
                f"<h3 style='color:{COLOR_MUSTARD}; margin:0 0 8px 0;'>⚠️ Condiciones mixtas</h3>"
                f"<b>{estado_sel}</b> tiene una informalidad ({ti_est:.1f}%) que supera el promedio nacional. "
                f"Requiere atención en políticas de formalización.")
        else:
            box("success",
                f"<h3 style='color:{COLOR_TEAL}; margin:0 0 8px 0;'>✅ Perfil relativamente formal</h3>"
                f"<b>{estado_sel}</b> tiene una informalidad ({ti_est:.1f}%) por debajo del promedio nacional, "
                f"lo que sugiere un mercado laboral más protegido.")

    # KPIs por estado
    st.markdown("## Indicadores principales")
    c1, c2, c3, c4 = st.columns(4)
    kpi(c1, "Desocupación",        f"{td_est:.2f}%"      if pd.notna(td_est) else "N/D",  "accent")
    kpi(c2, "Informalidad",        f"{ti_est:.2f}%"      if pd.notna(ti_est) else "N/D",  "mustard")
    kpi(c3, "Participación",       f"{tp_est:.2f}%"      if pd.notna(tp_est) else "N/D",  "teal")
    kpi(c4, "Ocupados",            f"{int(ocup_est):,}"  if pd.notna(ocup_est) else "N/D", "indigo")

    # Comparativa por sexo
    st.markdown("## 👥 Comparativa por sexo")
    tabs = st.tabs(["Desocupación", "Informalidad", "Ocupación"])

    def barras_sexo(patron_h, patron_m, titulo, ax):
        vh = val_indicador(df_master, patron_h, estado_sel, anio_global, trim_global)
        vm = val_indicador(df_master, patron_m, estado_sel, anio_global, trim_global)
        if pd.isna(vh) or pd.isna(vm):
            ax.text(0.5, 0.5, "Datos no disponibles", ha="center", va="center",
                    transform=ax.transAxes, color=COLOR_INK_SOFT)
            ax.axis("off"); return
        ax.bar(["Hombres","Mujeres"], [vh, vm], color=[COLOR_INDIGO, COLOR_ACCENT],
               edgecolor="white", linewidth=1.5, width=0.55)
        for i, v in enumerate([vh, vm]):
            ax.text(i, v, f"{v:,.2f}", ha="center", va="bottom", fontweight="bold",
                    color=COLOR_INK, fontsize=11)
        ax.set_title(titulo); ax.set_ylabel("Valor")
        estilo_ejes(ax); ax.margins(y=0.15)

    with tabs[0]:
        fig, ax = plt.subplots(figsize=(6, 4))
        barras_sexo(r"Poblacion desocupada \(hombres\)", r"Poblacion desocupada \(mujeres\)",
                    f"Población desocupada · {estado_sel}", ax)
        plt.tight_layout(); st.pyplot(fig)
    with tabs[1]:
        fig, ax = plt.subplots(figsize=(6, 4))
        barras_sexo(r"Informalidad Laboral hombres", r"Informalidad Laboral mujeres",
                    f"Tasa de informalidad (%) · {estado_sel}", ax)
        plt.tight_layout(); st.pyplot(fig)
    with tabs[2]:
        fig, ax = plt.subplots(figsize=(6, 4))
        barras_sexo(r"Poblacion ocupada \(hombres\)", r"Poblacion ocupada \(mujeres\)",
                    f"Población ocupada · {estado_sel}", ax)
        plt.tight_layout(); st.pyplot(fig)

    # Ubicación en clusters
    if estado_sel in clu["entidad"].values:
        perfil_est = clu[clu["entidad"] == estado_sel].iloc[0]
        box("info",
            f"<b>Clasificación del algoritmo K-Means:</b> "
            f"<span class='chip' style='background:{perfil_est['color']};'>{perfil_est['perfil']}</span>")


# =============================================================================
# SECCIÓN 4: TENDENCIAS HISTÓRICAS
# =============================================================================
elif seccion == "📈 Tendencias Históricas":
    st.markdown("# Evolución Temporal del Mercado Laboral")

    box("info",
        f"Esta sección muestra cómo han cambiado los indicadores clave a lo largo de "
        f"<b>21 años (2005–2026)</b>. Las áreas sombreadas destacan los dos grandes "
        f"choques del periodo: la <b>crisis financiera de 2009</b> y la "
        f"<b>pandemia de COVID-19 (2020)</b>.")

    ambito = st.radio("Ámbito:", ["Nacional", "Estatal"], horizontal=True)
    if ambito == "Estatal":
        estado_sel = st.selectbox("Entidad:", sorted(df_master[df_master["nivel"]=="Estatal"]["entidad"].unique()))
    else:
        estado_sel = "Estados Unidos Mexicanos"

    # Serie desocupación
    serie_td = df_master[
        (df_master["nombre_indicador"].str.contains("Tasa de Desocupacion", case=False, na=False)) &
        (df_master["entidad"] == estado_sel)
    ].groupby(["anio","trimestre"])["valor"].mean().reset_index().sort_values(["anio","trimestre"])

    # Serie informalidad
    serie_ti = df_master[
        (df_master["nombre_indicador"].str.contains("Informalidad Laboral total", case=False, na=False)) &
        (df_master["entidad"] == estado_sel)
    ].groupby(["anio","trimestre"])["valor"].mean().reset_index().sort_values(["anio","trimestre"])

    if not serie_td.empty:
        serie_td["periodo"] = serie_td["anio"].astype(str) + "-T" + serie_td["trimestre"].astype(str)
        serie_ti["periodo"] = serie_ti["anio"].astype(str) + "-T" + serie_ti["trimestre"].astype(str)

        fig, ax = plt.subplots(figsize=(13, 5))
        # Sombreado de crisis
        for i, (year_ini, year_fin, etiqueta) in enumerate([(2009,2009,"Crisis 2009"), (2020,2020,"COVID-19")]):
            mask_ini = (serie_td["anio"] == year_ini)
            mask_fin = (serie_td["anio"] == year_fin)
            if mask_ini.any():
                x0 = serie_td[mask_ini].index[0]
                x1 = serie_td[mask_fin].index[-1]
                ax.axvspan(x0, x1, alpha=0.15, color=[COLOR_MUSTARD, COLOR_ACCENT][i], label=etiqueta)

        ax.plot(range(len(serie_td)), serie_td["valor"], color=COLOR_ACCENT, linewidth=2.2,
                marker="o", markersize=3.5, label="Desocupación (%)")
        if not serie_ti.empty:
            ax2 = ax.twinx()
            ax2.plot(range(len(serie_ti)), serie_ti["valor"], color=COLOR_INDIGO, linewidth=2.2,
                     marker="s", markersize=3.5, label="Informalidad (%)", alpha=0.9)
            ax2.set_ylabel("Informalidad (%)", color=COLOR_INDIGO)
            ax2.tick_params(axis="y", labelcolor=COLOR_INDIGO)
            ax2.grid(False)

        n = len(serie_td)
        ticks = list(range(0, n, max(1, n // 12)))
        ax.set_xticks(ticks)
        ax.set_xticklabels([serie_td.iloc[i]["periodo"] for i in ticks], rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Desocupación (%)", color=COLOR_ACCENT)
        ax.tick_params(axis="y", labelcolor=COLOR_ACCENT)
        ax.set_title(f"Evolución de desocupación e informalidad · {estado_sel}")
        ax.legend(loc="upper left", frameon=False)
        estilo_ejes(ax)
        plt.tight_layout(); st.pyplot(fig)

        # Estadísticas de la serie
        c1, c2, c3, c4 = st.columns(4)
        kpi(c1, "TD promedio",     f"{serie_td['valor'].mean():.2f}%",  "accent")
        kpi(c2, "TD máximo",       f"{serie_td['valor'].max():.2f}%",   "accent")
        kpi(c3, "TI promedio",     f"{serie_ti['valor'].mean():.2f}%"   if not serie_ti.empty else "N/D", "indigo")
        kpi(c4, "TI actual",       f"{serie_ti['valor'].iloc[-1]:.2f}%" if not serie_ti.empty else "N/D", "indigo")

        box("hook",
            f"<b>¿Cómo interpretar el gráfico conjunto?</b> La desocupación (rojo) es "
            f"<b>cíclica</b>: sube en crisis y baja en recuperaciones. La informalidad (azul) "
            f"es <b>estructural</b>: cambia lentamente sin importar el ciclo económico. "
            f"Esta diferencia es la clave para entender por qué México puede tener "
            f"desempleo bajo y a la vez déficit generalizado de trabajo decente.")
    else:
        st.warning("Sin datos para la selección actual.")


# =============================================================================
# SECCIÓN 5: LABORATORIO DE PREDICCIÓN
# =============================================================================
elif seccion == "🤖 Laboratorio de Predicción":
    st.markdown("# Laboratorio de Predicción y Agrupación")

    tab_clu, tab_for = st.tabs(["🎯 Segmentación (K-Means)", "📊 Pronóstico (Regresión)"])

    # ----- CLUSTERING -----
    with tab_clu:
        st.markdown("## Análisis de la estructura laboral")
        box("info",
            f"El algoritmo <b>K-Means</b> agrupa a los estados con condiciones laborales parecidas "
            f"en 3 perfiles definidos por colores. Silueta = <b>0.406</b> "
            f"(un valor alto para 32 entidades = agrupación clara).")

        # Gráfica de dispersión con clusters
        fig, ax = plt.subplots(figsize=(11, 6.5))
        for perfil_nombre, grupo in clu.groupby("perfil"):
            ax.scatter(grupo["tasa_desocupacion"], grupo["tasa_informalidad"],
                       s=140, c=color_cluster[perfil_nombre], edgecolor="white",
                       linewidth=1.5, label=perfil_nombre, alpha=0.85)
            for _, r in grupo.iterrows():
                ax.annotate(r["entidad"], (r["tasa_desocupacion"], r["tasa_informalidad"]),
                            fontsize=6.5, ha="center", va="bottom",
                            xytext=(0, 6), textcoords="offset points", color=COLOR_INK)
        ax.set_xlabel("Tasa de desocupación (%)")
        ax.set_ylabel("Tasa de informalidad (%)")
        ax.set_title("Segmentación laboral de los 32 estados")
        ax.legend(loc="upper right", frameon=True, framealpha=0.95)
        estilo_ejes(ax)
        plt.tight_layout(); st.pyplot(fig)

        # Guía de grupos
        st.markdown("## Guía de interpretación de grupos")
        c1, c2, c3 = st.columns(3)
        info_grupos = [
            (c1, "Norte/Centro formal", COLOR_INDIGO,
             "Estados industriales con más empleo formal, pero también más gente buscando trabajo. "
             "Nuevo León, CDMX, Chihuahua, Querétaro."),
            (c2, "Sur informal", COLOR_ACCENT,
             "La paradoja al extremo: casi no hay desempleo, pero 7 de cada 10 empleos son informales. "
             "Oaxaca, Chiapas, Guerrero, Puebla."),
            (c3, "Costero/turístico", COLOR_TEAL,
             "Economías de turismo con la mayor participación laboral. "
             "Quintana Roo, BCS, Yucatán, Colima, Nayarit."),
        ]
        for col, nombre, color, desc in info_grupos:
            with col:
                n_est = (clu["perfil"] == nombre).sum()
                st.markdown(
                    f"<div style='background:white; border-radius:12px; padding:16px; "
                    f"border-top:5px solid {color}; box-shadow:0 2px 8px rgba(17,52,123,0.08); height:100%;'>"
                    f"<div style='font-weight:700; color:{color}; margin-bottom:8px;'>{nombre}</div>"
                    f"<div style='font-size:24px; font-weight:700; color:{COLOR_INK};'>{n_est} estados</div>"
                    f"<div style='color:{COLOR_INK_SOFT}; font-size:13px; margin-top:8px;'>{desc}</div>"
                    f"</div>", unsafe_allow_html=True)

    # ----- PRONÓSTICO -----
    with tab_for:
        st.markdown("## Simulador de pronóstico")
        box("info",
            f"Este simulador usa el modelo de <b>regresión lineal</b> entrenado con datos reales "
            f"(MAE en prueba cronológica: <b>{mae_test:.3f} pp</b>, ganador vs. Random Forest y baseline). "
            f"Mueve los controles para ver cómo cambia la predicción del próximo trimestre.")

        ultimo = fore.iloc[-1]
        c1, c2 = st.columns([2, 1])
        with c1:
            st.markdown("#### Ajusta las variables de entrada")
            s_l1 = st.slider("Desocupación trimestre anterior (%)", 2.0, 6.0,
                            float(round(ultimo["tasa_desocupacion"], 2)), 0.05)
            s_l2 = st.slider("Desocupación 2 trimestres atrás (%)", 2.0, 6.0,
                            float(round(ultimo["lag_1"], 2)), 0.05)
            s_l4 = st.slider("Desocupación 1 año atrás (%)", 2.0, 6.0,
                            float(round(ultimo["lag_2"], 2)), 0.05)
            s_mm = st.slider("Promedio últimos 4 trimestres (%)", 2.0, 6.0,
                            float(round(ultimo["media_movil_4"], 2)), 0.05)
        with c2:
            X_pred = pd.DataFrame([[s_l1, s_l2, s_l4, s_mm]],
                                  columns=["lag_1","lag_2","lag_4","media_movil_4"])
            pred = float(reg_final.predict(X_pred)[0])
            st.markdown("#### Predicción del modelo")
            if pred < 3.2:
                color = COLOR_TEAL; nivel = "Estable"; icono = "✅"
            elif pred <= 4.2:
                color = COLOR_MUSTARD; nivel = "Alerta"; icono = "⚠️"
            else:
                color = COLOR_ACCENT; nivel = "Crítico"; icono = "🚨"
            st.markdown(
                f"<div style='background:white; border-radius:14px; padding:24px; text-align:center; "
                f"border-top:6px solid {color}; box-shadow:0 4px 16px rgba(17,52,123,0.08);'>"
                f"<div style='color:{COLOR_INK_SOFT}; font-size:13px; font-weight:600; text-transform:uppercase;'>"
                f"Tasa estimada</div>"
                f"<div style='font-size:44px; font-weight:800; color:{color}; line-height:1.1;'>{pred:.2f}%</div>"
                f"<div style='color:{color}; font-weight:700; font-size:15px; margin-top:8px;'>{icono} {nivel}</div>"
                f"</div>", unsafe_allow_html=True)

        # Proyección de 4 trimestres
        st.markdown("## Proyección: próximos 4 trimestres")
        F = ["lag_1","lag_2","lag_4","media_movil_4"]
        vals = list(fore["tasa_desocupacion"].values)
        per_names = ["2026/T2", "2026/T3", "2026/T4", "2027/T1"]
        proy_val = []
        for _ in per_names:
            x = pd.DataFrame([[vals[-1], vals[-2], vals[-4], np.mean(vals[-4:])]], columns=F)
            v = float(reg_final.predict(x)[0])
            proy_val.append(v); vals.append(v)

        fig, ax = plt.subplots(figsize=(13, 5))
        hist_n = min(24, len(fore))
        hist = fore.iloc[-hist_n:].reset_index(drop=True)
        ax.plot(range(hist_n), hist["tasa_desocupacion"], color=COLOR_INDIGO,
                linewidth=2.2, marker="o", markersize=5, label="Histórico")
        ax.plot(range(hist_n-1, hist_n+len(proy_val)),
                [hist["tasa_desocupacion"].iloc[-1]] + proy_val,
                color=COLOR_ACCENT, linewidth=2.2, marker="s", markersize=6,
                linestyle="--", label="Proyección")
        ax.axvspan(hist_n-0.5, hist_n+len(proy_val)-0.5, alpha=0.10, color=COLOR_ACCENT)
        etiquetas_x = list(hist["periodo"]) + per_names
        ax.set_xticks(range(len(etiquetas_x)))
        ax.set_xticklabels(etiquetas_x, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Tasa de desocupación (%)")
        ax.set_title("Serie histórica y proyección iterativa")
        ax.legend(frameon=False)
        estilo_ejes(ax)
        plt.tight_layout(); st.pyplot(fig)

        c1, c2, c3, c4 = st.columns(4)
        for col, (p, v) in zip([c1,c2,c3,c4], zip(per_names, proy_val)):
            kpi(col, p, f"{v:.2f}%", "accent")


# =============================================================================
# SECCIÓN 6: GUÍA INTERACTIVA ODS 8 (chat offline por palabras clave)
# =============================================================================
elif seccion == "💡 Guía Interactiva ODS 8":
    st.markdown("# Guía Interactiva del ODS 8")

    box("info",
        f"<b>Asistente conceptual.</b> Escribe tu pregunta y el asistente responderá "
        f"a partir del contenido del proyecto. Funciona sin conexión externa.")

    # Base de conocimiento offline (palabra clave → respuesta)
    def responder(consulta):
        q = consulta.lower()
        if any(k in q for k in ["ods 8", "ods8", "objetivo 8", "trabajo decente"]):
            return ("El **ODS 8** es el Objetivo de Desarrollo Sostenible 8 de la ONU: "
                    "*Trabajo decente y crecimiento económico*. Busca empleo pleno y "
                    "productivo con condiciones dignas: seguro médico, pensión, contrato y "
                    "seguridad. La meta 8.3 promueve la formalización y la 8.5 el empleo pleno.")
        if any(k in q for k in ["informalidad", "informal"]):
            return ("La **informalidad laboral** es trabajar sin seguridad social, sin contrato "
                    "ni prestaciones. En México supera el **50%** desde hace dos décadas. "
                    "Ejemplos: comercio ambulante, empleo sin IMSS, trabajo doméstico no registrado.")
        if any(k in q for k in ["desempleo", "desocup"]):
            return ("La **tasa de desempleo (o desocupación)** mide qué porcentaje de la población "
                    "que busca trabajo no lo encuentra. En México es de solo **2.55%** (2026/T1), "
                    "una de las más bajas de la OCDE. Pero no captura el trabajo decente.")
        if any(k in q for k in ["paradoja", "paradox"]):
            return ("La **paradoja laboral mexicana**: los estados con menos desempleo son los que "
                    "tienen más informalidad. La correlación es de **r = -0.598** (p = 0.0003, "
                    "estadísticamente significativa). Sin seguro de desempleo, la gente acepta "
                    "cualquier trabajo aunque sea informal, y deja de contar como desempleada.")
        if any(k in q for k in ["k-means", "kmeans", "cluster", "agrupamiento"]):
            return ("**K-Means** es un algoritmo de aprendizaje no supervisado que agrupa los 32 "
                    "estados en 3 perfiles según sus tasas de desocupación, informalidad y "
                    "participación:\n\n"
                    "- 🔵 **Norte/Centro formal** (16 estados): mercado más formal.\n"
                    "- 🔴 **Sur informal** (11 estados): la paradoja al extremo.\n"
                    "- 🟢 **Costero/turístico** (5 estados): alta participación laboral.")
        if any(k in q for k in ["regresión", "regresion", "predicción", "prediccion", "pronóstico", "pronostico", "modelo"]):
            return ("El proyecto usa **regresión lineal** con retardos (lag 1, 2, 4) y media móvil "
                    f"para pronosticar la desocupación. MAE en prueba cronológica: **{mae_test:.3f} pp**. "
                    "Le ganó al Random Forest, mostrando que con series cortas los modelos simples "
                    "generalizan mejor.")
        if any(k in q for k in ["enoe", "inegi"]):
            return ("La **ENOE** (Encuesta Nacional de Ocupación y Empleo) del **INEGI** es la fuente "
                    "oficial. Publica indicadores laborales cada trimestre para las 32 entidades. "
                    "El proyecto consume su API con un token y consolida 44,880 registros de 2005 a 2026.")
        if any(k in q for k in ["dataset", "datos", "cuántos", "cuantos"]):
            return ("El dataset consolidado tiene **44,880 registros**: 16 indicadores × 33 áreas "
                    "geográficas (32 estados + nacional) × 85 trimestres (2005/T1 a 2026/T1).")
        if any(k in q for k in ["arquitectura", "servidor", "nube", "vps", "contabo"]):
            return ("El pipeline corre en un **VPS Ubuntu (Contabo)** con JupyterLab. Las etapas son: "
                    "descarga API → EDA → features → clustering / predicción. Cada etapa es un "
                    "notebook independiente conectado por archivos CSV.")
        if any(k in q for k in ["género", "genero", "hombre", "mujer", "sexo"]):
            return ("El dataset incluye desgloses por sexo para desocupación, informalidad y ocupación. "
                    "La sección **Análisis por Estado** compara hombres y mujeres para cualquier "
                    "entidad y periodo.")
        if any(k in q for k in ["autor", "equipo", "quien"]):
            return ("El equipo del proyecto: **Alberto Saúl López Crespo**, **Jeshua Vera Flores** y "
                    "**Betel Zurisadai Hernández Sánchez**. Grupo IDIA 224, Universidad Politécnica "
                    "de Querétaro.")
        return ("No estoy seguro de esa consulta. Intenta con temas como: **ODS 8**, "
                "**informalidad**, **paradoja**, **K-Means**, **regresión**, **ENOE**, "
                "**dataset**, **arquitectura**, **género** o **equipo**.")

    if "chat" not in st.session_state:
        st.session_state.chat = [
            ("asistente",
             "👋 Hola. Puedo explicarte los conceptos del proyecto: **ODS 8, informalidad, "
             "paradoja laboral, K-Means, regresión, ENOE**… ¿Qué te gustaría saber?")
        ]

    for autor, msg in st.session_state.chat:
        if autor == "usuario":
            st.markdown(
                f"<div style='display:flex; justify-content:flex-end; margin:8px 0;'>"
                f"<div style='background:{COLOR_INDIGO}; color:white; padding:10px 16px; "
                f"border-radius:16px 16px 4px 16px; max-width:70%; box-shadow:0 2px 6px rgba(0,0,0,0.1);'>"
                f"{msg}</div></div>", unsafe_allow_html=True)
        else:
            st.markdown(
                f"<div style='display:flex; justify-content:flex-start; margin:8px 0;'>"
                f"<div style='background:white; color:{COLOR_INK}; padding:10px 16px; "
                f"border-radius:16px 16px 16px 4px; max-width:70%; box-shadow:0 2px 6px rgba(17,52,123,0.08); "
                f"border-left:4px solid {COLOR_TEAL};'>"
                f"<b style='color:{COLOR_TEAL};'>💡 Asistente</b><br>{msg}</div></div>",
                unsafe_allow_html=True)

    entrada = st.chat_input("Escribe tu pregunta aquí…")
    if entrada:
        st.session_state.chat.append(("usuario", entrada))
        st.session_state.chat.append(("asistente", responder(entrada)))
        st.rerun()

    st.markdown("---")
    st.markdown("**Sugerencias rápidas:**")
    ej_cols = st.columns(4)
    sugerencias = ["¿Qué es el ODS 8?", "¿Qué es la paradoja?", "¿Cómo funciona K-Means?", "¿Cuántos datos hay?"]
    for col, sug in zip(ej_cols, sugerencias):
        with col:
            if st.button(sug, use_container_width=True, key=f"sug_{sug}"):
                st.session_state.chat.append(("usuario", sug))
                st.session_state.chat.append(("asistente", responder(sug)))
                st.rerun()


# =============================================================================
# SECCIÓN 7: PREGUNTAS FRECUENTES
# =============================================================================
elif seccion == "❓ Preguntas Frecuentes":
    st.markdown("# Preguntas Frecuentes · Auditoría Académica")

    box("info",
        f"<b>Marco de sustentación teórica y respuestas metodológicas.</b><br>"
        f"Esta sección responde las preguntas típicas sobre el diseño y la validez del proyecto.")

    faqs = [
        ("1. ¿Cómo se alinea el proyecto con el ODS 8?",
         "El ODS 8 promueve el trabajo decente y el crecimiento económico. El proyecto lo aborda "
         "cuestionando si la baja tasa de desempleo de México refleja realmente empleo digno. Al "
         "demostrar que baja desocupación coexiste con alta informalidad, se muestra que la "
         "medición tradicional es insuficiente para monitorear la meta 8.3 (formalización) y la "
         "meta 8.5 (empleo pleno y productivo)."),
        ("2. ¿Cuáles son las fuentes de datos y cómo se capturan?",
         "La fuente es la **Encuesta Nacional de Ocupación y Empleo (ENOE)** del **INEGI**, "
         "consumida vía su API pública (Banco de Información Económica, BIE) con token personal. "
         "Se descargan 16 indicadores en formato JSON sobre HTTPS. Un mecanismo de contingencia "
         "local garantiza la disponibilidad si la API falla."),
        ("3. ¿Por qué K-Means y no otro algoritmo de agrupamiento?",
         "K-Means es apropiado porque: (a) segmenta observaciones por similitud en variables "
         "continuas (las tres tasas), (b) tiene interpretación clara mediante el centroide de "
         "cada grupo, y (c) es reproducible con semilla fija. Se estandarizaron las variables con "
         "**Z-score** porque las tres tasas viven en escalas muy distintas (desempleo 2–6% vs. "
         "informalidad 35–80%), y sin estandarizar la informalidad dominaría la distancia "
         "euclidiana. K=3 se validó con el método del codo y coeficiente de silueta (0.406)."),
        ("4. ¿Qué significan y cómo se interpretan los tres clusters?",
         "**Grupo 1 (Norte/Centro formal, 16 estados):** economías industriales con más empleo "
         "formal pero más gente activamente buscando trabajo. Ejemplo: Nuevo León, CDMX.\n\n"
         "**Grupo 2 (Sur informal, 11 estados):** la paradoja al extremo. Baja desocupación con "
         "informalidad superior al 70%. Ejemplo: Oaxaca, Chiapas.\n\n"
         "**Grupo 3 (Costero/turístico, 5 estados):** economías de turismo con la mayor "
         "participación laboral. Ejemplo: Quintana Roo, BCS."),
        ("5. ¿Bajo qué lógica opera el modelo predictivo de regresión lineal?",
         "El modelo predice la tasa de desocupación del siguiente trimestre a partir de: (a) el "
         "valor del trimestre anterior (lag 1), (b) el de 2 trimestres atrás (lag 2), (c) el del "
         "mismo trimestre del año pasado (lag 4, captura estacionalidad), y (d) el promedio móvil "
         "de los últimos 4 trimestres (captura tendencia). "
         f"MAE en prueba cronológica: **{mae_test:.3f} pp**, ganador sobre Random Forest (0.252 pp) "
         "y baseline ingenuo (0.213 pp). El split es cronológico estricto para evitar fuga de datos."),
        ("6. ¿Cómo se garantiza la reproducibilidad del análisis?",
         "Cada etapa está codificada en notebooks con dependencias fijas en un entorno virtual "
         "Python 3.12 sobre un VPS Ubuntu. Todo el pipeline (`descarga → EDA → features → "
         "modelado`) puede re-ejecutarse cuando el INEGI publique un nuevo trimestre. Todos los "
         "resultados del reporte, dashboard y póster se generan al vuelo desde los mismos datos."),
        ("7. ¿Cuáles son las limitaciones del estudio?",
         "**(a)** Serie corta: solo 81 trimestres útiles, lo que impide modelos muy complejos. "
         "**(b)** El modelo asume que patrones pasados persisten; no anticipa shocks como el 2020. "
         "**(c)** El clustering usa promedios históricos, ocultando dinámicas dentro de cada estado. "
         "**(d)** La tasa de informalidad mide el margen extensivo (cuántos son informales) pero no "
         "la intensidad (horas, ingresos, condiciones), por lo que la brecha real es mayor."),
    ]

    for pregunta, respuesta in faqs:
        with st.expander(pregunta):
            st.markdown(respuesta)

    st.markdown("<br>", unsafe_allow_html=True)
    box("hook",
        f"<b>Referencias bibliográficas clave:</b><br>"
        f"• Maloney, W. F. (2004). <i>Informality Revisited</i>. World Development.<br>"
        f"• Levy, S. (2008). <i>Good Intentions, Bad Outcomes</i>. Brookings Institution Press.<br>"
        f"• OIT (2018). <i>Women and Men in the Informal Economy</i>.<br>"
        f"• Hyndman & Athanasopoulos (2018). <i>Forecasting: Principles and Practice</i>.")
