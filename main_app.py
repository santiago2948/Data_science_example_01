"""
Dashboard interactivo de COVID-19 (datos sintéticos) — EAFIT 2026.

Unifica generación de datos sintéticos y el tablero Streamlit.
Ejercicio del curso de Ciencia de Datos · Universidad EAFIT (2026).

Clave de acceso al dashboard: 1234

Ejecución:
    streamlit run main_app.py

CSV por consola:
    python main_app.py csv
    python main_app.py csv 10000
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

CSV_PATH = "datos_covid.csv"
CLAVE_ACCESO = "1234"

# --------------------------------------------------------------------------- #
# Catálogos
# --------------------------------------------------------------------------- #
REGIONES = {
    "Andina": {
        "ciudades": ["Bogotá", "Medellín", "Cali", "Bucaramanga"],
        "poblacion_base": 2_800_000,
        "temp_media": 18.0,
        "precip_media": 95.0,
        "natalidad_base": 14.5,
        "mortalidad_base": 6.2,
    },
    "Caribe": {
        "ciudades": ["Barranquilla", "Cartagena", "Santa Marta"],
        "poblacion_base": 1_200_000,
        "temp_media": 28.5,
        "precip_media": 70.0,
        "natalidad_base": 16.8,
        "mortalidad_base": 5.8,
    },
    "Pacífico": {
        "ciudades": ["Buenaventura", "Quibdó", "Tumaco"],
        "poblacion_base": 450_000,
        "temp_media": 26.0,
        "precip_media": 280.0,
        "natalidad_base": 18.2,
        "mortalidad_base": 7.1,
    },
    "Orinoquía": {
        "ciudades": ["Villavicencio", "Yopal", "Arauca"],
        "poblacion_base": 380_000,
        "temp_media": 27.0,
        "precip_media": 140.0,
        "natalidad_base": 15.4,
        "mortalidad_base": 5.5,
    },
    "Amazonía": {
        "ciudades": ["Leticia", "Florencia", "Mocoa"],
        "poblacion_base": 220_000,
        "temp_media": 26.5,
        "precip_media": 320.0,
        "natalidad_base": 19.0,
        "mortalidad_base": 6.8,
    },
}

SEXOS = ["Femenino", "Masculino"]
ESTADOS = ["Recuperado", "Activo", "Hospitalizado", "Fallecido"]
COMORBILIDADES = [
    "Ninguna",
    "Hipertensión",
    "Diabetes",
    "Obesidad",
    "Enf. respiratoria",
    "Cardiopatía",
]
FUENTES_CONTAGIO = ["Comunitario", "Familiar", "Laboral", "Viaje", "Desconocido"]

COLOR_ESTADO = {
    "Recuperado": "#2ca02c",
    "Activo": "#ff7f0e",
    "Hospitalizado": "#1f77b4",
    "Fallecido": "#d62728",
}
COLOR_GRAVEDAD = {
    "Leve": "#8bc34a",
    "Moderado": "#ffc107",
    "Grave": "#ff9800",
    "Crítico": "#e53935",
}
COLOR_REGION = {
    "Andina": "#1565c0",
    "Caribe": "#00897b",
    "Pacífico": "#6a1b9a",
    "Orinoquía": "#ef6c00",
    "Amazonía": "#2e7d32",
}


def _grupo_edad(edad: int) -> str:
    if edad < 18:
        return "0-17"
    if edad < 30:
        return "18-29"
    if edad < 45:
        return "30-44"
    if edad < 60:
        return "45-59"
    if edad < 75:
        return "60-74"
    return "75+"


def generar_datos_covid(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Genera n registros sintéticos con variables demográficas, climáticas y clínicas."""
    rng = np.random.default_rng(seed)

    regiones = rng.choice(
        list(REGIONES.keys()),
        size=n,
        p=[0.40, 0.22, 0.14, 0.14, 0.10],
    )
    ciudades = np.array([rng.choice(REGIONES[r]["ciudades"]) for r in regiones])

    # Población del municipio/ciudad (miles → personas)
    poblacion = np.array(
        [
            int(REGIONES[r]["poblacion_base"] * rng.uniform(0.55, 1.45))
            for r in regiones
        ]
    )

    # Clima: temperatura (°C) y precipitaciones (mm/mes) con ruido por región
    temperatura = np.array(
        [
            REGIONES[r]["temp_media"] + rng.normal(0, 1.8)
            for r in regiones
        ]
    ).clip(8, 38).round(1)

    precipitaciones = np.array(
        [
            max(0.0, REGIONES[r]["precip_media"] + rng.normal(0, 35))
            for r in regiones
        ]
    ).round(1)

    # Índices demográficos (por mil habitantes) con ligera correlación regional
    indice_natalidad = np.array(
        [
            REGIONES[r]["natalidad_base"] + rng.normal(0, 1.2)
            for r in regiones
        ]
    ).clip(8, 30).round(2)

    indice_mortalidad = np.array(
        [
            REGIONES[r]["mortalidad_base"] + rng.normal(0, 0.8)
            for r in regiones
        ]
    ).clip(3, 15).round(2)

    # Edad del paciente
    edades = rng.normal(loc=42, scale=18, size=n).clip(1, 98).astype(int)
    sexos = rng.choice(SEXOS, size=n, p=[0.51, 0.49])

    # Vacunación
    prob_vacuna = np.where(edades >= 18, 0.72, 0.35)
    vacunado = rng.random(n) < prob_vacuna
    dosis = np.where(vacunado, rng.integers(1, 4, size=n), 0)

    # Comorbilidades
    prob_comorbilidad = (edades / 130).clip(0.05, 0.8)
    tiene_comorbilidad = rng.random(n) < prob_comorbilidad
    comorbilidad = np.where(
        tiene_comorbilidad,
        rng.choice(COMORBILIDADES[1:], size=n),
        "Ninguna",
    )

    # Riesgo: edad + comorbilidad + clima extremo + mortalidad regional − vacuna
    temp_extrema = (np.abs(temperatura - 22) / 20).clip(0, 1)
    precip_alta = (precipitaciones / 400).clip(0, 1)
    riesgo = (
        (edades / 100) * 1.5
        + tiene_comorbilidad * 0.55
        - vacunado * 0.5
        + temp_extrema * 0.25
        + precip_alta * 0.15
        + (indice_mortalidad / 20) * 0.3
        + rng.normal(0, 0.22, size=n)
    )

    gravedad = np.select(
        [riesgo < 0.35, riesgo < 0.7, riesgo < 1.05],
        ["Leve", "Moderado", "Grave"],
        default="Crítico",
    )

    estado = np.empty(n, dtype=object)
    mapa_prob_estado = {
        "Leve": [0.90, 0.08, 0.02, 0.00],
        "Moderado": [0.78, 0.10, 0.10, 0.02],
        "Grave": [0.55, 0.08, 0.27, 0.10],
        "Crítico": [0.30, 0.05, 0.35, 0.30],
    }
    for g, probs in mapa_prob_estado.items():
        mask = gravedad == g
        estado[mask] = rng.choice(ESTADOS, size=mask.sum(), p=probs)

    uci = np.where(
        np.isin(gravedad, ["Grave", "Crítico"]),
        rng.random(n) < 0.6,
        rng.random(n) < 0.05,
    )

    saturacion = np.select(
        [gravedad == "Leve", gravedad == "Moderado", gravedad == "Grave"],
        [
            rng.normal(97, 1.2, n),
            rng.normal(94, 1.8, n),
            rng.normal(89, 2.5, n),
        ],
        default=rng.normal(83, 3.5, n),
    ).clip(60, 100).round(0)

    dias_recuperacion = np.select(
        [gravedad == "Leve", gravedad == "Moderado", gravedad == "Grave"],
        [
            rng.normal(8, 3, n),
            rng.normal(15, 4, n),
            rng.normal(24, 6, n),
        ],
        default=rng.normal(35, 9, n),
    ).clip(2, 90).astype(int)
    dias_recuperacion = np.where(estado == "Fallecido", 0, dias_recuperacion)

    # Serie de tiempo: fechas a lo largo de ~2 años
    dia_offset = rng.integers(0, 730, size=n)
    fechas = pd.to_datetime("2020-03-01") + pd.to_timedelta(dia_offset, unit="D")

    fuente = rng.choice(FUENTES_CONTAGIO, size=n, p=[0.40, 0.25, 0.15, 0.08, 0.12])

    df = pd.DataFrame(
        {
            "id_paciente": np.arange(1, n + 1),
            "fecha_diagnostico": fechas,
            "region": regiones,
            "ciudad": ciudades,
            "poblacion": poblacion,
            "temperatura": temperatura,
            "precipitaciones": precipitaciones,
            "indice_mortalidad": indice_mortalidad,
            "indice_natalidad": indice_natalidad,
            "edad": edades,
            "grupo_edad": [_grupo_edad(e) for e in edades],
            "sexo": sexos,
            "vacunado": np.where(vacunado, "Sí", "No"),
            "dosis": dosis,
            "comorbilidad": comorbilidad,
            "gravedad": gravedad,
            "estado": estado,
            "uci": np.where(uci, "Sí", "No"),
            "saturacion_oxigeno": saturacion.astype(int),
            "dias_recuperacion": dias_recuperacion,
            "fuente_contagio": fuente,
        }
    )

    return df.sort_values("fecha_diagnostico").reset_index(drop=True)


@st.cache_data
def cargar_datos(n: int, seed: int) -> pd.DataFrame:
    df = generar_datos_covid(n=n, seed=seed)
    df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")
    df["fecha_diagnostico"] = pd.to_datetime(df["fecha_diagnostico"])
    df["mes"] = df["fecha_diagnostico"].dt.to_period("M").dt.to_timestamp()
    return df


def _inyectar_estilos() -> None:
    st.markdown(
        """
        <style>
        .eafit-hero {
            background: linear-gradient(135deg, #0b3d2e 0%, #1b6b4a 45%, #0d47a1 100%);
            border-radius: 18px;
            padding: 2.2rem 2.4rem;
            color: #f7faf8;
            margin-bottom: 1.4rem;
            box-shadow: 0 12px 28px rgba(11, 61, 46, 0.28);
        }
        .eafit-hero .badge {
            display: inline-block;
            background: rgba(255,255,255,0.16);
            border: 1px solid rgba(255,255,255,0.28);
            padding: 0.25rem 0.75rem;
            border-radius: 999px;
            font-size: 0.85rem;
            letter-spacing: 0.04em;
            margin-bottom: 0.85rem;
        }
        .eafit-hero h1 {
            margin: 0 0 0.55rem 0;
            font-size: 2.05rem;
            line-height: 1.15;
            font-weight: 700;
        }
        .eafit-hero p {
            margin: 0;
            max-width: 54rem;
            font-size: 1.05rem;
            opacity: 0.95;
        }
        .eafit-hero .meta {
            margin-top: 1rem;
            font-size: 0.92rem;
            opacity: 0.88;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _hero_eafit() -> None:
    st.markdown(
        """
        <div class="eafit-hero">
          <div class="badge">Universidad EAFIT · Ciencia de Datos · 2026</div>
          <h1>Dashboard COVID-19 · Ejercicio académico</h1>
          <p>
            Este tablero forma parte de un <strong>ejercicio del curso de Ciencia de Datos
            EAFIT 2026</strong>. Explora datos sintéticos con variables demográficas,
            climáticas y clínicas, interactúa con filtros y series de tiempo, y cuenta
            la historia de cómo el contexto territorial y la edad se relacionan con
            la gravedad de los casos.
          </p>
          <div class="meta">
            Datos 100% ficticios · Acceso con clave de operación · Streamlit + Plotly
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _pantalla_acceso() -> bool:
    """Solicita la clave 1234. Devuelve True si el usuario ya autenticó."""
    if st.session_state.get("autenticado"):
        return True

    _inyectar_estilos()
    _hero_eafit()

    st.subheader("🔐 Acceso al dashboard")
    st.write(
        "Ingresa la **clave de operación** del ejercicio para explorar el tablero interactivo."
    )

    col1, col2, col3 = st.columns([1, 1.2, 1])
    with col2:
        with st.form("form_acceso", clear_on_submit=False):
            clave = st.text_input(
                "Clave de acceso",
                type="password",
                placeholder="Ingresa la clave…",
                help="Clave del ejercicio académico.",
            )
            enviar = st.form_submit_button("Entrar al dashboard", use_container_width=True)

        if enviar:
            if clave.strip() == CLAVE_ACCESO:
                st.session_state["autenticado"] = True
                st.rerun()
            else:
                st.error("Clave incorrecta. Intenta de nuevo.")

        st.caption("Pista del ejercicio: la clave son cuatro dígitos consecutivos empezando en 1.")

    return False


def ejecutar_dashboard() -> None:
    st.set_page_config(
        page_title="EAFIT 2026 · Dashboard COVID-19",
        page_icon="🎓",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    if not _pantalla_acceso():
        st.stop()

    _inyectar_estilos()
    _hero_eafit()

    # ----------------------------------------------------------------------- #
    # Panel lateral interactivo
    # ----------------------------------------------------------------------- #
    st.sidebar.title("⚙️ Panel de control")
    if st.sidebar.button("Cerrar sesión"):
        st.session_state["autenticado"] = False
        st.rerun()

    st.sidebar.subheader("1. Simulación de datos")
    n_registros = st.sidebar.slider(
        "Número de registros a simular",
        min_value=1000,
        max_value=15000,
        value=5000,
        step=500,
        help="Los datos se generan al vuelo dentro de la plataforma.",
    )
    semilla = st.sidebar.number_input(
        "Semilla aleatoria",
        min_value=0,
        max_value=9999,
        value=42,
        help="Cambia la semilla para obtener un escenario distinto.",
    )

    df = cargar_datos(int(n_registros), int(semilla))

    st.sidebar.subheader("2. Filtros interactivos")

    fecha_min = df["fecha_diagnostico"].min().date()
    fecha_max = df["fecha_diagnostico"].max().date()
    rango_fechas = st.sidebar.date_input(
        "Rango de fechas",
        value=(fecha_min, fecha_max),
        min_value=fecha_min,
        max_value=fecha_max,
    )

    regiones_sel = st.sidebar.multiselect(
        "Región",
        options=sorted(df["region"].unique()),
        default=sorted(df["region"].unique()),
    )
    sexo_sel = st.sidebar.multiselect(
        "Sexo",
        options=sorted(df["sexo"].unique()),
        default=sorted(df["sexo"].unique()),
    )
    gravedad_sel = st.sidebar.multiselect(
        "Gravedad",
        options=["Leve", "Moderado", "Grave", "Crítico"],
        default=["Leve", "Moderado", "Grave", "Crítico"],
    )
    vacunado_sel = st.sidebar.radio(
        "Estado de vacunación",
        options=["Todos", "Sí", "No"],
        horizontal=True,
    )

    edad_min, edad_max = st.sidebar.slider(
        "Rango de edad",
        min_value=int(df["edad"].min()),
        max_value=int(df["edad"].max()),
        value=(int(df["edad"].min()), int(df["edad"].max())),
    )
    temp_min, temp_max = st.sidebar.slider(
        "Temperatura (°C)",
        min_value=float(df["temperatura"].min()),
        max_value=float(df["temperatura"].max()),
        value=(float(df["temperatura"].min()), float(df["temperatura"].max())),
        step=0.5,
    )

    st.sidebar.subheader("3. Serie de tiempo")
    metrica_serie = st.sidebar.selectbox(
        "Métrica de la serie temporal",
        options=[
            "Casos diagnosticados",
            "Índice de mortalidad (promedio)",
            "Índice de natalidad (promedio)",
            "Temperatura (promedio)",
            "Precipitaciones (promedio)",
        ],
    )
    agrupar_region = st.sidebar.checkbox("Desglosar serie por región", value=False)

    # --- Aplicar filtros ---
    dff = df.copy()
    if isinstance(rango_fechas, (tuple, list)) and len(rango_fechas) == 2:
        ini, fin = pd.to_datetime(rango_fechas[0]), pd.to_datetime(rango_fechas[1])
        dff = dff[(dff["fecha_diagnostico"] >= ini) & (dff["fecha_diagnostico"] <= fin)]
    dff = dff[dff["region"].isin(regiones_sel)]
    dff = dff[dff["sexo"].isin(sexo_sel)]
    dff = dff[dff["gravedad"].isin(gravedad_sel)]
    dff = dff[(dff["edad"] >= edad_min) & (dff["edad"] <= edad_max)]
    dff = dff[(dff["temperatura"] >= temp_min) & (dff["temperatura"] <= temp_max)]
    if vacunado_sel != "Todos":
        dff = dff[dff["vacunado"] == vacunado_sel]

    st.markdown(
        """
Interactúa con el **panel de control** (izquierda): simula más registros, filtra por
región, edad, temperatura y vacunación, y elige qué variable ver en la **serie de tiempo**.
La historia que contamos: **el contexto territorial (clima, población, índices demográficos)
y la edad se relacionan con la gravedad de los casos**, mientras la vacunación protege.
"""
    )

    if dff.empty:
        st.warning("No hay registros con los filtros seleccionados. Ajusta los filtros.")
        st.stop()

    # ----------------------------------------------------------------------- #
    # Métricas cuantitativas
    # ----------------------------------------------------------------------- #
    st.header("📊 Métricas cuantitativas")
    st.caption("Indicadores numéricos de la población filtrada.")

    total = len(dff)
    fallecidos = int((dff["estado"] == "Fallecido").sum())
    recuperados = int((dff["estado"] == "Recuperado").sum())
    activos = int(dff["estado"].isin(["Activo", "Hospitalizado"]).sum())
    tasa_letalidad = fallecidos / total * 100
    edad_prom = dff["edad"].mean()
    poblacion_prom = dff["poblacion"].mean()
    temp_prom = dff["temperatura"].mean()
    precip_prom = dff["precipitaciones"].mean()
    mort_prom = dff["indice_mortalidad"].mean()
    nat_prom = dff["indice_natalidad"].mean()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Casos totales", f"{total:,}")
    c2.metric("Recuperados", f"{recuperados:,}", f"{recuperados / total * 100:.1f}%")
    c3.metric("Casos activos", f"{activos:,}", f"{activos / total * 100:.1f}%")
    c4.metric(
        "Fallecidos",
        f"{fallecidos:,}",
        f"-{tasa_letalidad:.1f}%",
        delta_color="inverse",
    )

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Edad promedio", f"{edad_prom:.0f} años")
    c6.metric("Población promedio", f"{poblacion_prom:,.0f}")
    c7.metric("Índice mortalidad", f"{mort_prom:.2f} ‰")
    c8.metric("Índice natalidad", f"{nat_prom:.2f} ‰")

    c9, c10, c11, c12 = st.columns(4)
    c9.metric("Temperatura media", f"{temp_prom:.1f} °C")
    c10.metric("Precipitación media", f"{precip_prom:.1f} mm")
    c11.metric("Tasa de letalidad", f"{tasa_letalidad:.2f}%")
    c12.metric("% Vacunados", f"{(dff['vacunado'] == 'Sí').mean() * 100:.1f}%")

    st.info(
        f"💡 En el filtro actual, la edad media es **{edad_prom:.0f} años**, "
        f"la temperatura promedio **{temp_prom:.1f} °C** y el índice de mortalidad "
        f"**{mort_prom:.2f} por mil**. Usa la serie de tiempo para ver cómo evolucionan "
        f"estas variables mes a mes."
    )

    # ----------------------------------------------------------------------- #
    # Métricas cualitativas
    # ----------------------------------------------------------------------- #
    st.header("🏷️ Métricas cualitativas")
    st.caption("Categorías predominantes del perfil filtrado.")

    region_top = dff["region"].mode()[0]
    grupo_top = dff["grupo_edad"].mode()[0]
    comorb_top = dff.loc[dff["comorbilidad"] != "Ninguna", "comorbilidad"]
    comorb_top = comorb_top.mode()[0] if not comorb_top.empty else "Ninguna"
    gravedad_top = dff["gravedad"].mode()[0]
    ciudad_top = dff["ciudad"].mode()[0]

    q1, q2, q3, q4, q5 = st.columns(5)
    q1.metric("Región con más casos", region_top)
    q2.metric("Grupo etario dominante", grupo_top)
    q3.metric("Comorbilidad frecuente", comorb_top)
    q4.metric("Gravedad predominante", gravedad_top)
    q5.metric("Ciudad con más casos", ciudad_top)

    # ----------------------------------------------------------------------- #
    # Serie de tiempo (interactiva)
    # ----------------------------------------------------------------------- #
    st.header("⏱️ Serie de tiempo interactiva")
    st.caption(
        "Elige la métrica en el panel lateral. Esta serie simula la evolución mensual "
        "de la pandemia y del contexto territorial."
    )

    mapa_agregacion = {
        "Casos diagnosticados": ("count", None),
        "Índice de mortalidad (promedio)": ("mean", "indice_mortalidad"),
        "Índice de natalidad (promedio)": ("mean", "indice_natalidad"),
        "Temperatura (promedio)": ("mean", "temperatura"),
        "Precipitaciones (promedio)": ("mean", "precipitaciones"),
    }
    tipo_agg, col_agg = mapa_agregacion[metrica_serie]

    if agrupar_region:
        if tipo_agg == "count":
            serie = (
                dff.groupby(["mes", "region"]).size().reset_index(name="valor")
            )
        else:
            serie = (
                dff.groupby(["mes", "region"])[col_agg]
                .mean()
                .reset_index(name="valor")
            )
        fig_ts = px.line(
            serie,
            x="mes",
            y="valor",
            color="region",
            markers=True,
            color_discrete_map=COLOR_REGION,
            labels={"mes": "Mes", "valor": metrica_serie, "region": "Región"},
        )
    else:
        if tipo_agg == "count":
            serie = dff.groupby("mes").size().reset_index(name="valor")
        else:
            serie = (
                dff.groupby("mes")[col_agg].mean().reset_index(name="valor")
            )
        fig_ts = px.area(
            serie,
            x="mes",
            y="valor",
            markers=True,
            labels={"mes": "Mes", "valor": metrica_serie},
            color_discrete_sequence=["#1565c0"],
        )

    fig_ts.update_layout(hovermode="x unified", height=420)
    st.plotly_chart(fig_ts, use_container_width=True)
    st.markdown(
        f"**Storytelling:** la serie muestra **{metrica_serie.lower()}** a lo largo "
        "del tiempo. Las subidas y bajadas representan olas de contagio o cambios "
        "estacionales del clima y de los índices demográficos. Combínala con filtros "
        "de región o edad para comparar escenarios."
    )

    # Serie dual: casos vs índice de mortalidad (siempre visible como apoyo)
    st.subheader("Casos vs índice de mortalidad en el tiempo")
    dual = (
        dff.groupby("mes")
        .agg(casos=("id_paciente", "count"), mortalidad=("indice_mortalidad", "mean"))
        .reset_index()
    )
    fig_dual = make_subplots(specs=[[{"secondary_y": True}]])
    fig_dual.add_trace(
        go.Scatter(
            x=dual["mes"],
            y=dual["casos"],
            name="Casos",
            mode="lines+markers",
            line=dict(color="#1565c0", width=2.5),
            fill="tozeroy",
        ),
        secondary_y=False,
    )
    fig_dual.add_trace(
        go.Scatter(
            x=dual["mes"],
            y=dual["mortalidad"],
            name="Índice mortalidad (‰)",
            mode="lines+markers",
            line=dict(color="#c62828", width=2.5, dash="dot"),
        ),
        secondary_y=True,
    )
    fig_dual.update_layout(height=400, hovermode="x unified", legend_title_text="")
    fig_dual.update_yaxes(title_text="Casos diagnosticados", secondary_y=False)
    fig_dual.update_yaxes(title_text="Índice de mortalidad (‰)", secondary_y=True)
    st.plotly_chart(fig_dual, use_container_width=True)
    st.markdown(
        "**Storytelling:** el eje izquierdo son los casos; el derecho, el índice de "
        "mortalidad promedio del mes. Comparar ambas curvas ayuda a ver si los picos "
        "de contagio coinciden con contextos de mayor mortalidad territorial."
    )

    # ----------------------------------------------------------------------- #
    # Más gráficas
    # ----------------------------------------------------------------------- #
    st.header("📈 Más gráficas e historia de los datos")

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Casos por región")
        por_region = dff["region"].value_counts().reset_index()
        por_region.columns = ["region", "casos"]
        fig_reg = px.bar(
            por_region,
            x="region",
            y="casos",
            color="region",
            color_discrete_map=COLOR_REGION,
            labels={"region": "Región", "casos": "Casos"},
        )
        fig_reg.update_layout(height=360, showlegend=False)
        st.plotly_chart(fig_reg, use_container_width=True)
        st.markdown(
            "**Storytelling:** no todas las regiones aportan el mismo volumen de casos. "
            "La región Andina suele concentrar más registros por su mayor población."
        )

    with col_b:
        st.subheader("Temperatura vs precipitaciones")
        muestra = dff.sample(min(1500, len(dff)), random_state=1)
        fig_clima = px.scatter(
            muestra,
            x="temperatura",
            y="precipitaciones",
            color="region",
            size="poblacion",
            hover_data=["ciudad", "edad", "gravedad"],
            color_discrete_map=COLOR_REGION,
            labels={
                "temperatura": "Temperatura (°C)",
                "precipitaciones": "Precipitaciones (mm)",
                "region": "Región",
            },
            opacity=0.65,
        )
        fig_clima.update_layout(height=360)
        st.plotly_chart(fig_clima, use_container_width=True)
        st.markdown(
            "**Storytelling:** cada punto es un paciente. El tamaño refleja la "
            "población de su ciudad. El Caribe es más cálido y seco; el Pacífico "
            "y la Amazonía concentran más lluvia."
        )

    col_c, col_d = st.columns(2)

    with col_c:
        st.subheader("El desenlace cambia con la edad")
        edad_estado = (
            dff.groupby(["grupo_edad", "estado"]).size().reset_index(name="casos")
        )
        orden_edad = ["0-17", "18-29", "30-44", "45-59", "60-74", "75+"]
        fig_bar = px.bar(
            edad_estado,
            x="grupo_edad",
            y="casos",
            color="estado",
            category_orders={"grupo_edad": orden_edad},
            color_discrete_map=COLOR_ESTADO,
            labels={"grupo_edad": "Grupo de edad", "casos": "Casos", "estado": "Estado"},
        )
        fig_bar.update_layout(height=360, barmode="stack", legend_title_text="")
        st.plotly_chart(fig_bar, use_container_width=True)
        st.markdown(
            "**Storytelling:** a mayor edad aumentan hospitalizaciones y fallecimientos. "
            "Los grupos jóvenes se recuperan con más frecuencia."
        )

    with col_d:
        st.subheader("Natalidad y mortalidad por región")
        dem = (
            dff.groupby("region")[["indice_natalidad", "indice_mortalidad"]]
            .mean()
            .reset_index()
            .melt(id_vars="region", var_name="indice", value_name="valor")
        )
        dem["indice"] = dem["indice"].map(
            {
                "indice_natalidad": "Natalidad",
                "indice_mortalidad": "Mortalidad",
            }
        )
        fig_dem = px.bar(
            dem,
            x="region",
            y="valor",
            color="indice",
            barmode="group",
            labels={"region": "Región", "valor": "Índice (‰)", "indice": "Índice"},
            color_discrete_sequence=["#00897b", "#c62828"],
        )
        fig_dem.update_layout(height=360)
        st.plotly_chart(fig_dem, use_container_width=True)
        st.markdown(
            "**Storytelling:** compara natalidad y mortalidad promedio entre regiones. "
            "Ayuda a entender el contexto demográfico detrás de los casos clínicos."
        )

    st.subheader("Del territorio al desenlace (vista jerárquica)")
    fig_sun = px.sunburst(
        dff,
        path=["region", "gravedad", "estado"],
        color="gravedad",
        color_discrete_map=COLOR_GRAVEDAD,
    )
    fig_sun.update_layout(height=480, margin=dict(t=20, l=0, r=0, b=0))
    st.plotly_chart(fig_sun, use_container_width=True)
    st.markdown(
        "**Storytelling:** haz clic en una región para profundizar. El anillo interno "
        "es la región, el intermedio la gravedad y el externo el desenlace clínico."
    )

    # ----------------------------------------------------------------------- #
    # Datos y descarga
    # ----------------------------------------------------------------------- #
    st.header("🗂️ Datos simulados")
    st.caption(
        f"Mostrando {len(dff):,} registros filtrados de {len(df):,} generados. "
        "Datos sintéticos · ejercicio EAFIT 2026."
    )
    st.dataframe(dff, use_container_width=True, height=320)

    csv_bytes = dff.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
    st.download_button(
        "⬇️ Descargar datos filtrados (CSV)",
        data=csv_bytes,
        file_name="datos_covid_filtrados.csv",
        mime="text/csv",
    )

    st.caption(
        "Universidad EAFIT · Curso de Ciencia de Datos 2026 · Ejercicio académico · Datos ficticios."
    )


def _generar_csv_por_consola() -> None:
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 5000
    datos = generar_datos_covid(n=n)
    datos.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")
    print(f"Archivo '{CSV_PATH}' generado con {len(datos):,} registros.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "csv":
        _generar_csv_por_consola()
    else:
        ejecutar_dashboard()
