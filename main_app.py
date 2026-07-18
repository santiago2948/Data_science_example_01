"""
Dashboard interactivo de COVID-19 (datos sintéticos) construido con Streamlit.

Este único archivo unifica la generación de datos sintéticos y el dashboard.
El objetivo es contar, mediante datos ficticios pero coherentes, la historia de
cómo la edad, las comorbilidades y la vacunación influyeron en la gravedad y el
desenlace de los pacientes durante la pandemia simulada (2020-2022).

Ejecución del dashboard:
    streamlit run main_app.py

Generación del CSV por consola (opcional):
    python main_app.py            # genera datos_covid.csv con 5.000 registros
    python main_app.py 10000      # genera 10.000 registros
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

CSV_PATH = "datos_covid.csv"

# --------------------------------------------------------------------------- #
# Catálogos usados para simular las variables cualitativas
# --------------------------------------------------------------------------- #
PAISES = {
    "Colombia": ["Bogotá", "Medellín", "Cali", "Barranquilla", "Cartagena"],
    "México": ["CDMX", "Guadalajara", "Monterrey", "Puebla"],
    "Argentina": ["Buenos Aires", "Córdoba", "Rosario"],
    "Perú": ["Lima", "Arequipa", "Trujillo"],
    "Chile": ["Santiago", "Valparaíso", "Concepción"],
}

SEXOS = ["Femenino", "Masculino"]
ESTADOS = ["Recuperado", "Activo", "Hospitalizado", "Fallecido"]
GRAVEDAD = ["Leve", "Moderado", "Grave", "Crítico"]
COMORBILIDADES = ["Ninguna", "Hipertensión", "Diabetes", "Obesidad", "Enf. respiratoria", "Cardiopatía"]
FUENTES_CONTAGIO = ["Comunitario", "Familiar", "Laboral", "Viaje", "Desconocido"]

# Paleta coherente para toda la app
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


# --------------------------------------------------------------------------- #
# Generación de datos sintéticos
# --------------------------------------------------------------------------- #
def _grupo_edad(edad: int) -> str:
    """Clasifica la edad en un rango etario legible."""
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
    """Genera un DataFrame con `n` registros sintéticos de COVID-19.

    La simulación incorpora relaciones realistas: la edad y las comorbilidades
    aumentan la gravedad y la probabilidad de fallecimiento, mientras que la
    vacunación las reduce. Así el dashboard cuenta una historia coherente.
    """
    rng = np.random.default_rng(seed)

    # --- Variables demográficas ---
    edades = rng.normal(loc=42, scale=18, size=n).clip(1, 98).astype(int)
    sexos = rng.choice(SEXOS, size=n, p=[0.51, 0.49])

    paises = rng.choice(list(PAISES.keys()), size=n, p=[0.35, 0.25, 0.15, 0.15, 0.10])
    ciudades = np.array([rng.choice(PAISES[p]) for p in paises])

    # --- Vacunación (probabilidad mayor en adultos) ---
    prob_vacuna = np.where(edades >= 18, 0.72, 0.35)
    vacunado = rng.random(n) < prob_vacuna
    dosis = np.where(vacunado, rng.integers(1, 4, size=n), 0)

    # --- Comorbilidades (más frecuentes con la edad) ---
    prob_comorbilidad = (edades / 130).clip(0.05, 0.8)
    tiene_comorbilidad = rng.random(n) < prob_comorbilidad
    comorbilidad = np.where(
        tiene_comorbilidad,
        rng.choice(COMORBILIDADES[1:], size=n),
        "Ninguna",
    )

    # --- Puntaje de riesgo -> define la gravedad ---
    riesgo = (
        (edades / 100) * 1.6
        + tiene_comorbilidad * 0.6
        - vacunado * 0.5
        + rng.normal(0, 0.25, size=n)
    )

    gravedad = np.select(
        [riesgo < 0.35, riesgo < 0.7, riesgo < 1.05],
        ["Leve", "Moderado", "Grave"],
        default="Crítico",
    )

    # --- Estado clínico dependiente de la gravedad ---
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

    # --- Variables clínicas cuantitativas ---
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
    # Los fallecidos no registran días de recuperación
    dias_recuperacion = np.where(estado == "Fallecido", 0, dias_recuperacion)

    # --- Fechas de diagnóstico distribuidas en ~2 años de pandemia ---
    dia_offset = rng.integers(0, 730, size=n)
    fechas = pd.to_datetime("2020-03-01") + pd.to_timedelta(dia_offset, unit="D")

    fuente = rng.choice(FUENTES_CONTAGIO, size=n, p=[0.40, 0.25, 0.15, 0.08, 0.12])

    df = pd.DataFrame(
        {
            "id_paciente": np.arange(1, n + 1),
            "fecha_diagnostico": fechas,
            "pais": paises,
            "ciudad": ciudades,
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


# --------------------------------------------------------------------------- #
# Carga / generación de datos para el dashboard
# --------------------------------------------------------------------------- #
@st.cache_data
def cargar_datos(n: int, seed: int) -> pd.DataFrame:
    """Genera los datos sintéticos y los guarda en CSV (cacheados por n y seed)."""
    df = generar_datos_covid(n=n, seed=seed)
    df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")
    df["fecha_diagnostico"] = pd.to_datetime(df["fecha_diagnostico"])
    df["mes"] = df["fecha_diagnostico"].dt.to_period("M").dt.to_timestamp()
    return df


def ejecutar_dashboard() -> None:
    """Construye y renderiza el dashboard de Streamlit."""
    st.set_page_config(
        page_title="Dashboard COVID-19 | Datos Sintéticos",
        page_icon="🦠",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ----------------------------------------------------------------------- #
    # Barra lateral: simulación y filtros
    # ----------------------------------------------------------------------- #
    st.sidebar.title("⚙️ Panel de control")

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

    df = cargar_datos(n_registros, semilla)

    st.sidebar.subheader("2. Filtros interactivos")

    fecha_min = df["fecha_diagnostico"].min().date()
    fecha_max = df["fecha_diagnostico"].max().date()
    rango_fechas = st.sidebar.date_input(
        "Rango de fechas de diagnóstico",
        value=(fecha_min, fecha_max),
        min_value=fecha_min,
        max_value=fecha_max,
    )

    paises_sel = st.sidebar.multiselect(
        "País", options=sorted(df["pais"].unique()), default=sorted(df["pais"].unique())
    )
    sexo_sel = st.sidebar.multiselect(
        "Sexo", options=sorted(df["sexo"].unique()), default=sorted(df["sexo"].unique())
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

    # --- Aplicar filtros ---
    dff = df.copy()
    if isinstance(rango_fechas, (tuple, list)) and len(rango_fechas) == 2:
        ini, fin = pd.to_datetime(rango_fechas[0]), pd.to_datetime(rango_fechas[1])
        dff = dff[(dff["fecha_diagnostico"] >= ini) & (dff["fecha_diagnostico"] <= fin)]
    dff = dff[dff["pais"].isin(paises_sel)]
    dff = dff[dff["sexo"].isin(sexo_sel)]
    dff = dff[dff["gravedad"].isin(gravedad_sel)]
    if vacunado_sel != "Todos":
        dff = dff[dff["vacunado"] == vacunado_sel]

    # ----------------------------------------------------------------------- #
    # Encabezado y storytelling
    # ----------------------------------------------------------------------- #
    st.title("🦠 Dashboard COVID-19 — Simulación con datos sintéticos")
    st.markdown(
        """
Este tablero utiliza **datos totalmente ficticios** (generados dentro de la propia
plataforma) para recrear la evolución de una pandemia. La historia que buscamos
contar es sencilla: **la edad y las enfermedades previas empujan los casos hacia
cuadros más graves, mientras que la vacunación actúa como escudo protector.**

Usa el **panel de control** de la izquierda para simular más registros o filtrar
la población y observar cómo cambia cada indicador.
"""
    )

    if dff.empty:
        st.warning("No hay registros con los filtros seleccionados. Ajusta los filtros.")
        st.stop()

    # ----------------------------------------------------------------------- #
    # 1. MÉTRICAS CUANTITATIVAS (KPIs)
    # ----------------------------------------------------------------------- #
    st.header("📊 Métricas cuantitativas")
    st.caption("Indicadores numéricos clave de la población filtrada.")

    total = len(dff)
    fallecidos = int((dff["estado"] == "Fallecido").sum())
    recuperados = int((dff["estado"] == "Recuperado").sum())
    activos = int(dff["estado"].isin(["Activo", "Hospitalizado"]).sum())
    tasa_letalidad = fallecidos / total * 100
    pct_vacunados = (dff["vacunado"] == "Sí").mean() * 100
    pct_uci = (dff["uci"] == "Sí").mean() * 100
    edad_prom = dff["edad"].mean()
    sat_prom = dff["saturacion_oxigeno"].mean()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Casos totales", f"{total:,}")
    c2.metric("Recuperados", f"{recuperados:,}", f"{recuperados/total*100:.1f}%")
    c3.metric("Casos activos", f"{activos:,}", f"{activos/total*100:.1f}%")
    c4.metric("Fallecidos", f"{fallecidos:,}", f"-{tasa_letalidad:.1f}%", delta_color="inverse")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Tasa de letalidad", f"{tasa_letalidad:.2f}%")
    c6.metric("% Vacunados", f"{pct_vacunados:.1f}%")
    c7.metric("% Ingreso a UCI", f"{pct_uci:.1f}%")
    c8.metric("Edad promedio", f"{edad_prom:.0f} años")

    st.info(
        f"💡 **Lectura rápida:** de {total:,} casos simulados, la saturación de oxígeno "
        f"promedio es de **{sat_prom:.0f}%** y la edad media de los pacientes es de "
        f"**{edad_prom:.0f} años**. La letalidad se ubica en **{tasa_letalidad:.2f}%**."
    )

    # ----------------------------------------------------------------------- #
    # 2. MÉTRICAS CUALITATIVAS
    # ----------------------------------------------------------------------- #
    st.header("🏷️ Métricas cualitativas")
    st.caption("Categorías predominantes que describen el perfil de la población.")

    grupo_top = dff["grupo_edad"].mode()[0]
    comorb_top = dff.loc[dff["comorbilidad"] != "Ninguna", "comorbilidad"]
    comorb_top = comorb_top.mode()[0] if not comorb_top.empty else "Ninguna"
    fuente_top = dff["fuente_contagio"].mode()[0]
    gravedad_top = dff["gravedad"].mode()[0]
    ciudad_top = dff["ciudad"].mode()[0]

    q1, q2, q3, q4, q5 = st.columns(5)
    q1.metric("Grupo etario más afectado", grupo_top)
    q2.metric("Comorbilidad frecuente", comorb_top)
    q3.metric("Fuente de contagio principal", fuente_top)
    q4.metric("Gravedad predominante", gravedad_top)
    q5.metric("Ciudad con más casos", ciudad_top)

    # ----------------------------------------------------------------------- #
    # 3. GRÁFICAS DINÁMICAS (Plotly) + STORYTELLING
    # ----------------------------------------------------------------------- #
    st.header("📈 Gráficas dinámicas e historia de los datos")

    # 3.1 Evolución temporal
    st.subheader("La curva de la pandemia en el tiempo")
    serie = dff.groupby("mes").size().reset_index(name="casos")
    fig_line = px.area(
        serie,
        x="mes",
        y="casos",
        markers=True,
        labels={"mes": "Mes", "casos": "Casos diagnosticados"},
        color_discrete_sequence=["#1f77b4"],
    )
    fig_line.update_layout(hovermode="x unified", height=380)
    st.plotly_chart(fig_line, use_container_width=True)
    st.markdown(
        "**Storytelling:** cada punto es el número de casos diagnosticados por mes. "
        "Las subidas representan las *olas* de contagio y los valles los periodos de "
        "control. Filtra por país o vacunación para ver cómo cambia la curva."
    )

    col_a, col_b = st.columns(2)

    # 3.2 Distribución por gravedad (dona)
    with col_a:
        st.subheader("¿Qué tan graves fueron los casos?")
        grav = dff["gravedad"].value_counts().reindex(
            ["Leve", "Moderado", "Grave", "Crítico"]
        ).fillna(0).reset_index()
        grav.columns = ["gravedad", "casos"]
        fig_dona = px.pie(
            grav,
            names="gravedad",
            values="casos",
            hole=0.5,
            color="gravedad",
            color_discrete_map=COLOR_GRAVEDAD,
        )
        fig_dona.update_traces(textinfo="percent+label")
        fig_dona.update_layout(height=360, showlegend=False)
        st.plotly_chart(fig_dona, use_container_width=True)
        st.markdown(
            "**Storytelling:** la mayoría de los casos son *leves o moderados*, pero la "
            "porción roja (grave/crítico) es la que presiona al sistema de salud."
        )

    # 3.3 Estado clínico por grupo de edad (barras apiladas)
    with col_b:
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
            "**Storytelling:** a mayor edad, crece la proporción de hospitalizaciones y "
            "fallecimientos (barras azul y roja). Los jóvenes se recuperan casi siempre."
        )

    col_c, col_d = st.columns(2)

    # 3.4 Efecto de la vacunación (barras agrupadas)
    with col_c:
        st.subheader("La vacuna como escudo protector")
        vac = (
            dff.groupby(["vacunado", "gravedad"]).size().reset_index(name="casos")
        )
        # Normalizar a porcentaje dentro de cada grupo de vacunación
        vac["pct"] = vac.groupby("vacunado")["casos"].transform(lambda s: s / s.sum() * 100)
        fig_vac = px.bar(
            vac,
            x="vacunado",
            y="pct",
            color="gravedad",
            barmode="group",
            color_discrete_map=COLOR_GRAVEDAD,
            category_orders={"gravedad": ["Leve", "Moderado", "Grave", "Crítico"]},
            labels={"vacunado": "¿Vacunado?", "pct": "% de casos", "gravedad": "Gravedad"},
        )
        fig_vac.update_layout(height=360, legend_title_text="")
        st.plotly_chart(fig_vac, use_container_width=True)
        st.markdown(
            "**Storytelling:** entre los vacunados domina el color verde (leve). Entre los "
            "no vacunados aumenta la proporción de casos graves y críticos."
        )

    # 3.5 Relación edad vs saturación (dispersión)
    with col_d:
        st.subheader("Edad, oxígeno y gravedad")
        muestra = dff.sample(min(1500, len(dff)), random_state=1)
        fig_scatter = px.scatter(
            muestra,
            x="edad",
            y="saturacion_oxigeno",
            color="gravedad",
            color_discrete_map=COLOR_GRAVEDAD,
            opacity=0.6,
            category_orders={"gravedad": ["Leve", "Moderado", "Grave", "Crítico"]},
            labels={"edad": "Edad", "saturacion_oxigeno": "Saturación O₂ (%)"},
        )
        fig_scatter.update_layout(height=360, legend_title_text="")
        st.plotly_chart(fig_scatter, use_container_width=True)
        st.markdown(
            "**Storytelling:** cada punto es un paciente. Al aumentar la edad, la "
            "saturación de oxígeno tiende a bajar y aparecen más casos críticos (rojo)."
        )

    # 3.6 Mapa jerárquico (sunburst): país -> gravedad -> estado
    st.subheader("Del país al desenlace: vista jerárquica")
    fig_sun = px.sunburst(
        dff,
        path=["pais", "gravedad", "estado"],
        color="gravedad",
        color_discrete_map=COLOR_GRAVEDAD,
    )
    fig_sun.update_layout(height=480, margin=dict(t=20, l=0, r=0, b=0))
    st.plotly_chart(fig_sun, use_container_width=True)
    st.markdown(
        "**Storytelling:** haz clic en un país para profundizar. El anillo interno es el "
        "país, el intermedio la gravedad y el externo el desenlace clínico. Permite "
        "explorar de lo general a lo particular con un solo gráfico."
    )

    # ----------------------------------------------------------------------- #
    # 4. Datos y descarga
    # ----------------------------------------------------------------------- #
    st.header("🗂️ Datos simulados")
    st.caption(
        f"Mostrando {len(dff):,} registros filtrados de {len(df):,} generados. "
        "Los datos son sintéticos y no corresponden a personas reales."
    )
    st.dataframe(dff, use_container_width=True, height=320)

    csv_bytes = dff.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
    st.download_button(
        "⬇️ Descargar datos filtrados (CSV)",
        data=csv_bytes,
        file_name="datos_covid_filtrados.csv",
        mime="text/csv",
    )

    st.caption("Proyecto académico · Fundamentos de Ciencia de Datos · Datos 100% ficticios.")


# --------------------------------------------------------------------------- #
# Punto de entrada
# --------------------------------------------------------------------------- #
def _generar_csv_por_consola() -> None:
    """Genera el CSV desde la línea de comandos: python main_app.py [n]."""
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 5000
    datos = generar_datos_covid(n=n)
    datos.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")
    print(f"Archivo '{CSV_PATH}' generado con {len(datos):,} registros.")


# Uso: `python main_app.py csv [n]` genera solo el CSV.
# Con `streamlit run main_app.py` se ejecuta el dashboard completo.
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "csv":
        _generar_csv_por_consola()
    else:
        ejecutar_dashboard()
