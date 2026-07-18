# 🦠 Dashboard COVID-19 (Datos Sintéticos) — Streamlit

Tablero interactivo construido con **Streamlit + Plotly** que simula la evolución
de una pandemia usando **datos ficticios generados dentro de la propia plataforma**.
El objetivo es practicar la construcción de un dashboard de ciencia de datos con
métricas cuantitativas, cualitativas y gráficas con *storytelling*.

> ⚠️ Todos los datos son **sintéticos**: no corresponden a personas ni casos reales.

## 🎯 Historia que cuenta el dashboard

La edad y las enfermedades previas (comorbilidades) empujan los casos hacia
cuadros más graves, mientras que **la vacunación actúa como escudo protector**,
reduciendo la gravedad y la letalidad.

## 📦 Estructura del proyecto

| Archivo | Descripción |
|---|---|
| `main_app.py` | Archivo único: generación de datos sintéticos + dashboard de Streamlit. |
| `datos_covid.csv` | Dataset generado (se crea automáticamente al ejecutar). |
| `requirements.txt` | Dependencias del proyecto. |

## 🚀 Instalación y ejecución

1. (Opcional) Crea un entorno virtual:

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows PowerShell
```

2. Instala las dependencias:

```bash
pip install -r requirements.txt
```

3. (Opcional) Genera el CSV manualmente:

```bash
python main_app.py csv           # 5.000 registros por defecto
python main_app.py csv 10000     # 10.000 registros
```

4. Ejecuta el dashboard:

```bash
streamlit run main_app.py
```

## 🧭 Funcionalidades

- **Simulación interactiva:** ajusta el número de registros (1.000–15.000) y la
  semilla aleatoria desde el panel lateral.
- **Filtros:** rango de fechas, país, sexo, gravedad y estado de vacunación.
- **Métricas cuantitativas (KPIs):** casos totales, recuperados, activos,
  fallecidos, tasa de letalidad, % vacunados, % UCI, edad promedio.
- **Métricas cualitativas:** grupo etario más afectado, comorbilidad frecuente,
  fuente de contagio principal, gravedad predominante y ciudad con más casos.
- **Gráficas dinámicas (Plotly):** curva temporal, dona de gravedad, barras por
  edad y estado, efecto de la vacunación, dispersión edad/oxígeno y *sunburst*
  jerárquico país → gravedad → estado.
- **Descarga** de los datos filtrados en CSV.

## 🗃️ Diccionario de datos (variables simuladas)

`id_paciente`, `fecha_diagnostico`, `pais`, `ciudad`, `edad`, `grupo_edad`,
`sexo`, `vacunado`, `dosis`, `comorbilidad`, `gravedad`, `estado`, `uci`,
`saturacion_oxigeno`, `dias_recuperacion`, `fuente_contagio`.

---
Proyecto académico · Fundamentos de Ciencia de Datos.
