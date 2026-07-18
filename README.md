# Dashboard COVID-19 · Ejercicio EAFIT 2026

Tablero interactivo de **Streamlit + Plotly** con datos sintéticos de COVID-19.
Forma parte de un **ejercicio del curso de Ciencia de Datos · Universidad EAFIT (2026)**.

> ⚠️ Todos los datos son **sintéticos**. No corresponden a personas ni casos reales.

**Clave de acceso al dashboard:** `1234`

## Historia que cuenta el dashboard

El contexto territorial (región, población, temperatura, precipitaciones) y los
índices demográficos (mortalidad y natalidad), junto con la **edad**, se relacionan
con la gravedad de los casos. La vacunación actúa como factor protector.

## Estructura

| Archivo | Descripción |
|---|---|
| `main_app.py` | Generación de datos + dashboard (archivo único). |
| `datos_covid.csv` | Se crea automáticamente al ejecutar. |
| `requirements.txt` | Dependencias. |

## Instalación y ejecución

```bash
pip install -r requirements.txt
streamlit run main_app.py
```

CSV opcional por consola:

```bash
python main_app.py csv
python main_app.py csv 10000
```

Al abrir la app, ingresa la clave **1234** para operar el tablero.

## Interacción del usuario

- Acceso con clave de operación.
- Simular entre 1.000 y 15.000 registros y cambiar la semilla.
- Filtrar por fechas, región, sexo, gravedad, vacunación, edad y temperatura.
- Elegir la métrica de la **serie de tiempo** y desglosarla por región.
- Explorar gráficas Plotly y descargar el CSV filtrado.

## Variables principales

`region`, `ciudad`, `poblacion`, `temperatura`, `precipitaciones`,
`indice_mortalidad`, `indice_natalidad`, `edad`, `grupo_edad`, `sexo`,
`vacunado`, `gravedad`, `estado`, `fecha_diagnostico`, entre otras.

---
Universidad EAFIT · Ciencia de Datos 2026 · Datos ficticios.
