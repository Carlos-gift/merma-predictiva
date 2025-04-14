import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib

st.title("🚛 Control de Mermas en Transporte de Combustibles")

archivo = st.file_uploader("📂 Sube tu archivo CSV", type=["csv"])

if archivo is not None:
    df = pd.read_csv(archivo)

    st.subheader("📋 Datos cargados")
    st.dataframe(df)

    # Agrupar por ruta
    agrupado = df.groupby("Ruta")
    resumen = agrupado[["Volumen_Cargado_L", "Merma_L"]].sum()
    resumen["Viajes"] = agrupado.size()
    resumen["Porcentaje_Merma_Real"] = (resumen["Merma_L"] / resumen["Volumen_Cargado_L"]) * 100
    resumen = resumen.round(2)

    # Top 3 por ruta
    merma_ruta_camion = df.groupby(["Ruta", "Placa_Camion"])["Merma_L"].sum().reset_index()
    merma_ruta_camion = merma_ruta_camion.sort_values(by=["Ruta", "Merma_L"], ascending=[True, False])
    top3_por_ruta = merma_ruta_camion.groupby("Ruta").head(3)

    st.subheader("🚛 Top 3 camiones con mayor merma por ruta")
    st.dataframe(top3_por_ruta)

    # Clasificación semáforo
    def clasificar_estado(merma):
        if merma > 1.85:
            return "🔴 Alerta"
        elif merma > 1.70:
            return "🟡 Moderado"
        else:
            return "🟢 Controlado"

    resumen["Estado_Semaforo"] = resumen["Porcentaje_Merma_Real"].apply(clasificar_estado)

    st.subheader("🚦 Estado por ruta (Semáforo)")
    st.dataframe(resumen)

    st.success("Análisis completado. Revisa las rutas en estado rojo. 🚨")

    # Gráfico porcentaje de merma (semáforo)
    colores = {
        "🔴 Alerta": "red",
        "🟡 Moderado": "gold",
        "🟢 Controlado": "green"
    }

    colores_barras = resumen["Estado_Semaforo"].map(colores)

    fig, ax = plt.subplots(figsize=(8, 5))
    resumen["Porcentaje_Merma_Real"].plot(kind="bar", color=colores_barras, ax=ax)

    for i, v in enumerate(resumen["Porcentaje_Merma_Real"]):
        ax.text(i, v + 0.05, f'{v:.2f}%', ha='center', va='bottom', fontsize=9)

    ax.set_title("Porcentaje Real de Merma por Ruta (Semáforo)")
    ax.set_ylabel("Porcentaje de Merma (%)")
    ax.set_xlabel("Ruta")
    ax.set_ylim(0, resumen["Porcentaje_Merma_Real"].max() + 1)
    ax.set_xticks(range(len(resumen.index)))
    ax.set_xticklabels(resumen.index, rotation=0)
    st.pyplot(fig)

    # Gráfico litros perdidos por ruta
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    resumen["Merma_L"].plot(kind="bar", color="orange", ax=ax2)

    for i, v in enumerate(resumen["Merma_L"]):
        ax2.text(i, v + 200, f'{v:,.0f} L', ha='center', va='bottom', fontsize=9)

    ax2.set_title("Total de Litros Perdidos por Ruta")
    ax2.set_ylabel("Litros de Merma")
    ax2.set_xlabel("Ruta")
    ax2.set_ylim(0, resumen["Merma_L"].max() * 1.15)
    ax2.set_xticks(range(len(resumen.index)))
    ax2.set_xticklabels(resumen.index, rotation=0)
    st.pyplot(fig2)

    

    st.subheader("🧩 Predicción de Merma para Nuevas Condiciones")

    modelo_porcentaje = joblib.load("modelo_porcentaje_merma.pkl")
    modelo_litros = joblib.load("modelo_merma_litros.pkl")

    # Detectar tipo_camion y capacidad antes del formulario
    producto_input = st.selectbox("Tipo de producto", df["Producto"].unique())
    patente_input = st.selectbox("Patente del camión", df["Placa_Camion"].unique())

    tipo_camion = df[df["Placa_Camion"] == patente_input]["Tipo_Camion"].values[0]
    st.info(f"🚛 Tipo de camión detectado: **{tipo_camion.capitalize()}**")

    capacidad_maxima = {
        "pequeño": 15000,
        "mediano": 22000,
        "grande": 34000
    }
    litros_max = capacidad_maxima.get(tipo_camion, 50000)

    with st.form("form_prediccion"):
        volumen_input = st.number_input(
            f"Litros cargados (máx: {litros_max:,} L)", 
            min_value=500.0, 
            max_value=float(litros_max), 
            value=10000.0, 
            step=100.0
        )

        ruta = st.selectbox("Ruta", df["Ruta"].unique())
        duracion = st.number_input("Duración del viaje (horas)", min_value=0.0, step=0.1)
        distancia = st.number_input("Distancia del viaje (km)", min_value=0.0, step=1.0)
        clima = st.selectbox("Condición climática", df["Clima"].unique())

        submitted = st.form_submit_button("🔍 Predecir Merma")

        if submitted:
            entrada = pd.DataFrame([{
                "Ruta": ruta,
                "Duracion_Viaje_Horas": duracion,
                "Distancia_km": distancia,
                "Tipo_Camion": tipo_camion,
                "Clima": clima,
                "Producto": producto_input,
                "Volumen_Cargado_L": volumen_input,
            }])

            porcentaje_predicho = modelo_porcentaje.predict(entrada)[0]
            litros_predichos = modelo_litros.predict(entrada)[0]

            st.success(f"🔢 Porcentaje estimado de merma: {porcentaje_predicho:.2f}%")
            st.info(f"🛢️ Litros aproximados de pérdida: {litros_predichos:.2f} L")
