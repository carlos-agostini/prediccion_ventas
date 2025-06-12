import pandas as pd
import numpy as np
import statsmodels.api as sm
import streamlit as st
import plotly.graph_objects as go

# === Cargar y preparar datos ===
df = pd.read_csv('df_final_websales.zip', compression='zip')      # si usas zip
df['fecha_compra'] = pd.to_datetime(df['fecha_compra'], format='%d/%m/%Y %H:%M')
df['revenue'] = df['precio_unitario'] - df['descuentos']
df = df[df['fecha_compra'].dt.year <= 2024]
weekly_sales = df.set_index('fecha_compra').resample('W')['revenue'].sum().to_frame(name='sales')
start_date = weekly_sales.index.min().normalize()
end_date = weekly_sales.index.max().normalize()
all_weeks = pd.date_range(start=start_date, end=end_date, freq='W')
weekly_sales = weekly_sales.reindex(all_weeks).fillna(0)

# === UI Streamlit ===
st.title("📊 Predicción de Ventas para 2025")

with st.sidebar:
    st.header("⚙️ Parámetros de Simulación")
    st.markdown("---")
    
    inversion_incremento = st.slider("📈 Incremento de inversión publicitaria (%)", 0, 200, 109, 1)
    # Sidebar - presentación limpia con número verde
    st.markdown("**Inversión aumentada:** <span style='font-size:22px; color:lightgreen'><b>{}%</b></span>".format(inversion_incremento), unsafe_allow_html=True)
    st.markdown("**Año proyectado:** <span style='font-size:22px; color:lightgreen'><b>2025</b></span>", unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("📐 Distribución por trimestre (%)")
    q1_pct = st.number_input("Q1", value=25.0, min_value=0.0, max_value=100.0, step=0.1)
    q2_pct = st.number_input("Q2", value=18.33, min_value=0.0, max_value=100.0, step=0.1)
    q3_pct = st.number_input("Q3", value=18.67, min_value=0.0, max_value=100.0, step=0.1)
    q4_pct = st.number_input("Q4", value=38.0, min_value=0.0, max_value=100.0, step=0.1)
    
    suma_pct = q1_pct + q2_pct + q3_pct + q4_pct
    if suma_pct > 100:
        st.error(f"⚠️ La suma de los porcentajes por trimestre es {suma_pct:.2f}%. No debe superar el 100%.")
        st.stop()

# === Datos base ===
inversion_2024 = {1: 15971, 2: 93957, 3: 146673, 4: 165937}
total_2024 = sum(inversion_2024.values())

st.subheader("💰 Inversión publicitaria 2024 por Trimestre")
st.markdown(f"**Total 2024:** ${total_2024:,.0f}")
df_inversion = pd.DataFrame([list(inversion_2024.values())], columns=["Q1", "Q2", "Q3", "Q4"])
st.table(df_inversion.style.format("{:,.0f}").set_table_styles([
    {'selector': 'thead th', 'props': [('background-color', '#3e4e6e'), ('color', 'white'), ('font-weight', 'bold')]},
    {'selector': 'tbody td', 'props': [('text-align', 'right'), ('font-size', '14px')]}
]))

# === Preparar datos para modelado ===
weekly_sales['marketing'] = 0.0
for q, budget in inversion_2024.items():
    mask = (weekly_sales.index.year == 2024) & (weekly_sales.index.quarter == q)
    if mask.sum() > 0:
        weekly_sales.loc[mask, 'marketing'] = budget / mask.sum()

factor = 1 + (inversion_incremento / 100)
budgets_selected_year = {q: budget * factor for q, budget in inversion_2024.items()}
weeks_selected_year = pd.date_range(start='2025-01-01', end='2025-12-31', freq='W')
future_df = pd.DataFrame(index=weeks_selected_year)
future_df['sales'] = np.nan
future_df['marketing'] = 0.0
for q, budget in budgets_selected_year.items():
    mask = (future_df.index.year == 2025) & (future_df.index.quarter == q)
    if mask.sum() > 0:
        future_df.loc[mask, 'marketing'] = budget / mask.sum()

# === SARIMAX ===
model = sm.tsa.SARIMAX(
    weekly_sales['sales'],
    exog=weekly_sales['marketing'],
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 52),
    enforce_stationarity=False,
    enforce_invertibility=False
)
results = model.fit(disp=False)
forecast = results.get_forecast(steps=len(future_df), exog=future_df['marketing'])
future_df['sales_pred'] = forecast.predicted_mean

# === Gráfica Plotly ===
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=weekly_sales.index,
    y=weekly_sales['sales'],
    mode='lines',
    name='Ventas reales (2023–2024)',
    line=dict(color='royalblue', width=2),
    hovertemplate='Fecha: %{x|%d-%b-%Y}<br>Ventas: $%{y:,.2f}<extra></extra>'
))
fig.add_trace(go.Scatter(
    x=future_df.index,
    y=future_df['sales_pred'],
    mode='lines',
    name='Proyección ventas 2025',
    line=dict(color='orange', width=2, dash='dash'),
    hovertemplate='Fecha: %{x|%d-%b-%Y}<br>Proyección: $%{y:,.2f}<extra></extra>'
))
fig.update_layout(
    title=dict(text='📈 Ventas Semanales: Reales vs. Proyectadas (2023–2025)', x=0),
    xaxis=dict(title='Fecha (semanas)', tickformat='%b-%Y'),
    yaxis=dict(title='Ventas semanales ($)', tickprefix='$', separatethousands=True, range=[0, 8000000]),
    template='plotly_white',
    legend=dict(orientation='h', yanchor='bottom', y=-0.3, xanchor='center', x=0.5),
    margin=dict(t=80, b=60)
)
st.plotly_chart(fig)

# === Incremento de ventas proyectado ===
ventas_2024 = weekly_sales['sales'].sum()
ventas_2025 = future_df['sales_pred'].sum()
incremento = (ventas_2025 - ventas_2024) / ventas_2024 * 100
# Incremento de ventas proyectado con número grande y verde
st.markdown(
    f"📈 Incremento de ventas proyectado para 2025: "
    f"<span style='font-size:28px; color:lightgreen'><b>{incremento:.2f}%</b></span>",
    unsafe_allow_html=True
)
# === Tabla: Distribución ajustada por porcentaje interactivo ===
st.subheader("📊 Distribución de Inversión 2025 por Trimestre (ajustada por %)")
porcentajes = {"Q1": q1_pct / 100, "Q2": q2_pct / 100, "Q3": q3_pct / 100, "Q4": q4_pct / 100}
total_inversion_2025 = round(sum(inversion_2024[q] * factor for q in inversion_2024))
st.markdown(f"**Total 2025 (con incremento):** ${total_inversion_2025:,.0f}")
montos_ajustados = {q: round(total_inversion_2025 * pct) for q, pct in porcentajes.items()}
df_montos = pd.DataFrame([montos_ajustados], columns=["Q1", "Q2", "Q3", "Q4"])
st.table(df_montos.style.format("{:,.0f}").set_table_styles([
    {'selector': 'thead th', 'props': [('background-color', '#1a3c5d'), ('color', 'white'), ('font-weight', 'bold')]},
    {'selector': 'tbody td', 'props': [('text-align', 'right'), ('font-size', '14px')]}
]))

# === Botón para descargar predicción ===
csv = future_df[['sales_pred', 'marketing']].copy()
csv.index.name = 'fecha'
csv.columns = ['ventas_proyectadas', 'marketing_2025']
csv_data = csv.reset_index().to_csv(index=False).encode('utf-8')

st.download_button(
    label="📥 Descargar predicción en CSV",
    data=csv_data,
    file_name='prediccion_ventas_2025.csv',
    mime='text/csv'
)
