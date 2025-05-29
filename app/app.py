import streamlit as st
import pandas as pd
import joblib
from xgboost import XGBRegressor
from sklearn.impute import SimpleImputer

st.set_page_config(page_title="Previsões Brasileirão", layout="wide")

# Carregar modelo
@st.cache_resource
def carregar_modelo():
    return joblib.load("../models/modelo_brasileirao_xgb.pkl")

modelo = carregar_modelo()

# Carregar dados históricos
@st.cache_data
def carregar_dados():
    return pd.read_csv("../data/brasileirao/brasileirao_scores.csv", encoding='cp1252')

df = carregar_dados()

# Features usadas no modelo
features_usadas = [
    'Home_xG_Avg', 'Away_xG_Avg', 'Team_Rank', 'Away_Team_Rank',
    'Total_Goals_For', 'Home_Goals_Against', 'Total_Goals_Against',
    'Away_Goals_Against', 'Last_5_Wins'
]

# Título
st.title("⚽ Previsões de Gols - Brasileirão")

# Seleção dos times
times_casa = sorted(df["Home"].dropna().astype(str).unique())
time_casa = st.selectbox("Selecione o time da casa", times_casa)
times_fora = sorted(df["Away"].dropna().astype(str).unique())
time_fora = st.selectbox("Selecione o time visitante", times_fora)

# Previsão se ambos os times forem selecionados e diferentes
if time_casa and time_fora and time_casa != time_fora:
    linha_casa = df[(df["Home"] == time_casa) & (df["Away"] == time_fora)]

    if not linha_casa.empty:
        # Preparar entrada para previsão dos gols do mandante
        linha_casa = linha_casa.iloc[0]
        X_casa = linha_casa[features_usadas].to_frame().T

        # Imputação
        imputer = SimpleImputer(strategy='mean')
        imputer.fit(df[features_usadas])
        X_casa_imputado = imputer.transform(X_casa)

        # Previsão dos gols do mandante
        gols_casa = modelo.predict(X_casa_imputado)[0]

        # Tentar prever gols do visitante (invertendo os times)
        linha_fora = df[(df["Home"] == time_fora) & (df["Away"] == time_casa)]

        if not linha_fora.empty:
            linha_fora = linha_fora.iloc[0]
            X_fora = linha_fora[features_usadas].to_frame().T
            X_fora_imputado = imputer.transform(X_fora)
            gols_fora = modelo.predict(X_fora_imputado)[0]
        else:
            gols_fora = None

        # Exibir resultados
        st.subheader("🎯 Previsões de Gols")
        st.metric("🔵 Gols previstos para o time da casa", f"{gols_casa:.2f}")

        if gols_fora is not None:
            st.metric("🟡 Gols previstos para o time visitante", f"{gols_fora:.2f}")
            st.metric("⚖️ Placar total previsto", f"{(gols_casa + gols_fora):.2f}")
        else:
            st.info("⚠️ Não há dados suficientes para prever os gols do visitante (inverso do confronto).")

# Tabela do Brasileirão
st.header("📊 Tabela Atual do Brasileirão")
try:
    tabela = pd.read_csv("../data/brasileirao/tabela_brasileirao.csv", encoding='utf-8-sig')
    st.dataframe(tabela)
except:
    st.warning("⚠️ Não foi possível carregar a tabela.")
