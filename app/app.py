import streamlit as st
import pandas as pd
import joblib
from xgboost import XGBRegressor
from sklearn.impute import SimpleImputer

st.set_page_config(page_title="Previsões Brasileirão", layout="wide")

# Carregar modelo
@st.cache_resource
def carregar_modelo():
    return joblib.load("models/modelo_brasileirao_xgb.pkl")

modelo = carregar_modelo()

# Carregar dados históricos
@st.cache_data
def carregar_dados():
    return pd.read_csv("data/brasileirao/brasileirao_scores.csv", encoding='cp1252')

df = carregar_dados()

# Features usadas no modelo
features_usadas = [
    'Home_xG_Avg', 'Away_xG_Avg', 'Team_Rank', 'Away_Team_Rank',
    'Total_Goals_For', 'Home_Goals_Against', 'Total_Goals_Against',
    'Away_Goals_Against', 'Last_5_Wins'
]

# Título
st.title("⚽ Previsões de Gols - Brasileirão")

# Selecionar rodada
rodadas_disponiveis = sorted(df["k"].dropna().unique())
rodada = st.selectbox("Escolha a rodada:", rodadas_disponiveis)

# Filtrar confrontos da rodada
confrontos_df = df[df["k"] == rodada][["Home", "Away"]]
confrontos_lista = confrontos_df.apply(lambda row: f"{row['Home']} x {row['Away']}", axis=1).tolist()
jogo_escolhido = st.selectbox("Escolha o confronto:", confrontos_lista)

# Prever e exibir estatísticas
if jogo_escolhido:
    time_casa, time_fora = jogo_escolhido.split(" x ")
    linha_jogo = df[(df["Home"] == time_casa) & (df["Away"] == time_fora)]

    if not linha_jogo.empty:
        linha = linha_jogo.iloc[0]
        st.write("🧩 Colunas disponíveis na linha selecionada:", linha.index.tolist())
        X_jogo = linha[features_usadas].to_frame().T

        # Imputação
        imputer = SimpleImputer(strategy='mean')
        X_imputado = imputer.fit_transform(df[features_usadas])  # fit no dataset inteiro
        X_jogo_imputado = imputer.transform(X_jogo)

        # Previsão
        pred = modelo.predict(X_jogo_imputado)
        st.metric("🔮 Gols previstos para o mandante", f"{pred[0]:.2f}")

        # Estatísticas dos times
        st.subheader("📊 Estatísticas recentes do time da casa")
        st.dataframe(df[df["Home"] == time_casa][features_usadas].describe())

        st.subheader("📊 Estatísticas recentes do time visitante")
        st.dataframe(df[df["Away"] == time_fora][features_usadas].describe())

# Tabela do Brasileirão
st.header("📊 Tabela Atual do Brasileirão")
try:
    tabela = pd.read_csv("data/brasileirao/tabela_brasileirao.csv", encoding='utf-8-sig')
    st.dataframe(tabela)
except:
    st.warning("⚠️ Não foi possível carregar a tabela.")
