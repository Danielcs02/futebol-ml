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

# Título
st.title("⚽ Previsões de Gols - Brasileirão")

# Upload de arquivo CSV
st.header("📤 Envie os jogos para prever")
arquivo = st.file_uploader("Escolha um CSV com os jogos", type=["csv"])

# Previsão
if arquivo:
    df_input = pd.read_csv(arquivo)

    st.subheader("🔍 Visualização dos dados enviados")
    st.dataframe(df_input)

    # Garantir que as colunas esperadas existam
    colunas_esperadas = ['Home_xG_Avg', 'Away_xG_Avg', 'Team_Rank', 'Away_Team_Rank',
                         'Total_Goals_For', 'Home_Goals_Against', 'Total_Goals_Against',
                         'Away_Goals_Against', 'Last_5_Wins']
    if not all(col in df_input.columns for col in colunas_esperadas):
        st.error("O CSV está faltando uma ou mais colunas necessárias para previsão.")
    else:
        # Imputação e previsão
        imputer = SimpleImputer(strategy='mean')
        X_novo = imputer.fit_transform(df_input[colunas_esperadas])
        pred = modelo.predict(X_novo)

        df_input['Gols_Previstos'] = pred

        st.success("✅ Previsões geradas!")
        st.dataframe(df_input[['Home', 'Away', 'Gols_Previstos']])

        # Botão para download
        csv_resultado = df_input.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Baixar resultados", csv_resultado, "previsoes.csv", "text/csv")

# Tabela do Brasileirão
st.header("📊 Tabela Atual do Brasileirão")
try:
    tabela = pd.read_csv("../data/brasileirao/tabela_brasileirao.csv", encoding='utf-8-sig')
    st.dataframe(tabela)
except:
    st.warning("⚠️ Não foi possível carregar a tabela.")
