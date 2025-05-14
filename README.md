# ⚽ Previsão de Gols no Campeonato Brasileiro de Futebol

Projeto de machine learning para prever a **quantidade total de gols** em partidas do Campeonato Brasileiro, usando dados públicos do FBref e métricas como xG (expected goals).


## 📊 Variáveis usadas pelo Modelo para fazer as projeções
Variáveis utilizadas no modelo:
- Home_xG_Avg: Média de xG (gols esperados) para o time da casa jogando em casa.
- Away_xG_Avg: Média de xG para o time visitante jogando fora de casa.
- Team_Rank: Ranking (posição) do time da casa no campeonato.
- Away_Team_Rank: Ranking (posição) do time visitante no campeonato.
- Total_Goals_For: Total de gols feitos pelo time da casa no campeonato inteiro.
- Home_Goals_Against: Total de gols sofridos pelo time da casa no campeonato inteiro.
- Total_Goals_Against: Total de gols feitos pelo time visitante no campeonato inteiro.
- Away_Goals_Against: Total de gols sofridos pelo time visitante no campeonato inteiro.
- Last_5_Wins: Número de vitórias nos últimos 5 jogos de cada time (com base nos placares anteriores).

## 📊 Tecnologias utilizadas
- Python
- Pandas
- Scikit-Learn
- Matplotlib
- Jupyter Notebook

## 📁 Estrutura do projeto
futebol-ml/ ├── data/ # CSV com dados das partidas ├── notebooks/ # Notebook com análise e previsões ├── src/ # Código auxiliar ├── .gitignore # Arquivos ignorados pelo Git ├── requirements.txt # Bibliotecas necessárias └── README.md # Este arquivo

## 🚀 Como executar

1. Clone o repositório:
https://github.com/Danielcs02/futebol-ml.git


2. Instale os pacotes:
pip install -r requirements.txt

3. Execute o notebook:
Abra `notebooks/exploracao_premier.ipynb` no Jupyter ou VSCode.

## 📈 Funcionalidades
- Previsão de gols por partida
- Visualização de confrontos com placar estimado
- Avaliação com MAE e RMSE

## 🔮 Próximos passos
- Previsão por time
- Atualização automática por rodada
- Deploy interativo com Streamlit
