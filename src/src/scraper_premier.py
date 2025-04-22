import requests
from bs4 import BeautifulSoup
import pandas as pd

# URL da tabela de jogos da Premier League
url = "https://fbref.com/en/comps/9/schedule/Premier-League-Scores-and-Fixtures"

# Faz a requisição
res = requests.get(url)
soup = BeautifulSoup(res.text, 'lxml')

# Encontra todas as tabelas da página
tables = soup.find_all('table')

# Busca pela tabela com o título certo
target_table = None
for table in tables:
    caption = table.find('caption')
    if caption and "Scores & Fixtures" in caption.text:
        target_table = table
        break

# Se a tabela foi encontrada, continua
if target_table:
    headers = [th.getText() for th in target_table.find('thead').find_all('th')]

    rows = target_table.find('tbody').find_all('tr')
    data = []
    for row in rows:
        if row.find('td'):
            data.append([td.getText() for td in row.find_all('td')])

    # Cria o DataFrame
    df = pd.DataFrame(data, columns=headers[1:])  # remove coluna em branco
    df.to_csv('data/premier/premier_league_scores.csv', index=False)
    print("✅ CSV salvo com sucesso!")
else:
    print("❌ Tabela 'Scores & Fixtures' não encontrada.")

