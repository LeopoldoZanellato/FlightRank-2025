#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import os
import subprocess
import zipfile
import matplotlib.pyplot as plt
from itertools import chain
from sklearn.model_selection import GroupKFold
import lightgbm as lgb
import numpy as np


# “Dado um conjunto de voos apresentados a um usuário, qual é o mais provável que ele selecione?”
# 
# É um problema clássico de:
# 
# -Aprendizado supervisionado
# 
# -Com objetivo de ranking
# 
# -E grupos estruturados (ranker_id)
# 
# HitRate@3
# Você acerta se o voo real (selected = 1) estiver entre os 3 primeiros ranks do seu modelo para cada grupo com mais de 10 opções.
# 
# 
# 
# Principais perguntas:
# 
# - Qual é a média de opções de voo por sessão aberta
# - A quantidade de opções de voo impactam na decisão do cliente? Mais opções, mais compras?
# - Quantas sessões que eu não tive nenhuma compra
# 

# In[2]:


pd.reset_option("all")
pd.set_option('display.max_columns', None)


# In[3]:


def download_files():
    # utilizando a API do kaggle para download
    # adicionado a /data no git ignore
    # Define caminhos
    zip_path = "data/aeroclub-recsys-2025.zip"
    extract_path = "data/aeroclub"
    
    # Cria a pasta base se necessário
    os.makedirs("data", exist_ok=True)

    # Verifica se o arquivo .zip já foi baixado
    if not os.path.exists(zip_path):
        print("🔽 Baixando arquivos da competição...")
        subprocess.run([
            "kaggle", "competitions", "download",
            "-c", "aeroclub-recsys-2025",
            "-p", "data"
        ])
    else:
        print("✅ Arquivo ZIP já existe. Pulando download.")

    # Verifica se os arquivos já foram extraídos
    if not os.path.exists(extract_path) or not os.listdir(extract_path):
        print("📦 Extraindo arquivos...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
    else:
        print("✅ Arquivos já extraídos. Pulando extração.")

# Executa
download_files()


# In[4]:


def reduce_memory_usage(df):
    for col in df.columns:
        col_type = df[col].dtypes
        
        if col_type == 'float64':
            df[col] = pd.to_numeric(df[col], downcast='float')
        elif col_type == 'int64':
            df[col] = pd.to_numeric(df[col], downcast='integer')
        elif col_type == 'object':
            num_unique = df[col].nunique()
            num_total = len(df[col])
            if num_unique / num_total < 0.5:
                df[col] = df[col].astype('category')
    
    return df


# In[5]:


train = pd.read_parquet("data/aeroclub/train.parquet")


# In[6]:


#df_train_raw = reduce_memory_usage(train)
df_train_raw = train.copy()


# In[7]:


# Define as colunas que você quer manter
columns_to_keep = [
    # Identifiers
    'Id',  # num
    'ranker_id', 
    'profileId', 
    'companyID',
    
    # User info
    'sex', 'nationality', 'frequentFlyer', 'isVip', 'bySelf', 'isAccess3D',

    # Company info
    'corporateTariffCode',

    # Search & route
    'searchRoute', 'requestDate',

    # Pricing
    'totalPrice', 'taxes',

    # Flight timing
    'legs0_departureAt', 'legs0_arrivalAt', 'legs0_duration',
    'legs1_departureAt', 'legs1_arrivalAt', 'legs1_duration',

    # Segment-level info (só do segmento 0 da ida para simplificar no baseline)
    'legs0_segments0_departureFrom_airport_iata',
    'legs0_segments0_arrivalTo_airport_iata',
    'legs0_segments0_arrivalTo_airport_city_iata',
    'legs0_segments0_marketingCarrier_code',
    'legs0_segments0_operatingCarrier_code',
    'legs0_segments0_aircraft_code',
    'legs0_segments0_flightNumber',
    'legs0_segments0_duration',
    'legs0_segments0_baggageAllowance_quantity',
    'legs0_segments0_baggageAllowance_weightMeasurementType',
    'legs0_segments0_cabinClass',
    'legs0_segments0_seatsAvailable',

    # Cancellation & exchange rules
    'miniRules0_monetaryAmount', 'miniRules0_percentage', 'miniRules0_statusInfos',
    'miniRules1_monetaryAmount', 'miniRules1_percentage', 'miniRules1_statusInfos',

    # Pricing policy
    'pricingInfo_isAccessTP', 'pricingInfo_passengerCount',

    # Target
    'selected'
]

# Filtra os dados para o baseline
rows_to_copy = 18_000_000
df_train = df_train_raw[columns_to_keep].iloc[:].copy()

# Garante que o ranker_id está em string
df_train['ranker_id'] = df_train['ranker_id'].astype(str)


# In[8]:


def fix_column_types(df):
    df_fixed = df.copy()
    for col in df.columns:
        if isinstance(df[col].dtype, pd.CategoricalDtype):
            # Tenta converter para tipo numérico
            try:
                df_fixed[col] = pd.to_numeric(df[col])
            except:
                # Se não for numérico, tenta bool
                unique_vals = df[col].dropna().unique()
                if set(unique_vals) <= {True, False}:
                    df_fixed[col] = df[col].astype(bool)
                else:
                    df_fixed[col] = df[col].astype(str)
    return df_fixed
df_train = fix_column_types(df_train)
df_train.dtypes  # Checar resultado


# In[9]:


# 🗓️ Colunas de datas e horários
cols_datetime = [
    'requestDate',
    'legs0_departureAt', 'legs0_arrivalAt',
    'legs1_departureAt', 'legs1_arrivalAt'
]

def process_datetime_and_duration(df):
    df_processed = df.copy()

    for col in cols_datetime:
        df_processed[col] = pd.to_datetime(df_processed[col], errors='coerce')

    # 📆 Features derivadas de datas
    df_processed['legs0_dep_hour'] = df_processed['legs0_departureAt'].dt.hour
    df_processed['legs0_dep_dayofweek'] = df_processed['legs0_departureAt'].dt.dayofweek
    df_processed['legs1_dep_hour'] = df_processed['legs1_departureAt'].dt.hour
    df_processed['trip_days'] = (df_processed['legs1_departureAt'] - df_processed['legs0_departureAt']).dt.days
    df_processed['booking_to_trip_days'] = (df_processed['legs0_departureAt'] - df_processed['requestDate']).dt.days

    # ⏱️ Função segura para converter duração para minutos
    def clean_and_convert_duration(col):
        return (
            col
            .fillna("00:00:00")
            .astype(str)
            .str.strip()
            .str.replace("nan", "00:00:00")
            .pipe(pd.to_timedelta, errors='coerce')
            .dt.total_seconds() / 60  # minutos
        )

    # ⏳ Colunas de duração
    cols_duration = ['legs0_duration', 'legs1_duration']
    for col in cols_duration:
        df_processed[col] = clean_and_convert_duration(df_processed[col])

    return df_processed

# ✅ Exemplo de uso:
df_train = process_datetime_and_duration(df_train)
df_train[['legs0_duration', 'trip_days', 'legs0_dep_hour']].head()


# In[10]:


df_train.drop(columns=cols_datetime, inplace=True)


# In[ ]:





# In[11]:


df_train.head()


# In[12]:


df_train


# In[13]:


df_train.columns[:20]


# In[14]:


group_sizes = df_train.groupby('ranker_id').size()
group_sizes.describe()


# In[15]:


group_sizes.hist(bins=100, figsize=(10, 5))
plt.title("Distribuição de número de voos por sessão (ranker_id)")
plt.xlabel("Quantidade de opções")
plt.ylabel("Número de sessões")
plt.xlim(0, 1500)  # Limita o eixo X para melhor visualização
plt.grid(True)
plt.show()


# In[16]:


df_resume = df_train[['ranker_id', 'selected']].groupby('ranker_id').sum()
df_resume = df_resume.reset_index()


# In[17]:


df_resume.sort_values(by='selected', ascending=False)


# In[18]:


df_resume.count()


# In[19]:


print("💰 Preço médio voo escolhido:", df_train['totalPrice'].mean())
print("💸 Preço médio voo não escolhido:", df_train['totalPrice'].mean())


# In[20]:


#df_selected


# In[21]:


def count_frequent_flyers(value):
    if pd.isna(value):
        return 0
    return len(str(value).split('/'))

df_train['frequentFlyer_count'] = df_train['frequentFlyer'].apply(count_frequent_flyers)


# In[22]:


# Converte string de tempo para timedelta e depois para minutos
df_train['legs0_duration_minutes'] = pd.to_timedelta(df_train['legs0_duration'], errors='coerce').dt.total_seconds() / 60
# Cria flag binária para frequent flyer
df_train['hasFrequentFlyer'] = df_train['frequentFlyer'].notnull().astype(int)
# Separar escolhidos e não escolhidos
df_selected = df_train[df_train['selected'] == 1]
df_not_selected = df_train[df_train['selected'] == 0]

# Variáveis corrigidas para comparação
features = {
    'totalPrice': '💰 Preço total',
    'taxes': '💸 Taxas',
    'legs0_duration_minutes': '⏱️ Duração (minutos)',
    'frequentFlyer_count': '🛫Quantidade Frequent Flyer',
    'isVip': '👑 VIP',
    'pricingInfo_isAccessTP': '✅ Segue política',
    'legs0_segments0_cabinClass': '💺 Classe do voo (ida - seg. 0)',
}

# Montar a tabela de médias
comparison = []

for col, label in features.items():
    s1 = pd.to_numeric(df_selected[col], errors='coerce')
    s0 = pd.to_numeric(df_not_selected[col], errors='coerce')
    
    mean_selected = s1.mean()
    mean_not_selected = s0.mean()
    diff = mean_selected - mean_not_selected
    
    comparison.append({
        '🔹 Variável': label,
        'Escolhido (1)': round(mean_selected, 2),
        'Não escolhido (0)': round(mean_not_selected, 2),
        'Diferença': round(diff, 2)
    })

# Criar DataFrame de comparação
df_comparison = pd.DataFrame(comparison)

# Exibir a tabela formatada
print("\n📊 Comparação de médias entre voos escolhidos e não escolhidos:\n")
print(df_comparison.to_string(index=False))


# In[23]:


df_comparison


# In[24]:


df_train['legs0_duration']


# In[25]:


df_train['frequentFlyer'] = df_train['frequentFlyer'].astype(str)


# In[26]:


df_train['frequentFlyer'].unique()


# In[27]:


# Substituir valores NaN por string vazia
ff_series = df_train['frequentFlyer'].fillna('').astype(str)

# Dividir por '/' para obter lista
ff_lists = ff_series.str.split('/')

all_programs = set(chain.from_iterable(ff_lists))
print(f"Total de companhias únicas: {len(all_programs)}")


# In[28]:


for program in all_programs:
    if program == '':
        continue  # pula string vazia
    df_train[f'ff_{program}'] = ff_lists.apply(lambda x: int(program in x))

for col in df_train.columns:
    if col.startswith("ff_"):
        df_train[col] = df_train[col].astype(pd.BooleanDtype())


# In[29]:


df_train.drop('frequentFlyer', axis=1, inplace=True)


# In[ ]:





# In[30]:


df_train['nationality'].unique()


# In[31]:


df_train['searchRoute']


# In[32]:


df_train["nationality"] = df_train["nationality"].astype("str")


# In[33]:


df_train["searchRoute"] = df_train["searchRoute"].astype("str")


# In[34]:


df_train['legs0_segments0_duration']


# In[35]:


df_train['legs0_segments0_duration']


# In[36]:


df_train['legs0_segments0_duration'] = (
    pd.to_timedelta(
        df_train['legs0_segments0_duration'].fillna("00:00:00").astype(str).str.strip(),
        errors='coerce'
    ).dt.total_seconds() / 60  # em minutos
)


# In[37]:


cols_cat = [
    'legs0_segments0_departureFrom_airport_iata',
    'legs0_segments0_arrivalTo_airport_iata',
    'legs0_segments0_arrivalTo_airport_city_iata',
    'legs0_segments0_marketingCarrier_code',
    'legs0_segments0_operatingCarrier_code',
    'legs0_segments0_aircraft_code'
]

for col in cols_cat:
    df_train[col] = df_train[col].astype('str')


# In[38]:


pd.set_option('display.max_columns', None)  # Mostra todas as colunas
pd.set_option('display.max_rows', 100)      # Limite de linhas visíveis (ajuste se quiser)
pd.set_option('display.max_colwidth', None) # Mostra conteúdo completo de colunas

#pd.reset_option('display.max_columns')
#pd.reset_option('display.max_rows')
#pd.reset_option('display.max_colwidth')


# In[39]:


df_train.dtypes  # Checar resultado


# In[40]:


df_train['pricingInfo_isAccessTP'].unique()


# In[41]:


df_train['hasFrequentFlyer'].unique()


# In[42]:


bool_cols = [
    'pricingInfo_isAccessTP',
    'hasFrequentFlyer',
]

for col in bool_cols:
    df_train[col] = df_train[col].astype('boolean')


# In[43]:


df_train.dtypes  # Checar resultado


# In[45]:


# --- Target e grupo
target_col = "selected"
group_col = "ranker_id"

# --- Categóricas para LightGBM
categorical_cols = [
    'nationality',
    'searchRoute',
    'legs0_segments0_departureFrom_airport_iata',
    'legs0_segments0_arrivalTo_airport_iata',
    'legs0_segments0_arrivalTo_airport_city_iata',
    'legs0_segments0_marketingCarrier_code',
    'legs0_segments0_operatingCarrier_code',
    'legs0_segments0_aircraft_code',
    'corporateTariffCode'
]

# --- Booleanas e numéricas
boolean_cols = [
    'sex', 'isVip', 'bySelf', 'isAccess3D', 'pricingInfo_isAccessTP', 'hasFrequentFlyer'
] + [col for col in df_train.columns if col.startswith("ff_")]

numeric_cols = [
    'totalPrice', 'taxes', 'legs0_duration', 'legs1_duration',
    'legs0_segments0_duration', 'legs0_segments0_baggageAllowance_quantity',
    'legs0_segments0_baggageAllowance_weightMeasurementType',
    'legs0_segments0_cabinClass', 'legs0_segments0_seatsAvailable',
    'miniRules0_monetaryAmount', 'miniRules0_percentage',
    'miniRules1_monetaryAmount', 'miniRules1_percentage',
    'booking_to_trip_days', 'trip_days', 'legs0_dep_hour',
    'legs1_dep_hour', 'frequentFlyer_count', 'legs0_duration_minutes'
]

features = numeric_cols + categorical_cols + boolean_cols

# --- Converte categóricas para category
for col in categorical_cols:
    df_train[col] = df_train[col].astype("category")

# --- Split com GroupShuffleSplit
from sklearn.model_selection import GroupShuffleSplit

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, val_idx = next(gss.split(df_train, groups=df_train["ranker_id"]))

df_train_split = df_train.iloc[train_idx].copy()
df_val = df_train.iloc[val_idx].copy()  # << IMPORTANTE

X_train = df_train_split[features]
y_train = df_train_split[target_col]
groups_train = df_train_split[group_col].value_counts().sort_index().values

X_val = df_val[features]
y_val = df_val[target_col]
groups_val = df_val[group_col].value_counts().sort_index().values

# --- LightGBM Dataset
import lightgbm as lgb

train_dataset = lgb.Dataset(X_train, y_train, group=groups_train, categorical_feature=categorical_cols)
val_dataset = lgb.Dataset(X_val, y_val, group=groups_val, categorical_feature=categorical_cols, reference=train_dataset)

# --- Parâmetros
params = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "ndcg_eval_at": [3],
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 20,
    "verbosity": -1,
}

# --- Treinamento com early stopping
model = lgb.train(
    params,
    train_dataset,
    valid_sets=[train_dataset, val_dataset],
    valid_names=["train", "valid"],
    num_boost_round=1000,
    callbacks=[lgb.early_stopping(stopping_rounds=50)],
    #verbose_eval=50
)

# --- Predição
y_pred = model.predict(X_val)

# --- Avaliação Top-1
df_pred = df_val.copy()
df_pred['y_true'] = y_val
df_pred['y_pred'] = y_pred

df_pred_sorted = df_pred.sort_values(['ranker_id', 'y_pred'], ascending=[True, False])
df_top1 = df_pred_sorted.groupby('ranker_id').head(1)

acertos = df_top1['y_true'].sum()
total = df_top1.shape[0]

print(f"Voos escolhidos corretamente (top1): {acertos} de {total} sessões")
print(f"Acurácia top1: {acertos / total:.4f}")


# In[46]:


# 1. Ler test.parquet
df_test = pd.read_parquet("data/aeroclub/test.parquet")

# 2. Aplicar as transformações mínimas necessárias

# Convertendo tipos
df_test['ranker_id'] = df_test['ranker_id'].astype(str)
df_test['nationality'] = df_test['nationality'].astype(str)
df_test['searchRoute'] = df_test['searchRoute'].astype(str)

# Frequent Flyer (cria os mesmos one-hot da base de treino)
df_test['frequentFlyer'] = df_test['frequentFlyer'].fillna('').astype(str)
ff_lists_test = df_test['frequentFlyer'].str.split('/')
for program in all_programs:
    if program == '':
        continue
    df_test[f'ff_{program}'] = ff_lists_test.apply(lambda x: program in x)

# Tipos booleanos
for col in [col for col in df_test.columns if col.startswith("ff_")]:
    df_test[col] = df_test[col].astype(pd.BooleanDtype())

# Outras variáveis derivadas
df_test['frequentFlyer_count'] = df_test['frequentFlyer'].apply(count_frequent_flyers)
df_test['hasFrequentFlyer'] = df_test['frequentFlyer'].notnull().astype(int)
df_test.drop(columns=['frequentFlyer'], inplace=True)

# Datas
cols_datetime = [
    'requestDate',
    'legs0_departureAt', 'legs0_arrivalAt',
    'legs1_departureAt', 'legs1_arrivalAt'
]
for col in cols_datetime:
    df_test[col] = pd.to_datetime(df_test[col], errors='coerce')

df_test['legs0_dep_hour'] = df_test['legs0_departureAt'].dt.hour
df_test['legs0_dep_dayofweek'] = df_test['legs0_departureAt'].dt.dayofweek
df_test['legs1_dep_hour'] = df_test['legs1_departureAt'].dt.hour
df_test['trip_days'] = (df_test['legs1_departureAt'] - df_test['legs0_departureAt']).dt.days
df_test['booking_to_trip_days'] = (df_test['legs0_departureAt'] - df_test['requestDate']).dt.days
df_test.drop(columns=cols_datetime, inplace=True)

# Duração
def clean_and_convert_duration(col):
    return (
        col
        .fillna("00:00:00")
        .astype(str)
        .str.strip()
        .str.replace("nan", "00:00:00")
        .pipe(pd.to_timedelta, errors='coerce')
        .dt.total_seconds() / 60
    )
df_test['legs0_duration'] = clean_and_convert_duration(df_test['legs0_duration'])
df_test['legs1_duration'] = clean_and_convert_duration(df_test['legs1_duration'])
df_test['legs0_segments0_duration'] = clean_and_convert_duration(df_test['legs0_segments0_duration'])
df_test['legs0_duration_minutes'] = df_test['legs0_duration']  # mesmo valor

# Corrigir categoricals
for col in categorical_cols:
    df_test[col] = df_test[col].astype("category")

# Corrigir booleanos
for col in ['sex', 'isVip', 'bySelf', 'isAccess3D', 'pricingInfo_isAccessTP', 'hasFrequentFlyer']:
    df_test[col] = df_test[col].astype("boolean")

# 3. Prever com o modelo
X_test = df_test[features]
df_test['y_pred'] = model.predict(X_test)

# 4. Gerar submissão com ranking por ranker_id
df_test_sorted = df_test.sort_values(['ranker_id', 'y_pred'], ascending=[True, False])
df_test_sorted['selected'] = df_test_sorted.groupby('ranker_id').cumcount() + 1

submission = df_test_sorted[['Id', 'ranker_id', 'selected']]
submission.to_csv("submission.csv", index=False)
print("✅ Arquivo de submissão salvo como 'submission.csv'")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




