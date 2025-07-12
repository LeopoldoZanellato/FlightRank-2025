#!/usr/bin/env python
# coding: utf-8

# # FlightRank 2025 - Baseline Model
# Previsão de seleção de voos com LightGBM (Lambdarank)
# 
# ---
# 
# ## 1. Imports e Configuração Inicial
# 
# Importação de bibliotecas e definição de opções globais.

# In[1]:


import pandas as pd
import os
import subprocess
import zipfile
import matplotlib.pyplot as plt
import lightgbm as lgb
import numpy as np
import lightgbm as lgb

# --- Split com GroupShuffleSplit
from sklearn.model_selection import GroupShuffleSplit
from sklearn.model_selection import GroupShuffleSplit
from itertools import chain
from sklearn.model_selection import GroupKFold
# --- LightGBM Dataset
import lightgbm as lgb


# In[2]:


# resetando as configurações 
pd.reset_option('display.max_columns')

# setando as configurações de coluna maxima
# importante pois tem muitas colunas
pd.set_option('display.max_columns', None)


# ## 2. Download e extração dos dados
# Verifica se os arquivos da competição já existem localmente, caso contrário, baixa e extrai os dados.
# 

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


# In[4]:


# Executa
download_files()


# ## 3. Leitura dos dados
# Lê o arquivo `train.parquet` com o Pandas.
# 

# In[5]:


train = pd.read_parquet("data/aeroclub/train.parquet")


# In[6]:


df_train_raw = train.copy()


# ## 4. Seleção de colunas relevantes
# Seleciona apenas as colunas que serão usadas no baseline.
# 

# In[135]:


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
rows_to_copy = 1_000_000
rows_to_copy = len(df_train_raw)
print(f"rows to read: {rows_to_copy}")
df_train = df_train_raw[columns_to_keep].iloc[:rows_to_copy].copy()


# ### 5. Engenharia de features (corrige dtypes)
# Corrige os dtypes

# In[136]:


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

# Ajusta a nacionalidade (está em Int)
df_train["nationality"] = df_train["nationality"].astype("str")


df_train.dtypes  # Checar resultado


# ## 5. Engenharia de features
# Divisão da coluna FrenquentFlyer
# 

# In[ ]:


def count_frequent_flyers(value):
    if pd.isna(value):
        return 0
    return len(str(value).split('/'))

df_train['frequentFlyer_count'] = df_train['frequentFlyer'].apply(count_frequent_flyers)

# Cria flag binária para frequent flyer
df_train['hasFrequentFlyer'] = df_train['frequentFlyer'].notnull().astype(int)

# Substituir valores NaN por string vazia
ff_series = df_train['frequentFlyer'].fillna('').astype(str)

# Dividir por '/' para obter lista
ff_lists = ff_series.str.split('/')

all_programs = set(chain.from_iterable(ff_lists))
print(f"Total de companhias únicas: {len(all_programs)}")


# In[ ]:


for program in all_programs:
    if program == '':
        continue  # pula string vazia
    df_train[f'ff_{program}'] = ff_lists.apply(lambda x: int(program in x))

for col in df_train.columns:
    if col.startswith("ff_"):
        df_train[col] = df_train[col].astype(pd.BooleanDtype())


# In[ ]:


df_train['searchRoute'].head()


# In[ ]:


df_train.drop('frequentFlyer', axis=1, inplace=True)


# ## 5. Engenharia de features
# Timedelta Columns
# 

# In[ ]:


df_train['legs0_departureAt']


# In[ ]:


def process_datetime_and_duration(df):
    df_processed = df.copy()

    # Datas para datetime
    for col in cols_datetime:
        df_processed[col] = pd.to_datetime(df_processed[col], errors='coerce')

    # Features de hora e dia da semana
    df_processed['legs0_dep_hour'] = df_processed['legs0_departureAt'].dt.hour
    df_processed['legs0_dep_dayofweek'] = df_processed['legs0_departureAt'].dt.dayofweek
    df_processed['legs1_dep_hour'] = df_processed['legs1_departureAt'].dt.hour
    df_processed['legs1_dep_dayofweek'] = df_processed['legs1_departureAt'].dt.dayofweek

    # Dias entre ida e volta (duração da viagem)
    df_processed['trip_days'] = (df_processed['legs1_departureAt'] - df_processed['legs0_departureAt']).dt.days

    # Dias de antecedência (request → ida)
    df_processed['booking_to_trip_days'] = (df_processed['legs0_departureAt'] - df_processed['requestDate']).dt.days

    # Final de semana (ida/volta)
    df_processed['ida_fds'] = df_processed['legs0_dep_dayofweek'].isin([5, 6]).astype(int)
    df_processed['volta_fds'] = df_processed['legs1_dep_dayofweek'].isin([5, 6]).astype(int)

    # Horário comercial (7h às 19h)
    def is_business_hour(hour):
        return int(7 <= hour <= 19)

    df_processed['ida_comercial'] = df_processed['legs0_dep_hour'].apply(is_business_hour)
    df_processed['volta_comercial'] = df_processed['legs1_dep_hour'].apply(is_business_hour)

    # ⏱️ Converter colunas de duração para minutos
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

    cols_duration = ['legs0_duration', 'legs1_duration']
    for col in cols_duration:
        df_processed[col] = clean_and_convert_duration(df_processed[col])

    return df_processed



# In[ ]:


df_train['legs0_duration_minutes'] = (
    pd.to_timedelta(
        df_train['legs0_segments0_duration'].fillna("00:00:00").astype(str).str.strip(),
        errors='coerce'
    ).dt.total_seconds() / 60  # em minutos
)

df_train.drop('legs0_segments0_duration', axis=1, inplace=True)


# In[ ]:


df_train['legs0_duration_minutes']


# In[ ]:


# ✅ Applicação
df_train = process_datetime_and_duration(df_train)


# In[ ]:


df_train.drop(columns=cols_datetime, inplace=True)


# ## 5. Engenharia de features
# booleans
# 

# In[ ]:


bool_cols = [
    'pricingInfo_isAccessTP',
    'hasFrequentFlyer',
]

for col in bool_cols:
    df_train[col] = df_train[col].astype('boolean')


# In[ ]:


df_train


# SEACH ROUTE

# In[ ]:


df_train['searchRoute'] = df_train['searchRoute'].astype(str)
df_train['searchRoute_count'] = df_train['searchRoute'].apply(lambda x: x.split("/"))
df_train['searchRoute_count'] = df_train['searchRoute_count'].apply(lambda x: len(x))
print(f" min {min(df_train['searchRoute_count'])}")
print(f" max {max(df_train['searchRoute_count'])}")
df_train.drop('searchRoute_count', axis=1, inplace=True)


# In[ ]:


# Garante que searchRoute está como string
df_train['searchRoute'] = df_train['searchRoute'].astype(str)

# Separa ida e volta
df_train[['route_ida', 'route_volta']] = df_train['searchRoute'].str.split('/', expand=True)

# Extrai origem e destino da ida
df_train['ida_from'] = df_train['route_ida'].str[:3]
df_train['ida_to'] = df_train['route_ida'].str[3:]

# Extrai origem e destino da volta (se existir)
df_train['volta_from'] = df_train['route_volta'].str[:3]
df_train['volta_to'] = df_train['route_volta'].str[3:]

df_train.drop('searchRoute', axis=1, inplace=True)


# In[ ]:


# Ver todos os dtypes
with pd.option_context('display.max_rows', None):
    display(df_train.dtypes)


# In[ ]:


# --- Target e grupo
target_col = "selected"
group_col = "ranker_id"

# --- Categóricas para LightGBM
categorical_cols = [
    'nationality',
    'legs0_segments0_departureFrom_airport_iata',
    'legs0_segments0_arrivalTo_airport_iata',
    'legs0_segments0_arrivalTo_airport_city_iata',
    'legs0_segments0_marketingCarrier_code',
    'legs0_segments0_operatingCarrier_code',
    'legs0_segments0_aircraft_code',
    'corporateTariffCode',
    
    # novas features categóricas da searchRoute
    'route_ida',
    'route_volta',
    'ida_from',
    'ida_to',
    'volta_from',
    'volta_to'
]

# --- Booleanas e numéricas
boolean_cols = [
    'sex', 'isVip', 'bySelf', 'isAccess3D',
    'pricingInfo_isAccessTP', 'hasFrequentFlyer',
    'ida_fds', 'volta_fds',
    'ida_comercial', 'volta_comercial'
] + [col for col in df_train.columns if col.startswith("ff_")]

numeric_cols = [
    'totalPrice', 'taxes',
    'legs0_duration', 'legs1_duration',
    #'legs0_segments0_duration',
    'legs0_segments0_baggageAllowance_quantity',
    'legs0_segments0_baggageAllowance_weightMeasurementType',
    'legs0_segments0_cabinClass',
    'legs0_segments0_seatsAvailable',
    'miniRules0_monetaryAmount', 'miniRules0_percentage',
    'miniRules1_monetaryAmount', 'miniRules1_percentage',
    'booking_to_trip_days', 'trip_days',
    'legs0_dep_hour', 'legs0_dep_dayofweek',
    'legs1_dep_hour', 'legs1_dep_dayofweek',
    'frequentFlyer_count', 'legs0_duration_minutes'
]
features = numeric_cols + categorical_cols + boolean_cols

# --- Converte categóricas para category
for col in categorical_cols:
    df_train[col] = df_train[col].astype("category")



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


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


-

