import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from datetime import datetime
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, Dense, Input, LSTM, Concatenate
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np

data = pd.read_parquet("train_dataset_hackaton2023_train.gzip")
del data['group_name']

customers = data['customer_id'].unique()

train_cust, test_cust = train_test_split(customers, test_size=0.2, random_state=77)

data['isin'] = data['customer_id'].isin(train_cust)

data['dish_name'] = data['dish_name'] + ' '

vectorizer = CountVectorizer()

vectorizer.fit(data['dish_name'])


train_df = data.loc[data['isin']]
test_df = data.loc[~data['isin']]

def get_features(df, plot=False):
    if plot: print('dish_name')
    X = df.groupby(['customer_id'], as_index=False)['dish_name'].sum()
    
    #Количество купленных блюд
    d = df.groupby(['customer_id'], as_index=False)['dish_name'].count()
    d.columns = ['customer_id', 'count_all_dishes']
    X = X.merge(d, on='customer_id', how='inner')
    
    #Количество купленных уникальных блюд
    d = df.groupby(['customer_id'], as_index=False)['dish_name'].nunique()
    d.columns = ['customer_id', 'count_uniq_dishes']
    X = X.merge(d, on='customer_id', how='inner')
    
    #Количество чеков
    d = df.groupby(['customer_id'], as_index=False)['startdatetime'].nunique()
    d.columns = ['customer_id', 'count_bills']
    X = X.merge(d, on='customer_id', how='inner')
    
    #Количество посещённых уникальных мест
    d = df.groupby(['customer_id'], as_index=False)['format_name'].nunique()
    d.columns = ['customer_id', 'count_places']
    X = X.merge(d, on='customer_id', how='inner')
    
    if plot: print('revenue')
    #Минимальная стоимость блюда
    d = df.groupby(['customer_id'], as_index=False)['revenue'].min()
    d.columns = ['customer_id', 'min_revenue']
    X = X.merge(d, on='customer_id', how='inner')
    
    #Максимальная стоимость блюда
    d = df.groupby(['customer_id'], as_index=False)['revenue'].max()
    d.columns = ['customer_id', 'max_revenue']
    X = X.merge(d, on='customer_id', how='inner')
    
    revenue_std = df.groupby('customer_id', as_index=False)['revenue'].std()
    revenue_std.columns = ['customer_id', 'revenue_std']
    X = X.merge(revenue_std, on='customer_id', how='left')
    
    #Средняя стоимость блюда
    d = df.groupby(['customer_id'], as_index=False)['revenue'].mean()
    d.columns = ['customer_id', 'mean_revenue']
    X = X.merge(d, on='customer_id', how='inner')
    
    #Минимальная стоимость чека
    temp = df.groupby(['customer_id', 'startdatetime'], as_index=False)['revenue'].sum()
    d = temp.groupby(['customer_id'], as_index=False)['revenue'].min()
    d.columns = ['customer_id', 'min_bill_revenue']
    X = X.merge(d, on='customer_id', how='inner')
    
    #Максимальная стоимость чека
    d = temp.groupby(['customer_id'], as_index=False)['revenue'].max()
    d.columns = ['customer_id', 'max_bill_revenue']
    X = X.merge(d, on='customer_id', how='inner')
    
    #Средняя стоимость чека
    d = temp.groupby(['customer_id'], as_index=False)['revenue'].mean()
    d.columns = ['customer_id', 'mean_bill_revenue']
    X = X.merge(d, on='customer_id', how='inner')
    del temp
    
    #Общая стоимость чека
    d = df.groupby(['customer_id'], as_index=False)['revenue'].sum()
    d.columns = ['customer_id', 'sum_revenue']
    X = X.merge(d, on='customer_id', how='inner')
    
    if plot: print('mean interval')
    #Средний интервал времени между покупками
    temp = data.sort_values(['customer_id', 'startdatetime'], ascending=True)
    temp['prev_date'] = temp.groupby('customer_id')['startdatetime'].shift(1)
    temp['days_between_purchases'] = (temp['startdatetime'] - temp['prev_date']).dt.days
    
    #Процент количества дней между покупками больших среднего
    mean_days = temp['days_between_purchases'].mean()
    d = temp.copy()
    d['days_more_mean'] = temp['days_between_purchases']>mean_days
    d = d.groupby('customer_id', as_index=False)['days_more_mean'].mean()
    d.columns = ['customer_id', 'percent_days_between_more_avg']
    X = X.merge(d, on='customer_id', how='left')
    
    #Последнее количество дней между покупками
    d = temp.groupby('customer_id', as_index=False)['days_between_purchases'].last()
    d.columns = ['customer_id', 'last_days_between']
    X = X.merge(d, on='customer_id', how='left')
    
    #Среднее количество дней между покупками
    d = temp.groupby('customer_id', as_index=False)['days_between_purchases'].mean()
    d.columns = ['customer_id', 'avg_days_between']
    X = X.merge(d, on='customer_id', how='left')
    
    #Среднее количество дней между покупками
    d = temp.groupby('customer_id', as_index=False)['days_between_purchases'].std()
    d.columns = ['customer_id', 'std_days_between']
    X = X.merge(d, on='customer_id', how='left')
    
    #Максимальное количество дней между покупками
    d = temp.groupby('customer_id', as_index=False)['days_between_purchases'].max()
    d.columns = ['customer_id', 'max_days_between']
    X = X.merge(d, on='customer_id', how='left')
    
    #Минимальная количество дней между покупками
    d = temp.groupby('customer_id', as_index=False)['days_between_purchases'].min()
    d.columns = ['customer_id', 'min_days_between']
    X = X.merge(d, on='customer_id', how='left')
    
    if plot: print('last_activity')
    
    #Последняя активность
    last_purchase = df.groupby('customer_id', as_index=False)['startdatetime'].max()
    last_purchase['last_activity'] = (df['startdatetime'].max()
                                      - last_purchase['startdatetime']).dt.days
    last_purchase.columns = ['customer_id', 'last_purchase_date', 'last_activity']
    X = X.merge(last_purchase[['customer_id', 'last_activity']], on='customer_id', how='left')
    
    print('date_diff')
    d_min = df.groupby(['customer_id'], as_index=False)['startdatetime'].min()
    d_min.columns = ['customer_id', 'min_data']
    d_max = df.groupby(['customer_id'], as_index=False)['startdatetime'].max()
    d_max.columns = ['customer_id', 'max_data']
    d = d_min.merge(d_max, on='customer_id', how='inner')
    d['diff_days'] = (d['max_data']-d['min_data']).dt.days
    d = d.drop(['min_data', 'max_data'], axis=1)
    X = X.merge(d, on='customer_id', how='inner')
    
    s_data = df.drop(['dish_name', 'revenue', 'isin'], axis=1)
    s_data = s_data.drop_duplicates()
    vc_fn = s_data.groupby('customer_id', as_index=False)['format_name'].value_counts()
    uniqs = data['format_name'].unique()
    for uniq in uniqs:
        g = vc_fn.loc[vc_fn['format_name']==uniq]
        g.columns = ['customer_id', 'format_name', 'count_format_name_'+uniq]
        X = X.merge(g.drop(['format_name'], axis=1), on='customer_id', how='left')
    X.fillna(0, inplace=True)
    
    return X

X_train = get_features(train_df, plot=True)
#del X_train['customer_id']
y_train_bin = train_df.groupby(['customer_id'], as_index=False)['buy_post'].first()
del y_train_bin['customer_id']


X_test = get_features(test_df, plot=True)
#del X_test['customer_id']
y_test_bin = test_df.groupby(['customer_id'], as_index=False)['buy_post'].first()
del y_test_bin['customer_id']

X = get_features(data, plot=True)
#del X['customer_id']
y_bin = data.groupby(['customer_id'], as_index=False)['buy_post'].first()
del y_bin['customer_id']

# Обработка текстовых данных
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train['dish_name'])
X_train_text = tokenizer.texts_to_sequences(X_train['dish_name'])
X_test_text = tokenizer.texts_to_sequences(X_test['dish_name'])

# Дополнение последовательностей до одинаковой длины
max_length = max(len(x) for x in X_train_text)
X_train_text = pad_sequences(X_train_text, maxlen=max_length, padding='post')
X_test_text = pad_sequences(X_test_text, maxlen=max_length, padding='post')

# Обработка числовых данных
numeric_features = X_train.columns.drop('dish_name')
scaler = StandardScaler()
X_train_numeric = scaler.fit_transform(X_train[numeric_features])
X_test_numeric = scaler.transform(X_test[numeric_features])

# Параметры для сети
vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 50

# Входы модели
text_input = Input(shape=(max_length,), name='text_input')
numeric_input = Input(shape=(X_train_numeric.shape[1],), name='numeric_input')

# Текстовая часть модели
text_part = Embedding(vocab_size, embedding_dim)(text_input)
text_part = LSTM(128)(text_part)

# Комбинация текстовых и числовых данных
combined = Concatenate()([text_part, numeric_input])

# Полносвязные слои
dense = Dense(128, activation='gelu')(combined)
dense = Dense(128, activation='gelu')(dense)
dense = Dense(128, activation='gelu')(dense)
output = Dense(1, activation='sigmoid')(dense)

# Создание и компиляция модели
model = Model(inputs=[text_input, numeric_input], outputs=output)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# Путь, по которому будет сохраняться модель
checkpoint_filepath = 'checkpoint'

# Создание объекта ModelCheckpoint
model_checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=False,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

# Обучение модели с callback
history = model.fit(
    [X_train_text, X_train_numeric], y_train_bin,
    validation_data=([X_test_text, X_test_numeric], y_test_bin),
    epochs=30,
    batch_size=32,
    callbacks=[model_checkpoint_callback])

np.random.randint(1, 8)