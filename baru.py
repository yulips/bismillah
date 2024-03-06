# -*- coding: utf-8 -*-
"""pca+knn.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/10XenWGC-7oaaxnzonKSfw9oCP1ircuwQ
"""

import numpy as np # linear algebra
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

from subprocess import check_output

URL = ('tbc.csv')

dataset = pd.read_csv(URL)

dataset['cough for two weeks'] = dataset['cough for two weeks'].astype('object')
dataset['weight loss'] = dataset['weight loss'].astype('object')
dataset['loss of appetite'] = dataset['loss of appetite'].astype('object')
dataset['night sweats'] = dataset['night sweats'].astype('object')
dataset['TB contact history'] = dataset['TB contact history'].astype('object')
dataset['cough with phlegm'] = dataset['cough with phlegm'].astype('object')
dataset['coughing blood'] = dataset['coughing blood'].astype('object')
dataset['BCG results appear fast'] = dataset['BCG results appear fast'].astype('object')
dataset['lumps that appear around the armpits and neck'] = dataset['lumps that appear around the armpits and neck'].astype('object')

df = pd.DataFrame(dataset)

# Memisahkan kolom 'Gender' menjadi dua kolom terpisah: 'laki-laki' dan 'perempuan'
# Kolom 'laki-laki' berisi 1 jika jenis kelamin laki-laki, 0 jika bukan
# Kolom 'perempuan' berisi 1 jika jenis kelamin perempuan, 0 jika bukan
df[['laki-laki', 'perempuan']] = pd.get_dummies(df['sex'])

# Simpan kolom 'Gender' ke variabel terpisah
kolom_gender_laki = df['laki-laki']
kolom_gender_perempuan = df['perempuan']

# Hapus kolom 'Gender' dari DataFrame
df.drop(columns=['laki-laki'], inplace=True)
df.drop(columns=['perempuan'], inplace=True)

# Masukkan kolom 'Gender' ke posisi pertama dalam DataFrame
df.insert(0, 'laki-laki', kolom_gender_laki)
df.insert(0, 'perempuan', kolom_gender_perempuan)

# Menghapus kolom 'Gender' dari DataFrame
df.drop(columns=['sex'], inplace=True)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

# Melakukan label encoding untuk fitur 'result'
df['result'] = le.fit_transform(df['result'])

# Memisahkan fitur dan target
X = df.drop(columns=['result'])
y = df['result']

# Membagi data menjadi data pelatihan dan pengujian
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=42)

# Standarisasi fitur-fitur
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Membuat DataFrame dari data
df = pd.DataFrame(X_scaled)

# Path file CSV yang akan dibuat
csv_file = 'standarisasi.csv'

# Menulis DataFrame ke dalam file CSV
df.to_csv(csv_file, index=False)