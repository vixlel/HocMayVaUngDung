import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
from flask import Flask, render_template, request

data = pd.read_csv('drug.csv')

# Chuyển đổi các đặc trưng phân loại sang dạng số sử dụng LabelEncoder
le_Age = LabelEncoder()
le_sex = LabelEncoder()
le_bp = LabelEncoder()
le_cholesterol = LabelEncoder()
le_Na_to_K = LabelEncoder()

data['Age'] = le_Age.fit_transform(data['Age'])
data['Sex'] = le_sex.fit_transform(data['Sex'])
data['BP'] = le_bp.fit_transform(data['BP'])
data['Cholesterol'] = le_cholesterol.fit_transform(data['Cholesterol'])
data['Na_to_K'] = le_Na_to_K.fit_transform(data['Na_to_K'])

X = data[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']]  
y = data['Drug']  

# Chia dữ liệu thành các tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Áp dụng Gaussian Naive Bayes
gaussian_nb = GaussianNB()
gaussian_nb.fit(X_train, y_train)

# Lưu mô hình Gaussian Naive Bayes và LabelEncoders bằng pickle
with open("gaussian_nb_model.pkl", "wb") as model_file:
    pickle.dump(gaussian_nb, model_file)

with open("le_Age.pkl", "wb") as le_age_file:
    pickle.dump(le_Age, le_age_file)

with open("le_Sex.pkl", "wb") as le_sex_file:
    pickle.dump(le_sex, le_sex_file)

with open("le_BP.pkl", "wb") as le_bp_file:
    pickle.dump(le_bp, le_bp_file)

with open("le_Cholesterol.pkl", "wb") as le_cholesterol_file:
    pickle.dump(le_cholesterol, le_cholesterol_file)

with open("le_Na_to_K.pkl", "wb") as le_na_to_k_file:
    pickle.dump(le_Na_to_K, le_na_to_k_file)
