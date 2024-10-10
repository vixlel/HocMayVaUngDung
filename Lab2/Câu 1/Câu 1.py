import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

data = pd.read_csv('Education.csv')

# Chuẩn bị dữ liệu
X = data['Text'] 
y = data['Label'].apply(lambda x: 1 if x == 'positive' else 0)

# Chia dữ liệu thành các tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Vector hóa dữ liệu văn bản 
vectorizer = CountVectorizer(binary=True)
X_train_vec = vectorizer.fit_transform(X_train)

# Áp dụng Bernoulli Naive Bayes
bernoulli_nb = BernoulliNB()
bernoulli_nb.fit(X_train_vec, y_train)

# Áp dụng Multinomial Naive Bayes
multinomial_nb = MultinomialNB()
multinomial_nb.fit(X_train_vec, y_train)

# Lưu mô hình và bộ vectorizer bằng pickle
with open("bernoulli_model.pkl", "wb") as bnb_file:
    pickle.dump(bernoulli_nb, bnb_file)

with open("multinomial_model.pkl", "wb") as mnb_file:
    pickle.dump(multinomial_nb, mnb_file)

with open("vectorizer.pkl", "wb") as vec_file:
    pickle.dump(vectorizer, vec_file)
