from flask import Flask, render_template, request
import pickle

# Khởi tạo Flask
app = Flask(__name__)

# Tải 3 mô hình và vectorizer đã lưu ở file trước
try:
    with open("bernoulli_model.pkl", "rb") as bnb_file:
        bnb_model = pickle.load(bnb_file)
    with open("multinomial_model.pkl", "rb") as mnb_file:
        mnb_model = pickle.load(mnb_file)
    with open("vectorizer.pkl", "rb") as vec_file:
        vectorizer = pickle.load(vec_file)
except FileNotFoundError:
    print("Error: Model or vectorizer file not found.")
    exit()

# Route trang chủ
@app.route('/')
def home():
    return render_template('index.html')

# Route để xử lý dự đoán
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        sentence = request.form['sentence']
        if sentence:
            X_new = vectorizer.transform([sentence])
            prediction_bernoulli = bnb_model.predict(X_new)[0]
            prediction_multinomial = mnb_model.predict(X_new)[0]

            # Xử lý hiển thị kết quả dự đoán
            return render_template('index.html', 
                                   sentence=sentence,
                                   prediction_bernoulli="Positive" if prediction_bernoulli == 1 else "Negative",
                                   prediction_multinomial="Positive" if prediction_multinomial == 1 else "Negative")
        else:
            return render_template('index.html', error="Please enter a sentence.")

if __name__ == '__main__':
    app.run(debug=True)
