from flask import Flask, render_template, request
import pickle

# Ứng dụng Flask để dự đoán đơn thuốc
app = Flask(__name__)

# Tải các mô hình và bộ mã hóa nhãn đã lưu
with open("gaussian_nb_model.pkl", "rb") as model_file:
    gaussian_nb_model = pickle.load(model_file)

with open("le_Age.pkl", "rb") as le_age_file:
    le_Age = pickle.load(le_age_file)

with open("le_Sex.pkl", "rb") as le_sex_file:
    le_sex = pickle.load(le_sex_file)

with open("le_BP.pkl", "rb") as le_bp_file:
    le_bp = pickle.load(le_bp_file)

with open("le_Cholesterol.pkl", "rb") as le_cholesterol_file:
    le_cholesterol = pickle.load(le_cholesterol_file)

with open("le_Na_to_K.pkl", "rb") as le_na_to_k_file:
    le_Na_to_K = pickle.load(le_na_to_k_file)

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        age = request.form['age']
        sex = request.form['sex']
        bp = request.form['bp']
        cholesterol = request.form['cholesterol']
        na_to_k = request.form['na_to_k']

        age_enc = le_Age.transform([age])[0]
        sex_enc = le_sex.transform([sex])[0]
        bp_enc = le_bp.transform([bp])[0]
        cholesterol_enc = le_cholesterol.transform([cholesterol])[0]
        na_to_k_enc = le_Na_to_K.transform([na_to_k])[0]

        X_new = [[age_enc, sex_enc, bp_enc, cholesterol_enc, na_to_k_enc]]
        prediction = gaussian_nb_model.predict(X_new)[0]

        return render_template('index.html', 
                               age=age, 
                               sex=sex, 
                               bp=bp, 
                               cholesterol=cholesterol, 
                               na_to_k=na_to_k, 
                               prediction=prediction)
    
if __name__ == '__main__':
    app.run(debug=True)