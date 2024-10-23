from flask import Flask, render_template
import pickle

app = Flask(__name__)

# Load the soft margin model data
with open('svm_model.pkl', 'rb') as file:
    soft_margin_model = pickle.load(file)

# Load the hard margin model data
with open('hard_margin_svm_model.pkl', 'rb') as file:
    hard_margin_model = pickle.load(file)

@app.route('/')
def combined_margin():
    # Extract variables from both models
    soft_w = soft_margin_model['w']
    soft_b = soft_margin_model['b']
    soft_slack = soft_margin_model['slack']

    hard_w = hard_margin_model['w']
    hard_b = hard_margin_model['b']
    hard_lamb = hard_margin_model['lamb']

    # Pass variables and image paths to the template
    return render_template('index.html', 
                           soft_w=soft_w, soft_b=soft_b, soft_slack=soft_slack,
                           hard_w=hard_w, hard_b=hard_b, hard_lamb=hard_lamb,
                           soft_margin_image='static/soft_margin_plot.png',
                           hard_margin_image='static/hard_margin_plot.png')

if __name__ == '__main__':
    app.run(debug=True)
