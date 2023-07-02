
    
from flask import Flask, render_template, request, redirect, url_for
from ml_model import pickle

app = Flask(__name__)
model = pickle.load(open('RandomForest.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [
        float(request.form['N']),
        float(request.form['P']),
        float(request.form['K']),
        float(request.form['temperature']),
        float(request.form['humidity']),
        float(request.form['ph']),
        float(request.form['rainfall'])
    ]
    prediction = model.predict([features])[0]

    return redirect(url_for('result', prediction=prediction))

@app.route('/result')
def result():
    prediction = request.args.get('prediction')

    return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run()
    
    
    
    
