from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import os
import pandas as pd 

#KNN
# Membuka file pickle dan memuat data knn
with open('./knn/data_file.pkl', 'rb') as f:
    loaded_data = pickle.load(f)

# Membuka file pickle untuk akurasi knn
with open('./knn/akurasiakhir-knn.pkl', 'rb') as f1:
    akurasi = pickle.load(f1)

# Membuka file pickle untuk classification report knn
with open('./knn/cfreport-knn.pkl', 'rb') as f2:
    classification_report = pickle.load(f2)

#PCA + KNN

# Membuka file pickle dan memuat data
with open('./knnpca/data_file_gab.pkl', 'rb') as f4:
    loaded_data = pickle.load(f4)

# Membuka file pickle untuk akurasi
with open('./knnpca/akurasiakhirgab.pkl', 'rb') as f5:
    accuracy = pickle.load(f5)

# Membuka file pickle untuk classification report
with open('./knnpca/cfreportgab.pkl', 'rb') as f6:
    classification_report_str = pickle.load(f6)

    
# Extract path gambar dan dataframe fitur dari data yang dimuat
#confusion_matrix_path = loaded_data['./knnpca/conmat-gab.png']

# Inisialisasi classification_report dengan string kosong
classification_report = loaded_data.get('classification_report', '')

# Load the KNN model
prediksi = './knnpca/tbc-gab.pkl'
model = pickle.load(open(prediksi, 'rb'))

# Inisialisasi objek Flask
app = Flask(__name__, template_folder='frontend', static_folder='static')

@app.route('/')
def home():
    return render_template('index.html', css_file='css/style.css')

@app.route('/about')
def about():
    return render_template('about-us.html', css_file='css/about-us.css')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html', css_file='css/boostrap.min.css')

@app.route('/pengujian')
def pengujian():  
    return render_template('pengujian.html', css_file='css/boostrap.min.css')

@app.route('/dataset')
def dataset():
    # Menentukan path lengkap file tbc.csv
    csv_tbc_path = os.path.join('tbc.csv')

    # Membaca dataset tbc.csv
    df_tbc = pd.read_csv(csv_tbc_path)

    # Menentukan path lengkap file tbc.csv
    csv_tbc_path = os.path.join('standarisasi.csv')

    # Membaca dataset tbc.csv
    df_tbc_baru = pd.read_csv(csv_tbc_path)


    return render_template('dataset.html', df_tbc=df_tbc,  df_tbc_baru=df_tbc_baru, css_file='css/boostrap.min.css')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Extracting form values
            name = str(request.form.get('name'))
            sex = int(request.form.get('sex'))
            age = int(request.form.get('age'))
            coufor = float(request.form['coufor'])
            nisw = float(request.form.get('nisw'))  
            welos = float(request.form.get('welos'))
            loap = float(request.form['loap'])
            tbch = float(request.form.get('tbch'))
            coph = float(request.form['coph'])
            thal = float(request.form.get('thal'))
            cb = float(request.form['cb'])
            bcg = float(request.form.get('bcg'))
            ltaa = float(request.form['ltaa'])
            
            # Creating a NumPy array for the input data
            data = np.array([[name, sex, age, coufor, nisw, welos, loap, tbch, coph, thal, cb, bcg, ltaa]])

            # Predict the class and probabilities
            predicted_class = model.predict(data)
            probabilities = model.predict_proba(data)
            predicted_prob = probabilities[0][predicted_class[0]]

            # Return prediction as JSON
            return jsonify({'Predicted Class': predicted_class.tolist(), 'Accuracy': predicted_prob})

        except Exception as e:
            return {"error": str(e)}

@app.route('/cm')
def confusion():

    # Membuka file pickle untuk akurasi knn
    with open('./knn/akurasiakhir-knn.pkl', 'rb') as f1:
        akurasi = pickle.load(f1)

    # Membuka file pickle untuk classification report knn
    with open('./knn/cfreport-knn.pkl', 'rb') as f2:
        classification_report = pickle.load(f2)

    # Membuka file pickle untuk akurasi
    with open('./knnpca/akurasiakhirgab.pkl', 'rb') as f5:
        accuracy = pickle.load(f5)

    # Membuka file pickle untuk classification report
    with open('./knnpca/cfreportgab.pkl', 'rb') as f6:
        classification_report_str = pickle.load(f6)

    return render_template('confusion_matrix.html', akurasi=akurasi, classification_report=classification_report ,accuracy=accuracy, classification_report_str=classification_report_str, css_file='css/boostrap.min.css')

@app.route('/gambarcm', methods=['POST'])
def gambarcm():
    if request.method == 'POST':
        try:
            # Menggunakan jsonify untuk mengirimkan path gambar dan data tabel ke HTML
            return jsonify({
                #'confusion_matrix': confusion_matrix_path,
            })

        except Exception as e:
            return {"error": str(e)}


if __name__ == '__main__':
    app.run(debug=True)