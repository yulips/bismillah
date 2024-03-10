from flask import Flask, render_template, request, jsonify
import pickle
import os
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

#KNN

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

# Inisialisasi classification_report dengan string kosong
classification_report = loaded_data.get('classification_report', '')

# Load Model
scaler = pickle.load(open('./models/scaler.pkl', 'rb'))
pca = pickle.load(open('./models/pca.pkl', 'rb'))
le = pickle.load(open('./models/label_encoder.pkl', 'rb'))
model = pickle.load(open('./models/model.pkl', 'rb'))

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
            form_values = request.form.to_dict()

            data = {}

            data = {key: [float(value)] if value.replace('.', '', 1).isdigit() else value for key, value in form_values.items()}

            # Membuat data kedalam bentuk dataframe
            df = pd.DataFrame(data)

            # Melakukan one-hot encoding (pre-processing) untuk kolom 'sex'
            df['laki-laki'] = df['sex'].apply(lambda x: 1 if x == 'Male' else 0)
            df['perempuan'] = df['sex'].apply(lambda x: 1 if x == 'Female' else 0)
            # Menghapus kolom 'sex' yang asli
            df.drop('sex', axis=1, inplace=True)
            # Mengurutkan kolom 'laki-laki' dan 'perempuan' ke depan DataFrame
            df = df.reindex(columns=['perempuan', 'laki-laki'] + [col for col in df.columns if col not in ['perempuan', 'laki-laki']])

            # Mengubah data ke dalam scaler
            new_data_std = scaler.transform(df)
            # Mentransformasi jumlah komponen sesuai PCA
            new_data_pca = pca.transform(new_data_std)

            # Prediksi hasil
            prediction = model.predict(new_data_pca)
            predicted_class = prediction[0]
            predicted_class_original = le.inverse_transform([predicted_class])[0]

            # Return prediction as JSON
            return jsonify(predicted_class_original)

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

    return render_template('confusion_matrix.html', akurasi=akurasi, classification_report=classification_report, accuracy=accuracy, classification_report_str=classification_report_str, css_file='css/boostrap.min.css')

if __name__ == '__main__':
    app.run(debug=True)