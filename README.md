# pro3
Streamlit web application สำหรับทำนายความเสี่ยงโรคสุขภาพด้วย Machine Learning และ Neural Network

health_ml_app/
├── app.py                     # หน้าหลัก (Home)
├── pages/
│   ├── 1_ML_Description.py   # อธิบาย ML Ensemble Model
│   ├── 2_NN_Description.py   # อธิบาย Neural Network Model
│   ├── 3_ML_Prediction.py    # ทดสอบ ML (Heart Disease)
│   └── 4_NN_Prediction.py    # ทดสอบ NN (Diabetes)
├── utils/
│   ├── data_prep.py          # Data preprocessing
│   └── models.py             # Model training & evaluation
├── data/
│   ├── heart_disease.csv     # Dataset 1 (UCI)
│   └── diabetes.csv          # Dataset 2 (Pima Indians)
├── models/                   # Saved model files (auto-generated)
├── .streamlit/config.toml    # Streamlit config
└── requirements.txt

