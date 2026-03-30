“””
Page 2 — Neural Network Model Description
“””
import streamlit as st
import pandas as pd
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(**file**)))

st.set_page_config(page_title=“Neural Network | HealthAI”, page_icon=“🧠”, layout=“wide”)

st.markdown(”””

<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.page-banner {
    background: linear-gradient(135deg, #831843 0%, #be185d 60%, #ec4899 100%);
    border-radius: 18px; padding: 2.5rem; color: white; margin-bottom: 2rem;
}
.page-banner h1 { font-family:'DM Serif Display',serif; font-size:2rem; margin:0 0 0.4rem; }
.page-banner p  { opacity:0.85; font-size:1rem; margin:0; font-weight:300; }
.section-card {
    background: white; border-radius: 14px; padding: 1.8rem;
    box-shadow: 0 2px 14px rgba(0,0,0,0.06); margin-bottom: 1.5rem;
}
.section-card h3 { color: #831843; margin-top:0; font-size:1.15rem; }
.layer-box {
    background: linear-gradient(135deg, #fdf2f8, #fce7f3);
    border: 2px solid #f9a8d4;
    border-radius: 12px; padding: 1rem 1.5rem;
    margin: 0.5rem 0; text-align:center;
}
.layer-title { font-weight:700; color:#be185d; font-size:0.95rem; }
.layer-desc { font-size:0.82rem; color:#6b7280; margin-top:0.2rem; }
.arrow { text-align:center; font-size:1.2rem; color:#ec4899; }
table { width:100%; border-collapse:collapse; font-size:0.88rem; }
th { background:#be185d; color:white; padding:0.6rem 1rem; text-align:left; }
td { padding:0.55rem 1rem; border-bottom:1px solid #f1f5f9; color:#374151; }
tr:nth-child(even) td { background:#fafafa; }
</style>

“””, unsafe_allow_html=True)

st.markdown(”””

<div class="page-banner">
  <div style='font-size:0.75rem; letter-spacing:2px; opacity:0.7; margin-bottom:0.5rem'>PAGE 2 / 4</div>
  <h1>🧠 Neural Network — MLP Model</h1>
  <p>Diabetes Prediction ด้วย Multi-Layer Perceptron พร้อม BatchNorm + Dropout ออกแบบโครงสร้างเอง</p>
</div>
""", unsafe_allow_html=True)

tabs = st.tabs([“📂 Dataset & Preprocessing”, “🧠 ทฤษฎี ANN”, “🏗️ โครงสร้างโมเดล”, “🔧 ขั้นตอนพัฒนา”, “📚 References”])

# ── Tab 1: Dataset ────────────────────────────────────────────────────────────

with tabs[0]:
st.markdown(’<div class="section-card">’, unsafe_allow_html=True)
st.markdown(”### 📂 ที่มาของ Dataset”)
st.markdown(”””
**Pima Indians Diabetes Dataset** มาจาก **National Institute of Diabetes and Digestive and Kidney Diseases**
และเผยแพร่บน **Kaggle** ประกอบด้วยข้อมูลสุขภาพของผู้หญิงเชื้อสาย Pima Indian
**768 ตัวอย่าง** เพื่อทำนายว่ามีเบาหวาน Type 2 หรือไม่

```
> 🔗 Source: [kaggle.com/datasets/uciml/pima-indians-diabetes-database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
""")
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown("### 🔍 Features ของ Dataset")
feat_df = pd.DataFrame({
    "Feature": ["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"],
    "คำอธิบาย": [
        "จำนวนครั้งที่ตั้งครรภ์",
        "ระดับ Plasma Glucose 2 ชั่วโมงหลัง Oral Glucose Tolerance Test",
        "Diastolic Blood Pressure (mm Hg)",
        "Triceps Skin Fold Thickness (mm)",
        "ระดับ 2-Hour Serum Insulin (mu U/ml)",
        "Body Mass Index (น้ำหนัก/ส่วนสูง²)",
        "ฟังก์ชันประมาณการถ่ายทอดทางพันธุกรรมของเบาหวาน",
        "อายุ (ปี)",
    ],
    "ปัญหา Missing": [
        "ไม่มี", "Zero = missing (ค่า 0 ไม่สมเหตุสมผล)", "Zero = missing",
        "Zero = missing", "Zero = missing (มีมาก)", "Zero = missing",
        "ไม่มี", "ไม่มี",
    ],
})
st.table(feat_df)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown("### 🔧 การเตรียมข้อมูล")
st.markdown("""
**ปัญหาพิเศษของ Dataset นี้**: ค่า **0** ใน Glucose, BloodPressure, SkinThickness, Insulin, BMI
ไม่ใช่ค่าจริง แต่เป็น missing values ที่บันทึกเป็น 0 แทน → ต้องแปลงเป็น NaN ก่อน
""")
steps = [
    ("Zero → NaN", "แปลง 0 ใน Glucose, BloodPressure, SkinThickness, Insulin, BMI เป็น NaN"),
    ("Median Imputation", "แทนค่า NaN ด้วย median ของแต่ละ column (robust ต่อ outlier)"),
    ("Duplicate Removal", "ลบ rows ซ้ำออก"),
    ("IQR Outlier Capping", "จำกัด outlier สำหรับ Glucose, BMI, Insulin"),
    ("StandardScaler", "Normalize ทุก feature → mean=0, std=1 เหมาะกับ Neural Network"),
    ("80/20 Stratified Split", "แบ่ง train/test พร้อม stratify เพื่อรักษา class ratio"),
]
for i, (t, d) in enumerate(steps, 1):
    st.markdown(f"""
    <div style='display:flex; gap:1rem; align-items:flex-start; padding:0.6rem 0; border-bottom:1px solid #fce7f3'>
      <div style='background:#be185d; color:white; border-radius:50%; width:26px; height:26px;
                  display:flex; align-items:center; justify-content:center;
                  font-size:0.75rem; font-weight:700; flex-shrink:0'>{i}</div>
      <div><b style='color:#831843'>{t}</b> — <span style='color:#374151; font-size:0.88rem'>{d}</span></div>
    </div>
    """, unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)
```

# ── Tab 2: Theory ─────────────────────────────────────────────────────────────

with tabs[1]:
st.markdown(’<div class="section-card">’, unsafe_allow_html=True)
st.markdown(”### 🧠 ทฤษฎี Artificial Neural Network”)
st.markdown(”””
**ANN (Artificial Neural Network)** จำลองการทำงานของสมองมนุษย์ ประกอบด้วย **neurons**
ที่เชื่อมกันเป็นชั้นๆ:

```
**Forward Propagation**: คำนวณค่า output จาก input ผ่านหลาย layers
```
z = W·x + b
a = activation(z)
```

**Backpropagation**: คำนวณ gradient ของ loss function แล้วอัพเดท weights
```
∂L/∂W = ∂L/∂a · ∂a/∂z · ∂z/∂W
W = W - α · ∂L/∂W
```
""")

col1, col2 = st.columns(2)
with col1:
    st.markdown("""
    **🔵 Activation Functions**
    - **ReLU**: f(x) = max(0, x) — ใช้ใน hidden layers
    - **Sigmoid**: f(x) = 1/(1+e⁻ˣ) — output layer สำหรับ binary classification
    - แก้ปัญหา vanishing gradient ได้ดีกว่า Sigmoid/Tanh
    """)
with col2:
    st.markdown("""
    **🟣 Regularization Techniques**
    - **Dropout**: ปิด neurons แบบสุ่มระหว่าง training ป้องกัน overfitting
    - **Batch Normalization**: normalize activations ช่วยให้ training เร็วขึ้นและเสถียร
    - **Early Stopping**: หยุด training เมื่อ val_loss ไม่ดีขึ้น
    """)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown("### ⚙️ Adam Optimizer")
st.markdown("""
โมเดลใช้ **Adam (Adaptive Moment Estimation)** optimizer:
- ผสม **Momentum** (ทิศทางการ update) และ **RMSProp** (ขนาดการ update)
- ปรับ learning rate แบบ adaptive สำหรับแต่ละ parameter
- เหมาะกับ sparse gradients และ noisy problems

**Binary Cross-Entropy Loss** สำหรับ binary classification:
```
L = -[y·log(ŷ) + (1-y)·log(1-ŷ)]
```
""")
st.markdown('</div>', unsafe_allow_html=True)
```

# ── Tab 3: Architecture ───────────────────────────────────────────────────────

with tabs[2]:
st.markdown(’<div class="section-card">’, unsafe_allow_html=True)
st.markdown(”### 🏗️ โครงสร้างโมเดล MLP”)
st.markdown(“โมเดล designed สำหรับ tabular health data โดยเฉพาะ:”)

```
layers = [
    ("Input Layer", "8 neurons", "รับ 8 features ของ Diabetes dataset"),
    ("Dense (128) + BatchNorm + ReLU + Dropout(0.3)", "128 neurons", "Hidden Layer 1 — เรียนรู้ feature interactions"),
    ("Dense (64) + BatchNorm + ReLU + Dropout(0.3)", "64 neurons", "Hidden Layer 2 — compression และ abstraction"),
    ("Dense (32) + BatchNorm + ReLU + Dropout(0.3)", "32 neurons", "Hidden Layer 3 — refined representations"),
    ("Output Layer (Dense 1, Sigmoid)", "1 neuron", "ทำนาย probability of Diabetes (0–1)"),
]

col_arch, col_why = st.columns([1, 1])
with col_arch:
    for name, size, desc in layers:
        st.markdown(f"""
        <div class="layer-box">
          <div class="layer-title">{name}</div>
          <div class="layer-desc">{size} — {desc}</div>
        </div>
        <div class="arrow">↓</div>
        """, unsafe_allow_html=True)

with col_why:
    st.markdown("#### 🎯 เหตุผลการออกแบบ")
    reasons = [
        ("128→64→32 (Funnel)", "ลดขนาด layer ทีละขั้น ช่วยให้โมเดล compress information อย่างค่อยเป็นค่อยไป"),
        ("BatchNorm ทุก layer", "ทำให้ gradient flow ดีขึ้น ลด internal covariate shift ฝึกเร็วขึ้น"),
        ("Dropout 0.3", "ปิด 30% ของ neurons แต่ละ batch ป้องกัน co-adaptation"),
        ("Early Stopping (patience=15)", "หยุดเมื่อ val_loss ไม่ลดลง 15 epochs ป้องกัน overfitting"),
        ("Adam lr=0.001", "เหมาะกับ tabular data ขนาดเล็ก balance ระหว่างความเร็วและความเสถียร"),
        ("Batch size=16", "เหมาะกับ dataset ขนาดเล็ก (~700 rows) ให้ gradient estimate ที่ดีพอ"),
    ]
    for r_title, r_desc in reasons:
        st.markdown(f"""
        <div style='padding:0.6rem 0; border-bottom:1px solid #fce7f3'>
          <b style='color:#be185d; font-size:0.88rem'>{r_title}</b><br>
          <span style='color:#374151; font-size:0.82rem'>{r_desc}</span>
        </div>
        """, unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown("### 📋 Model Summary")
model_summary = pd.DataFrame({
    "Layer": ["Dense", "BatchNorm", "Dropout", "Dense", "BatchNorm", "Dropout",
              "Dense", "BatchNorm", "Dropout", "Dense (Output)"],
    "Units / Rate": ["128", "-", "0.3", "64", "-", "0.3", "32", "-", "0.3", "1"],
    "Activation": ["ReLU", "-", "-", "ReLU", "-", "-", "ReLU", "-", "-", "Sigmoid"],
    "Parameters": ["1,152", "512", "-", "8,256", "256", "-", "2,080", "128", "-", "33"],
})
st.table(model_summary)
st.markdown('</div>', unsafe_allow_html=True)
```

# ── Tab 4: Dev Steps ──────────────────────────────────────────────────────────

with tabs[3]:
st.markdown(’<div class="section-card">’, unsafe_allow_html=True)
st.markdown(”### 🔧 ขั้นตอนการพัฒนา”)
dev_steps = [
(“1. Data Understanding”, [“วิเคราะห์ distribution และ zero-values”, “ตรวจสอบ class balance (35% positive, 65% negative)”]),
(“2. Preprocessing”, [“Zero → NaN, Median impute, Scale”, “Stratified 80/20 split”]),
(“3. Architecture Design”, [“เลือก Funnel MLP (128→64→32) เหมาะกับ tabular data ขนาดเล็ก”, “เพิ่ม BatchNorm + Dropout ป้องกัน overfitting”]),
(“4. Training Configuration”, [“Loss: Binary Cross-Entropy”, “Optimizer: Adam (lr=0.001)”, “Callbacks: EarlyStopping (patience=15, restore_best_weights=True)”]),
(“5. Evaluation”, [“Accuracy, Precision, Recall, F1, ROC-AUC บน test set”, “Plot training curves (loss & accuracy)”, “Confusion matrix และ threshold analysis”]),
(“6. Save & Deploy”, [“บันทึกด้วย model.save() (Keras format)”, “โหลดด้วย tf.keras.models.load_model()”, “ใช้ใน Streamlit รับ user input และแสดง probability”]),
]
for title, items in dev_steps:
with st.expander(title):
for item in items:
st.markdown(f”- {item}”)
st.markdown(’</div>’, unsafe_allow_html=True)

# ── Tab 5: References ─────────────────────────────────────────────────────────

with tabs[4]:
st.markdown(’<div class="section-card">’, unsafe_allow_html=True)
st.markdown(”### 📚 แหล่งอ้างอิง”)
refs = [
(“Pima Indians Diabetes Dataset”, “Smith, J.W. et al. (1988). ADAP Learning Algorithm. Proc. Annual Symp. on Computer Application in Medical Care.”, “https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database”),
(“TensorFlow / Keras”, “Abadi, M. et al. (2015). TensorFlow: Large-Scale Machine Learning on Heterogeneous Systems.”, “https://www.tensorflow.org/”),
(“Adam Optimizer”, “Kingma, D.P. & Ba, J. (2014). Adam: A Method for Stochastic Optimization. ICLR 2015.”, “https://arxiv.org/abs/1412.6980”),
(“Batch Normalization”, “Ioffe, S. & Szegedy, C. (2015). Batch Normalization. ICML 2015.”, “https://arxiv.org/abs/1502.03167”),
(“Dropout”, “Srivastava, N. et al. (2014). Dropout: A Simple Way to Prevent Neural Networks from Overfitting. JMLR 15.”, “http://jmlr.org/papers/v15/srivastava14a.html”),
(“Deep Learning Book”, “Goodfellow, I., Bengio, Y. & Courville, A. (2016). Deep Learning. MIT Press.”, “https://www.deeplearningbook.org/”),
]
for title, citation, url in refs:
st.markdown(f”””
<div style='padding:0.75rem 0; border-bottom:1px solid #fce7f3'>
<b style='color:#be185d'>{title}</b><br>
<span style='font-size:0.85rem; color:#64748b'>{citation}</span><br>
<a href='{url}' style='font-size:0.8rem; color:#ec4899'>{url}</a>
</div>
“””, unsafe_allow_html=True)
st.markdown(’</div>’, unsafe_allow_html=True)
