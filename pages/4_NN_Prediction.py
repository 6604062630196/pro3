“””
Page 4 — Neural Network Demo (Diabetes Prediction)
“””
import streamlit as st
import pandas as pd
import numpy as np
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(**file**)))

st.set_page_config(page_title=“NN Prediction | HealthAI”, page_icon=“💡”, layout=“wide”)

st.markdown(”””

<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.page-banner {
    background: linear-gradient(135deg, #831843 0%, #9d174d 50%, #ec4899 100%);
    border-radius: 18px; padding: 2rem 2.5rem; color: white; margin-bottom: 2rem;
}
.page-banner h1 { font-family:'DM Serif Display',serif; font-size:1.8rem; margin:0 0 0.3rem; }
.page-banner p  { opacity:0.85; font-size:0.95rem; margin:0; }
.card { background:white; border-radius:14px; padding:1.5rem;
        box-shadow:0 2px 14px rgba(0,0,0,0.06); margin-bottom:1.5rem; }
.result-safe {
    background: linear-gradient(135deg, #f0fdf4, #dcfce7);
    border: 2px solid #86efac; border-radius: 16px; padding: 2rem; text-align:center;
}
.result-risk {
    background: linear-gradient(135deg, #fdf2f8, #fce7f3);
    border: 2px solid #f9a8d4; border-radius: 16px; padding: 2rem; text-align:center;
}
.stButton > button {
    background: linear-gradient(135deg, #be185d, #9d174d) !important;
    color: white !important; border: none; border-radius: 10px;
    padding: 0.7rem 2rem; font-weight: 600; font-size: 1rem;
    box-shadow: 0 4px 14px rgba(190,24,93,0.4);
}
</style>

“””, unsafe_allow_html=True)

st.markdown(”””

<div class="page-banner">
  <div style='font-size:0.75rem; letter-spacing:2px; opacity:0.7; margin-bottom:0.5rem'>PAGE 4 / 4</div>
  <h1>💡 Neural Network — Diabetes Prediction</h1>
  <p>ป้อนข้อมูลสุขภาพเพื่อทำนายความเสี่ยงโรคเบาหวาน Type 2 ด้วย MLP Neural Network</p>
</div>
""", unsafe_allow_html=True)

# ── Load / Train Model ────────────────────────────────────────────────────────

@st.cache_resource
def get_diabetes_model():
from utils.data_prep import prepare_diabetes_data
from utils.models import train_nn, load_nn, evaluate_model

```
data = prepare_diabetes_data()
nn = load_nn("diabetes")

if nn is None:
    nn, metrics = train_nn(
        data["X_train"], data["y_train"],
        data["X_test"],  data["y_test"],
        name="diabetes"
    )
else:
    metrics = evaluate_model(nn, data["X_test"], data["y_test"], is_nn=True)

return nn, data, metrics
```

# TensorFlow availability check

tf_available = True
try:
import tensorflow as tf
except ImportError:
tf_available = False

if not tf_available:
st.warning(”””
⚠️ **TensorFlow ไม่ได้ติดตั้งในระบบ**

```
เพื่อรันโมเดล Neural Network กรุณาติดตั้ง:
```
pip install tensorflow
```
จากนั้น restart Streamlit แล้วลองใหม่อีกครั้ง

**ในระหว่างนี้**: หน้านี้จะแสดงสถาปัตยกรรมโมเดลและรับ input ได้ แต่ยังไม่สามารถทำนายได้จริง
""")
model_ok = False
data = None
metrics = {}
```

else:
with st.spinner(“⏳ กำลังโหลด/ฝึก Neural Network กรุณารอสักครู่…”):
try:
nn_model, data, metrics = get_diabetes_model()
model_ok = nn_model is not None
except Exception as e:
st.error(f”เกิดข้อผิดพลาด: {e}”)
model_ok = False
data = None
metrics = {}

# ── Model Metrics ─────────────────────────────────────────────────────────────

if model_ok and metrics:
st.markdown(”### 📊 ประสิทธิภาพโมเดล (Test Set)”)
c1, c2, c3, c4, c5 = st.columns(5)
for col, (label, key, color) in zip(
[c1, c2, c3, c4, c5],
[(“Accuracy”,“accuracy”,”#ec4899”),(“Precision”,“precision”,”#8b5cf6”),
(“Recall”,“recall”,”#f59e0b”),(“F1-Score”,“f1”,”#10b981”),(“ROC-AUC”,“roc_auc”,”#3b82f6”)]
):
with col:
val = metrics.get(key, 0)
st.markdown(f”””
<div style='background:white; border-radius:12px; padding:1rem; text-align:center;
box-shadow:0 2px 8px rgba(0,0,0,0.06); border-top:3px solid {color}'>
<div style='font-size:1.6rem; font-weight:800; color:{color}'>{val:.1%}</div>
<div style='font-size:0.78rem; color:#64748b; font-weight:500; margin-top:0.2rem'>{label}</div>
</div>
“””, unsafe_allow_html=True)

```
st.markdown("<br>", unsafe_allow_html=True)

# Training history
if "history" in metrics:
    with st.expander("📈 Training History"):
        hist = metrics["history"]
        col_loss, col_acc = st.columns(2)
        with col_loss:
            loss_df = pd.DataFrame({
                "Training Loss": hist["loss"],
                "Validation Loss": hist["val_loss"],
            })
            st.line_chart(loss_df)
            st.caption("Loss per Epoch")
        with col_acc:
            acc_df = pd.DataFrame({
                "Training Accuracy": hist["accuracy"],
                "Validation Accuracy": hist["val_accuracy"],
            })
            st.line_chart(acc_df)
            st.caption("Accuracy per Epoch")

with st.expander("📊 Confusion Matrix"):
    cm = metrics.get("confusion_matrix", [[0,0],[0,0]])
    cm_df = pd.DataFrame(
        cm,
        index=["Actual: No Diabetes","Actual: Diabetes"],
        columns=["Predicted: No Diabetes","Predicted: Diabetes"],
    )
    st.dataframe(cm_df)
```

# ── Architecture Display ───────────────────────────────────────────────────────

with st.expander(“🏗️ โครงสร้าง Neural Network”):
st.markdown(”””
`Input (8 features) ↓ Dense(128, ReLU) → BatchNorm → Dropout(0.3) ↓ Dense(64, ReLU)  → BatchNorm → Dropout(0.3) ↓ Dense(32, ReLU)  → BatchNorm → Dropout(0.3) ↓ Dense(1, Sigmoid) ↓ Output: P(Diabetes)  [0.0 – 1.0]`
**Loss**: Binary Cross-Entropy | **Optimizer**: Adam (lr=0.001) | **Batch**: 16 | **Early Stopping**: patience=15
“””)

# ── Prediction Input ──────────────────────────────────────────────────────────

st.markdown(”—”)
st.markdown(”### 🔬 ทดสอบการทำนาย”)

col_l, col_r = st.columns(2)
with col_l:
st.markdown(’<div class="card">’, unsafe_allow_html=True)
st.markdown(”**🤰 ข้อมูลทั่วไป**”)
pregnancies = st.slider(“จำนวนครั้งที่ตั้งครรภ์”, 0, 17, 3)
age         = st.slider(“อายุ (ปี)”, 18, 80, 33)
glucose     = st.slider(“ระดับ Glucose (mg/dl)”, 50, 200, 120)
blood_pressure = st.slider(“Blood Pressure (mmHg)”, 40, 130, 72)
st.markdown(’</div>’, unsafe_allow_html=True)

with col_r:
st.markdown(’<div class="card">’, unsafe_allow_html=True)
st.markdown(”**🩺 ข้อมูลร่างกาย**”)
skin_thickness = st.slider(“Skin Thickness (mm)”, 0, 60, 23)
insulin     = st.slider(“Insulin (mu U/ml)”, 0, 500, 85)
bmi         = st.slider(“BMI”, 15.0, 60.0, 28.5, 0.1)
dpf         = st.slider(“Diabetes Pedigree Function”, 0.05, 2.5, 0.35, 0.01)
st.markdown(’</div>’, unsafe_allow_html=True)

st.markdown(”<br>”, unsafe_allow_html=True)
col_btn = st.columns([2, 1, 2])[1]
with col_btn:
predict_btn = st.button(“🔮 ทำนายความเสี่ยง”, use_container_width=True)

if predict_btn:
if not model_ok or data is None:
st.error(“ไม่สามารถทำนายได้ กรุณาติดตั้ง TensorFlow ก่อน (`pip install tensorflow`)”)
else:
input_raw = np.array([[pregnancies, glucose, blood_pressure,
skin_thickness, insulin, bmi, dpf, age]])
input_scaled = data[“scaler”].transform(input_raw)

```
    prob = float(nn_model.predict(input_scaled, verbose=0)[0][0])
    pred = 1 if prob >= 0.5 else 0

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🎯 ผลการทำนาย")

    if pred == 1:
        st.markdown(f"""
        <div class="result-risk">
          <div style='font-size:3rem; margin-bottom:0.5rem'>⚠️</div>
          <div style='font-family:"DM Serif Display",serif; font-size:1.8rem; color:#be185d; margin:0 0 0.3rem'>
            มีความเสี่ยงเบาหวาน
          </div>
          <p style='color:#831843; margin:0.5rem 0'>Neural Network ตรวจพบสัญญาณความเสี่ยง ควรปรึกษาแพทย์</p>
          <div style='font-size:2rem; font-weight:800; color:#be185d'>{prob:.1%}</div>
          <div style='font-size:0.85rem; color:#9d174d'>Probability of Diabetes</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-safe">
          <div style='font-size:3rem; margin-bottom:0.5rem'>✅</div>
          <div style='font-family:"DM Serif Display",serif; font-size:1.8rem; color:#16a34a; margin:0 0 0.3rem'>
            ความเสี่ยงต่ำ
          </div>
          <p style='color:#14532d; margin:0.5rem 0'>Neural Network ไม่พบสัญญาณความเสี่ยงเบาหวานที่น่ากังวล</p>
          <div style='font-size:2rem; font-weight:800; color:#16a34a'>{prob:.1%}</div>
          <div style='font-size:0.85rem; color:#166534'>Probability of Diabetes</div>
        </div>
        """, unsafe_allow_html=True)

    # Risk gauge
    st.markdown("<br>", unsafe_allow_html=True)
    col_g = st.columns([1, 4, 1])[1]
    with col_g:
        bar_color = "#ec4899" if pred == 1 else "#22c55e"
        st.markdown(f"""
        <div style='background:#f1f5f9; border-radius:8px; height:16px; overflow:hidden'>
          <div style='width:{prob*100:.1f}%; height:100%; background:{bar_color}; border-radius:8px'></div>
        </div>
        <div style='display:flex; justify-content:space-between; font-size:0.78rem; color:#64748b; margin-top:0.3rem'>
          <span>0% (ต่ำ)</span><span>{prob:.1%}</span><span>100% (สูง)</span>
        </div>
        """, unsafe_allow_html=True)

    # Input summary
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("📋 สรุปข้อมูลที่ป้อน"):
        input_df = pd.DataFrame({
            "Feature": ["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"],
            "ค่าที่ป้อน": [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age],
            "Scaled Value": [f"{v:.3f}" for v in input_scaled[0]],
        })
        st.table(input_df)

    st.markdown("""
    <div style='background:#fdf2f8; border:1px solid #f9a8d4; border-radius:10px; padding:1rem; margin-top:1rem; font-size:0.85rem; color:#831843'>
    ⚠️ <b>คำเตือน</b>: ผลการทำนายนี้มาจาก Neural Network Model เท่านั้น
    ไม่ใช่การวินิจฉัยทางการแพทย์ กรุณาปรึกษาแพทย์หากมีข้อกังวลเกี่ยวกับสุขภาพ
    </div>
    """, unsafe_allow_html=True)
```
