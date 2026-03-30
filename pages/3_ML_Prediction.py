“””
Page 3 — ML Ensemble Model Demo (Heart Disease Prediction)
“””
import streamlit as st
import pandas as pd
import numpy as np
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(**file**)))

st.set_page_config(page_title=“ML Prediction | HealthAI”, page_icon=“⚡”, layout=“wide”)

st.markdown(”””

<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.page-banner {
    background: linear-gradient(135deg, #312e81 0%, #4338ca 60%, #6366f1 100%);
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
    background: linear-gradient(135deg, #fff1f2, #ffe4e6);
    border: 2px solid #fca5a5; border-radius: 16px; padding: 2rem; text-align:center;
}
.result-icon { font-size: 3rem; margin-bottom: 0.5rem; }
.result-title { font-family:'DM Serif Display',serif; font-size:1.8rem; margin:0 0 0.3rem; }
.prob-bar {
    height: 12px; border-radius: 6px;
    background: linear-gradient(90deg, #4ade80, #ef4444);
    margin: 0.5rem 0;
}
.stSlider > div > div { accent-color: #6366f1; }
.stButton > button {
    background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
    color: white !important; border: none; border-radius: 10px;
    padding: 0.7rem 2rem; font-weight: 600; font-size: 1rem;
    box-shadow: 0 4px 14px rgba(79,70,229,0.4);
}
</style>

“””, unsafe_allow_html=True)

st.markdown(”””

<div class="page-banner">
  <div style='font-size:0.75rem; letter-spacing:2px; opacity:0.7; margin-bottom:0.5rem'>PAGE 3 / 4</div>
  <h1>⚡ ML Ensemble — Heart Disease Prediction</h1>
  <p>ป้อนข้อมูลสุขภาพของคุณเพื่อรับการทำนายความเสี่ยงโรคหัวใจ</p>
</div>
""", unsafe_allow_html=True)

# ── Load / Train Model ────────────────────────────────────────────────────────

@st.cache_resource
def get_heart_model():
from utils.data_prep import prepare_heart_data
from utils.models import train_ensemble, load_ensemble

```
data = prepare_heart_data()
model = load_ensemble("heart")
if model is None:
    model, metrics = train_ensemble(
        data["X_train"], data["y_train"],
        data["X_test"],  data["y_test"],
        name="heart"
    )
else:
    from utils.models import evaluate_model
    metrics = evaluate_model(model, data["X_test"], data["y_test"])
return model, data, metrics
```

with st.spinner(“⏳ กำลังโหลด/ฝึกโมเดล กรุณารอสักครู่…”):
try:
model, data, metrics = get_heart_model()
model_ok = model is not None
except Exception as e:
st.error(f”เกิดข้อผิดพลาด: {e}”)
model_ok = False

# ── Model Metrics ─────────────────────────────────────────────────────────────

if model_ok:
st.markdown(”### 📊 ประสิทธิภาพโมเดล (Test Set)”)
c1, c2, c3, c4, c5 = st.columns(5)
for col, (label, key, color) in zip(
[c1, c2, c3, c4, c5],
[(“Accuracy”,“accuracy”,”#6366f1”),(“Precision”,“precision”,”#10b981”),
(“Recall”,“recall”,”#f59e0b”),(“F1-Score”,“f1”,”#ec4899”),(“ROC-AUC”,“roc_auc”,”#3b82f6”)]
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

# ── Confusion Matrix ──────────────────────────────────────────────────────
with st.expander("📊 Confusion Matrix & Trained Model Info"):
    cm = metrics.get("confusion_matrix", [[0,0],[0,0]])
    col_cm, col_info = st.columns([1, 1])
    with col_cm:
        cm_df = pd.DataFrame(
            cm,
            index=["Actual: No Disease","Actual: Disease"],
            columns=["Predicted: No Disease","Predicted: Disease"],
        )
        st.dataframe(cm_df, use_container_width=True)
    with col_info:
        st.markdown(f"""
        **โมเดล**: Soft Voting Ensemble  
        **Base models**: RF, GBM, AdaBoost, LR, SVM  
        **Training samples**: {len(data['X_train'])}  
        **Test samples**: {len(data['X_test'])}  
        **Features**: {len(data['feature_names'])}  
        """)

# ── Feature importance ────────────────────────────────────────────────────
with st.expander("🌳 Feature Importance (Random Forest component)"):
    try:
        rf_model = model.named_estimators_["rf"]
        importances = pd.Series(rf_model.feature_importances_, index=data["feature_names"]).sort_values(ascending=False)
        st.bar_chart(importances)
    except:
        st.info("Feature importance ไม่สามารถแสดงได้")
```

# ── Prediction Input ──────────────────────────────────────────────────────────

st.markdown(”—”)
st.markdown(”### 🔬 ทดสอบการทำนาย”)
st.markdown(“กรอกข้อมูลสุขภาพด้านล่าง แล้วกด **ทำนาย** เพื่อดูผลลัพธ์”)

col_l, col_r = st.columns(2)
with col_l:
st.markdown(’<div class="card">’, unsafe_allow_html=True)
st.markdown(”**👤 ข้อมูลทั่วไป**”)
age     = st.slider(“อายุ (ปี)”, 20, 80, 54)
sex     = st.selectbox(“เพศ”, [“ชาย (1)”, “หญิง (0)”])
sex_val = 1 if “ชาย” in sex else 0
cp      = st.selectbox(“อาการเจ็บหน้าอก (cp)”, [
“0 — Typical Angina”, “1 — Atypical Angina”,
“2 — Non-anginal Pain”, “3 — Asymptomatic”
])
cp_val  = int(cp[0])
trestbps = st.slider(“ความดันโลหิตขณะพัก (mmHg)”, 80, 200, 130)
chol     = st.slider(“คอเลสเตอรอล (mg/dl)”, 100, 400, 240)
fbs      = st.selectbox(“น้ำตาลในเลือด > 120 mg/dl”, [“ไม่ใช่ (0)”, “ใช่ (1)”])
fbs_val  = 1 if “ใช่” in fbs else 0
st.markdown(’</div>’, unsafe_allow_html=True)

with col_r:
st.markdown(’<div class="card">’, unsafe_allow_html=True)
st.markdown(”**🫀 ผลตรวจ ECG & Exercise**”)
restecg  = st.selectbox(“ผล ECG ขณะพัก”, [“0 — Normal”, “1 — ST-T abnormality”, “2 — LVH”])
restecg_val = int(restecg[0])
thalach  = st.slider(“อัตราการเต้นหัวใจสูงสุด”, 60, 220, 150)
exang    = st.selectbox(“เจ็บหน้าอกขณะออกกำลังกาย”, [“ไม่มี (0)”, “มี (1)”])
exang_val = 1 if “มี” in exang else 0
oldpeak  = st.slider(“ST Depression (oldpeak)”, 0.0, 6.0, 1.0, 0.1)
slope    = st.selectbox(“ความชัน ST (slope)”, [“0 — Upsloping”, “1 — Flat”, “2 — Downsloping”])
slope_val = int(slope[0])
ca       = st.slider(“จำนวนหลอดเลือดหลัก (ca)”, 0, 4, 0)
thal     = st.selectbox(“Thalassemia (thal)”, [“1 — Normal”, “2 — Fixed Defect”, “3 — Reversible Defect”])
thal_val = int(thal[0])
st.markdown(’</div>’, unsafe_allow_html=True)

# Predict button

st.markdown(”<br>”, unsafe_allow_html=True)
col_btn = st.columns([2, 1, 2])[1]
with col_btn:
predict_btn = st.button(“🔮 ทำนายความเสี่ยง”, use_container_width=True)

if predict_btn and model_ok:
input_raw = np.array([[age, sex_val, cp_val, trestbps, chol, fbs_val,
restecg_val, thalach, exang_val, oldpeak,
slope_val, ca, thal_val]])
input_scaled = data[“scaler”].transform(input_raw)

```
prob    = model.predict_proba(input_scaled)[0][1]
pred    = model.predict(input_scaled)[0]

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("### 🎯 ผลการทำนาย")

if pred == 1:
    st.markdown(f"""
    <div class="result-risk">
      <div class="result-icon">⚠️</div>
      <div class="result-title" style='color:#dc2626'>มีความเสี่ยงโรคหัวใจ</div>
      <p style='color:#7f1d1d; margin:0.5rem 0'>โมเดลพบสัญญาณความเสี่ยง ควรปรึกษาแพทย์เพิ่มเติม</p>
      <div style='font-size:2rem; font-weight:800; color:#dc2626'>{prob:.1%}</div>
      <div style='font-size:0.85rem; color:#9f1239'>Probability of Heart Disease</div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown(f"""
    <div class="result-safe">
      <div class="result-icon">✅</div>
      <div class="result-title" style='color:#16a34a'>ความเสี่ยงต่ำ</div>
      <p style='color:#14532d; margin:0.5rem 0'>โมเดลไม่พบสัญญาณความเสี่ยงโรคหัวใจที่น่ากังวล</p>
      <div style='font-size:2rem; font-weight:800; color:#16a34a'>{prob:.1%}</div>
      <div style='font-size:0.85rem; color:#166534'>Probability of Heart Disease</div>
    </div>
    """, unsafe_allow_html=True)

# Probability bar
st.markdown("<br>", unsafe_allow_html=True)
col_bar = st.columns([1, 6, 1])[1]
with col_bar:
    st.markdown("**ระดับความเสี่ยง**")
    bar_color = "#ef4444" if pred == 1 else "#22c55e"
    st.markdown(f"""
    <div style='background:#f1f5f9; border-radius:8px; height:16px; overflow:hidden'>
      <div style='width:{prob*100:.1f}%; height:100%; background:{bar_color}; border-radius:8px; transition:width 0.5s'></div>
    </div>
    <div style='display:flex; justify-content:space-between; font-size:0.78rem; color:#64748b; margin-top:0.3rem'>
      <span>0% (ต่ำ)</span><span>{prob:.1%}</span><span>100% (สูง)</span>
    </div>
    """, unsafe_allow_html=True)

# Each model vote
st.markdown("<br>", unsafe_allow_html=True)
with st.expander("🗳️ ผลโหวตของแต่ละโมเดล"):
    try:
        votes = []
        for name, est in model.named_estimators_.items():
            p = est.predict_proba(input_scaled)[0][1]
            votes.append({"โมเดล": name.upper(), "Probability": f"{p:.1%}", "ผล": "🔴 มีความเสี่ยง" if p >= 0.5 else "🟢 ไม่มีความเสี่ยง"})
        st.table(pd.DataFrame(votes))
    except:
        st.info("ไม่สามารถแสดงผลโหวตแต่ละโมเดลได้")

st.markdown("""
<div style='background:#fffbeb; border:1px solid #fde68a; border-radius:10px; padding:1rem; margin-top:1rem; font-size:0.85rem; color:#92400e'>
⚠️ <b>คำเตือน</b>: ผลการทำนายนี้เป็นเพียงการประมาณการจาก Machine Learning ไม่ใช่การวินิจฉัยทางการแพทย์
กรุณาปรึกษาแพทย์ผู้เชี่ยวชาญก่อนตัดสินใจเกี่ยวกับสุขภาพ
</div>
""", unsafe_allow_html=True)
```

elif predict_btn and not model_ok:
st.error(“ไม่สามารถโหลดโมเดลได้ กรุณาตรวจสอบ dependencies”)
