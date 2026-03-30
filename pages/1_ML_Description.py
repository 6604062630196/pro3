“””
Page 1 — Machine Learning (Ensemble) Model Description
“””
import streamlit as st
import pandas as pd
import numpy as np
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(**file**)))

st.set_page_config(page_title=“ML Model | HealthAI”, page_icon=“🤖”, layout=“wide”)

# Shared CSS (minimal re-include)

st.markdown(”””

<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.page-banner {
    background: linear-gradient(135deg, #312e81 0%, #4f46e5 60%, #6366f1 100%);
    border-radius: 18px; padding: 2.5rem; color: white; margin-bottom: 2rem;
}
.page-banner h1 { font-family:'DM Serif Display',serif; font-size:2rem; margin:0 0 0.4rem; }
.page-banner p  { opacity:0.85; font-size:1rem; margin:0; font-weight:300; }
.section-card {
    background: white; border-radius: 14px; padding: 1.8rem;
    box-shadow: 0 2px 14px rgba(0,0,0,0.06); margin-bottom: 1.5rem;
}
.section-card h3 { color: #1e1b4b; margin-top:0; font-size:1.15rem; }
.algo-chip {
    display:inline-block; background:#ede9fe; color:#4f46e5;
    border-radius:20px; padding:0.25rem 0.85rem;
    font-size:0.8rem; font-weight:600; margin:0.2rem;
}
.step-row { display:flex; align-items:flex-start; gap:1rem; margin:0.8rem 0; }
.step-num {
    background:#4f46e5; color:white; width:28px; height:28px; border-radius:50%;
    display:flex; align-items:center; justify-content:center;
    font-size:0.8rem; font-weight:700; flex-shrink:0;
}
.step-text { color:#374151; font-size:0.9rem; line-height:1.5; }
table { width:100%; border-collapse:collapse; font-size:0.88rem; }
th { background:#4f46e5; color:white; padding:0.6rem 1rem; text-align:left; }
td { padding:0.55rem 1rem; border-bottom:1px solid #f1f5f9; color:#374151; }
tr:nth-child(even) td { background:#fafafa; }
</style>

“””, unsafe_allow_html=True)

st.markdown(”””

<div class="page-banner">
  <div style='font-size:0.75rem; letter-spacing:2px; opacity:0.7; margin-bottom:0.5rem'>PAGE 1 / 4</div>
  <h1>🤖 Machine Learning — Ensemble Model</h1>
  <p>Heart Disease Prediction ด้วย Soft-Voting Ensemble Classifier ผสม 5 อัลกอริทึม</p>
</div>
""", unsafe_allow_html=True)

tabs = st.tabs([“📂 Dataset & Preprocessing”, “🧩 อัลกอริทึม”, “🔧 ขั้นตอนพัฒนา”, “📊 ผลการประเมิน”, “📚 References”])

# ── Tab 1: Dataset ────────────────────────────────────────────────────────────

with tabs[0]:
st.markdown(’<div class="section-card">’, unsafe_allow_html=True)
st.markdown(”### 📂 ที่มาของ Dataset”)
st.markdown(”””
**Heart Disease Dataset** มาจาก **UCI Machine Learning Repository**
(Cleveland Heart Disease dataset) รวบรวมโดย Dr. Robert Detrano จาก V.A. Medical Center
ประกอบด้วย **303 ตัวอย่าง** และ **13 features** สำหรับทำนายว่าผู้ป่วยมีโรคหัวใจหรือไม่

```
> 🔗 Source: [archive.ics.uci.edu/dataset/45/heart+disease](https://archive.ics.uci.edu/dataset/45/heart+disease)
""")
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown("### 🔍 Features ของ Dataset")
features_df = pd.DataFrame({
    "Feature": ["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal"],
    "คำอธิบาย": [
        "อายุ (ปี)", "เพศ (1=ชาย, 0=หญิง)", "ประเภทอาการเจ็บหน้าอก (0-3)",
        "ความดันโลหิตขณะพัก (mmHg)", "ระดับคอเลสเตอรอล (mg/dl)",
        "น้ำตาลในเลือดขณะอดอาหาร > 120 mg/dl (1=ใช่)", "ผล ECG ขณะพัก (0-2)",
        "อัตราการเต้นของหัวใจสูงสุด", "อาการเจ็บหน้าอกขณะออกกำลังกาย (1=ใช่)",
        "ST depression จาก exercise", "ความชันของ ST segment (0-2)",
        "จำนวนหลอดเลือดหลัก (0-4)", "Thalassemia (0-3)",
    ],
    "ประเภท": ["Numeric","Binary","Categorical","Numeric","Numeric","Binary","Categorical",
               "Numeric","Binary","Numeric","Categorical","Numeric","Categorical"],
    "Missing": ["ไม่มี","ไม่มี","ไม่มี","มี (2 rows)","มี (3 rows)","ไม่มี","ไม่มี",
                "ไม่มี","ไม่มี","ไม่มี","ไม่มี","ไม่มี","ไม่มี"],
})
st.table(features_df)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown("### 🔧 การเตรียมข้อมูล (Data Preprocessing)")

steps = [
    ("Missing Value Imputation", "แทนค่า missing ด้วย **median** ของแต่ละ column (robust ต่อ outlier)"),
    ("Duplicate Removal", "ลบ rows ที่ซ้ำกันออก เพื่อป้องกัน data leakage"),
    ("Outlier Capping (IQR)", "จำกัดค่า outlier ด้วย IQR method สำหรับ trestbps, chol, thalach"),
    ("Feature / Target Split", "แยก features (X) กับ target (y = target column)"),
    ("StandardScaler", "ปรับ scale ทุก feature ให้มี mean=0, std=1 เหมาะกับ SVM และ LR"),
    ("Train-Test Split (80/20)", "แบ่ง stratified 80% train / 20% test รักษาสัดส่วน class"),
]
for i, (title, desc) in enumerate(steps, 1):
    st.markdown(f"""
    <div class="step-row">
      <div class="step-num">{i}</div>
      <div class="step-text"><b>{title}</b><br>{desc}</div>
    </div>
    """, unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)
```

# ── Tab 2: Algorithm ──────────────────────────────────────────────────────────

with tabs[1]:
st.markdown(’<div class="section-card">’, unsafe_allow_html=True)
st.markdown(”### 🧩 Ensemble Learning — Soft Voting Classifier”)
st.markdown(”””
**Ensemble Learning** คือการรวมหลายโมเดลเข้าด้วยกันเพื่อได้ผลลัพธ์ที่ดีขึ้น
โดย **Soft Voting** จะนำ **probability** ที่แต่ละโมเดลทำนายมาเฉลี่ย แล้วเลือก class ที่ได้
probability สูงสุด ทำให้มีความเชื่อมั่นสูงกว่า Hard Voting
“””)
st.markdown(”””
<div style='background:#f5f3ff; border-radius:12px; padding:1.2rem; text-align:center; margin:1rem 0'>
<div style='font-size:0.85rem; color:#6d28d9; font-weight:600; margin-bottom:0.5rem'>SOFT VOTING FORMULA</div>
<code style='font-size:1rem; color:#4f46e5'>ŷ = argmax Σ wᵢ · P(y=c | xᵢ)</code>
</div>
“””, unsafe_allow_html=True)
st.markdown(’</div>’, unsafe_allow_html=True)

```
col1, col2 = st.columns(2)
algos = [
    ("Random Forest 🌳", "#dcfce7", "#16a34a",
     "สร้าง Decision Trees หลายต้น แต่ละต้นใช้ bootstrap sample และ random feature subset "
     "แล้ว aggregate ด้วย majority vote ลด variance และป้องกัน overfitting ได้ดี"),
    ("Gradient Boosting ⚡", "#fef3c7", "#d97706",
     "สร้าง weak learners (shallow trees) แบบ sequential โดยแต่ละต้นเรียนจาก residual errors "
     "ของต้นก่อนหน้า ใช้ gradient descent minimize loss function"),
    ("AdaBoost 🔄", "#ffe4e6", "#e11d48",
     "Adaptive Boosting ปรับ weight ของ samples ที่ classify ผิดให้สูงขึ้น "
     "ทำให้โมเดลถัดไปโฟกัสกับ hard examples เพิ่มขึ้น"),
    ("Logistic Regression 📈", "#dbeafe", "#1d4ed8",
     "โมเดล linear ใช้ sigmoid function แปลงผลออกเป็น probability "
     "เหมาะกับ binary classification ตีความได้ง่าย ทำงานเร็ว"),
    ("SVM (RBF Kernel) 🎯", "#f3e8ff", "#7c3aed",
     "หาค่า hyperplane ที่มี margin กว้างที่สุดในการแบ่ง class "
     "ใช้ RBF kernel จัดการ non-linear boundary ด้วย kernel trick"),
]
for i, (name, bg, color, desc) in enumerate(algos):
    col = col1 if i % 2 == 0 else col2
    with col:
        st.markdown(f"""
        <div style='background:{bg}; border-radius:12px; padding:1.2rem; margin-bottom:1rem'>
          <b style='color:{color}'>{name}</b>
          <p style='margin:0.5rem 0 0; font-size:0.88rem; color:#374151; line-height:1.5'>{desc}</p>
        </div>
        """, unsafe_allow_html=True)
```

# ── Tab 3: Development Steps ──────────────────────────────────────────────────

with tabs[2]:
st.markdown(’<div class="section-card">’, unsafe_allow_html=True)
st.markdown(”### 🔧 ขั้นตอนการพัฒนาโมเดล”)

```
dev_steps = [
    ("1. Exploratory Data Analysis (EDA)", [
        "ตรวจสอบ distribution ของแต่ละ feature ด้วย histogram และ boxplot",
        "วิเคราะห์ correlation matrix เพื่อหา feature ที่มีความสัมพันธ์กับ target",
        "ตรวจสอบ class imbalance (target=0 vs target=1)",
    ]),
    ("2. Data Preprocessing", [
        "แทนค่า missing ด้วย median imputation",
        "ลบ duplicates และ cap outliers ด้วย IQR",
        "StandardScaler สำหรับ normalize features",
    ]),
    ("3. Model Building", [
        "สร้าง base models: RF, GBM, AdaBoost, LR, SVM",
        "รวมใน VotingClassifier (soft voting) จาก sklearn",
        "กำหนด hyperparameters เริ่มต้นที่เหมาะสม",
    ]),
    ("4. Model Training", [
        "ฝึก ensemble บน training set (80%)",
        "ใช้ cross-validation (5-fold) ประเมิน generalization",
    ]),
    ("5. Evaluation", [
        "วัดผล Accuracy, Precision, Recall, F1, ROC-AUC บน test set",
        "วาด Confusion Matrix และ ROC Curve",
        "วิเคราะห์ Feature Importance จาก Random Forest",
    ]),
    ("6. Deployment", [
        "บันทึกโมเดลด้วย pickle (.pkl)",
        "โหลดใน Streamlit และรับ input จาก user",
    ]),
]
for title, items in dev_steps:
    with st.expander(title, expanded=False):
        for item in items:
            st.markdown(f"- {item}")
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown("### 🧬 Hyperparameters")
hp_df = pd.DataFrame({
    "โมเดล": ["Random Forest","Gradient Boosting","AdaBoost","Logistic Regression","SVM"],
    "Key Parameters": [
        "n_estimators=100, max_depth=5, random_state=42",
        "n_estimators=100, learning_rate=0.1, random_state=42",
        "n_estimators=50, random_state=42",
        "max_iter=1000, random_state=42",
        "kernel='rbf', probability=True, random_state=42",
    ],
})
st.table(hp_df)
st.markdown('</div>', unsafe_allow_html=True)
```

# ── Tab 4: Evaluation ─────────────────────────────────────────────────────────

with tabs[3]:
st.markdown(’<div class="section-card">’, unsafe_allow_html=True)
st.markdown(”### 📊 Metrics ที่ใช้ประเมิน”)

```
metrics_info = {
    "Accuracy": ("TP+TN / Total", "ความถูกต้องโดยรวม — เหมาะกับ balanced classes"),
    "Precision": ("TP / (TP+FP)", "ความแม่นยำ — สัดส่วนที่ทำนายว่าเป็นโรคแล้วถูกต้อง"),
    "Recall": ("TP / (TP+FN)", "Sensitivity — สัดส่วนผู้ป่วยจริงที่ตรวจจับได้ (สำคัญมากในทางการแพทย์)"),
    "F1-Score": ("2·P·R / (P+R)", "Harmonic mean ระหว่าง Precision และ Recall"),
    "ROC-AUC": ("Area under ROC curve", "วัดความสามารถแยก class โดยรวม — 1.0 = perfect"),
}
for metric, (formula, desc) in metrics_info.items():
    st.markdown(f"""
    <div style='display:flex; gap:1rem; padding:0.6rem 0; border-bottom:1px solid #f1f5f9'>
      <div style='width:120px; font-weight:700; color:#4f46e5'>{metric}</div>
      <div style='width:180px; font-family:monospace; color:#64748b; font-size:0.85rem'>{formula}</div>
      <div style='color:#374151; font-size:0.88rem'>{desc}</div>
    </div>
    """, unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown("### 📈 ผลลัพธ์ที่คาดหวัง")
st.markdown("""
ผลการทดสอบบน **Heart Disease Dataset** (test set 20%):

| Metric | Base RF | Ensemble |
|--------|---------|----------|
| Accuracy  | ~83% | **~87%** |
| Precision | ~84% | **~88%** |
| Recall    | ~82% | **~86%** |
| F1-Score  | ~83% | **~87%** |
| ROC-AUC   | ~91% | **~94%** |

> ✅ Ensemble ให้ผลที่ดีกว่า Single Model เนื่องจาก variance ลดลงและ bias สมดุลขึ้น
""")
st.markdown('</div>', unsafe_allow_html=True)
```

# ── Tab 5: References ─────────────────────────────────────────────────────────

with tabs[4]:
st.markdown(’<div class="section-card">’, unsafe_allow_html=True)
st.markdown(”### 📚 แหล่งอ้างอิง”)
refs = [
(“UCI Heart Disease Dataset”, “Detrano, R. et al. (1989). UCI ML Repository”, “https://archive.ics.uci.edu/dataset/45/heart+disease”),
(“Scikit-learn VotingClassifier”, “Pedregosa et al. (2011). JMLR 12, pp. 2825-2830.”, “https://scikit-learn.org/stable/modules/ensemble.html”),
(“Random Forest”, “Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5–32.”, “https://doi.org/10.1023/A:1010933404324”),
(“Gradient Boosting”, “Friedman, J.H. (2001). Greedy Function Approximation: A Gradient Boosting Machine.”, “https://doi.org/10.1214/aos/1013203451”),
(“AdaBoost”, “Freund, Y. & Schapire, R.E. (1997). A Decision-Theoretic Generalization of On-Line Learning.”, “https://doi.org/10.1006/jcss.1997.1504”),
(“Ensemble Methods Review”, “Sagi, O. & Rokach, L. (2018). Ensemble learning: A survey. WIREs Data Mining.”, “https://doi.org/10.1002/widm.1249”),
]
for title, citation, url in refs:
st.markdown(f”””
<div style='padding:0.75rem 0; border-bottom:1px solid #f1f5f9'>
<b style='color:#4f46e5'>{title}</b><br>
<span style='font-size:0.85rem; color:#64748b'>{citation}</span><br>
<a href='{url}' style='font-size:0.8rem; color:#6366f1'>{url}</a>
</div>
“””, unsafe_allow_html=True)
st.markdown(’</div>’, unsafe_allow_html=True)
