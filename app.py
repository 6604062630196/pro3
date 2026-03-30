
import streamlit as st

st.set_page_config(
page_title="HealthAI Platform",
page_icon="🏥",
layout="wide",
initial_sidebar_state="expanded",
)

# ─── Global CSS ──────────────────────────────────────────────────────────────

st.markdown("""

<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');

  html, body, [class*="css"] {
      font-family: 'DM Sans', sans-serif;
  }

  /* Sidebar */
  section[data-testid="stSidebar"] {
      background: linear-gradient(160deg, #0f1923 0%, #1a2e44 100%);
      border-right: 1px solid #2a3f55;
  }
  section[data-testid="stSidebar"] * { color: #c8d8e8 !important; }
  section[data-testid="stSidebar"] .stMarkdown h1,
  section[data-testid="stSidebar"] .stMarkdown h2,
  section[data-testid="stSidebar"] .stMarkdown h3 { color: #7dc4e4 !important; }

  /* Main background */
  .main .block-container {
      background: #f8f9fc;
      padding-top: 2rem;
  }

  /* Hero section */
  .hero-card {
      background: linear-gradient(135deg, #0f1923 0%, #1a3a5c 50%, #0d4f8c 100%);
      border-radius: 20px;
      padding: 3rem 3.5rem;
      color: white;
      margin-bottom: 2rem;
      position: relative;
      overflow: hidden;
  }
  .hero-card::before {
      content: '';
      position: absolute;
      top: -50%;
      right: -10%;
      width: 500px;
      height: 500px;
      background: radial-gradient(circle, rgba(100,180,255,0.15) 0%, transparent 70%);
      border-radius: 50%;
  }
  .hero-title {
      font-family: 'DM Serif Display', serif;
      font-size: 2.8rem;
      margin: 0 0 0.5rem 0;
      line-height: 1.1;
  }
  .hero-subtitle {
      font-size: 1.15rem;
      opacity: 0.8;
      margin: 0;
      font-weight: 300;
  }
  .hero-badge {
      display: inline-block;
      background: rgba(100,180,255,0.2);
      border: 1px solid rgba(100,180,255,0.4);
      color: #7dc4e4;
      padding: 0.25rem 0.75rem;
      border-radius: 20px;
      font-size: 0.8rem;
      font-weight: 500;
      margin-bottom: 1rem;
      letter-spacing: 1px;
      text-transform: uppercase;
  }

  /* Metric cards */
  .metric-grid { display: flex; gap: 1rem; flex-wrap: wrap; margin: 1.5rem 0; }
  .metric-card {
      background: white;
      border-radius: 14px;
      padding: 1.2rem 1.5rem;
      flex: 1;
      min-width: 160px;
      box-shadow: 0 2px 12px rgba(0,0,0,0.06);
      border-left: 4px solid;
  }
  .metric-card.blue  { border-color: #3b82f6; }
  .metric-card.teal  { border-color: #14b8a6; }
  .metric-card.amber { border-color: #f59e0b; }
  .metric-card.rose  { border-color: #f43f5e; }
  .metric-val { font-size: 1.8rem; font-weight: 700; line-height: 1; }
  .metric-lbl { font-size: 0.8rem; color: #64748b; margin-top: 0.25rem; font-weight: 500; }

  /* Feature cards on home page */
  .feature-card {
      background: white;
      border-radius: 16px;
      padding: 1.8rem;
      box-shadow: 0 2px 16px rgba(0,0,0,0.06);
      height: 100%;
      transition: transform 0.2s;
      border-top: 4px solid;
  }
  .feature-card.ml    { border-color: #6366f1; }
  .feature-card.nn    { border-color: #ec4899; }
  .feature-card.data  { border-color: #10b981; }
  .feature-card.demo  { border-color: #f59e0b; }
  .feature-icon { font-size: 2.2rem; margin-bottom: 0.75rem; }
  .feature-title { font-weight: 700; font-size: 1.1rem; color: #1e293b; margin-bottom: 0.4rem; }
  .feature-desc { color: #64748b; font-size: 0.9rem; line-height: 1.5; }

  /* Page header */
  .page-header {
      border-bottom: 2px solid #e2e8f0;
      padding-bottom: 1rem;
      margin-bottom: 2rem;
  }
  .page-header h1 {
      font-family: 'DM Serif Display', serif;
      color: #0f172a;
      font-size: 2rem;
      margin: 0;
  }
  .page-header p { color: #64748b; margin: 0.3rem 0 0 0; }

  /* Info boxes */
  .info-box {
      background: #eff6ff;
      border: 1px solid #bfdbfe;
      border-radius: 10px;
      padding: 1rem 1.25rem;
      margin: 1rem 0;
      color: #1e40af;
      font-size: 0.9rem;
  }
  .warn-box {
      background: #fffbeb;
      border: 1px solid #fde68a;
      border-radius: 10px;
      padding: 1rem 1.25rem;
      margin: 1rem 0;
      color: #92400e;
      font-size: 0.9rem;
  }

  /* Streamlit elements override */
  .stButton > button {
      background: linear-gradient(135deg, #3b82f6, #1d4ed8);
      color: white !important;
      border: none;
      border-radius: 10px;
      padding: 0.6rem 1.5rem;
      font-weight: 600;
      font-size: 0.95rem;
      transition: all 0.2s;
      box-shadow: 0 4px 14px rgba(59,130,246,0.35);
  }
  .stButton > button:hover {
      transform: translateY(-1px);
      box-shadow: 0 6px 20px rgba(59,130,246,0.45);
  }
  div[data-testid="metric-container"] {
      background: white;
      border-radius: 12px;
      padding: 1rem;
      box-shadow: 0 2px 8px rgba(0,0,0,0.06);
  }
</style>

""", unsafe_allow_html=True)

# ─── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
st.markdown("""
<div style='text-align:center; padding: 1.5rem 0 1rem'>
<div style='font-size:2.8rem'>🏥</div>
<div style='font-family:"DM Serif Display",serif; font-size:1.4rem; color:#7dc4e4; margin-top:0.3rem'>HealthAI</div>
<div style='font-size:0.75rem; color:#64748b; margin-top:0.25rem; letter-spacing:1px'>PREDICTION PLATFORM</div>
</div>
<hr style='border-color:#2a3f55; margin:0.5rem 0 1rem'>
""", unsafe_allow_html=True)

```
st.markdown("### 📚 เอกสารโมเดล")
st.page_link("pages/1_ML_Description.py",   label="🤖 Machine Learning Model",  icon=None)
st.page_link("pages/2_NN_Description.py",    label="🧠 Neural Network Model",    icon=None)

st.markdown("### 🔬 ทดสอบโมเดล")
st.page_link("pages/3_ML_Prediction.py",     label="⚡ ทดสอบ ML (Heart Disease)", icon=None)
st.page_link("pages/4_NN_Prediction.py",     label="💡 ทดสอบ NN (Diabetes)",     icon=None)

st.markdown("---")
st.markdown("""
<div style='font-size:0.75rem; color:#475569; line-height:1.7'>
<b style='color:#7dc4e4'>Dataset Sources</b><br>
• Heart Disease — UCI ML Repository<br>
• Diabetes — Pima Indians (Kaggle)<br><br>
<b style='color:#7dc4e4'>นักศึกษา</b><br>
วิชา Intelligent Systems<br>
ภาคเรียนที่ 2/2567
</div>
""", unsafe_allow_html=True)
```

# ─── Home Page ───────────────────────────────────────────────────────────────

st.markdown("""

<div class="hero-card">
  <div class="hero-badge">🏥 Health AI Platform</div>
  <h1 class="hero-title">HealthAI<br>Prediction Platform</h1>
  <p class="hero-subtitle">
    ระบบทำนายความเสี่ยงโรคด้วย Machine Learning &amp; Neural Network<br>
    วิเคราะห์ข้อมูลสุขภาพอย่างแม่นยำด้วย Ensemble Model และ Deep Learning
  </p>
</div>
""", unsafe_allow_html=True)

# Feature cards

col1, col2, col3, col4 = st.columns(4)
cards = [
("ml",   "🤖", "ML Ensemble Model",      "Voting Classifier ผสม Random Forest, GBM, AdaBoost, LR, SVM ทำนายโรคหัวใจ"),
("nn",   "🧠", "Neural Network Model",   "MLP ออกแบบเอง 3 hidden layers พร้อม BatchNorm + Dropout ทำนายเบาหวาน"),
("data", "📊", "2 Health Datasets",       "Heart Disease (UCI) และ Diabetes (Pima Indians) ผ่านการ preprocess แล้ว"),
("demo", "⚡", "Live Prediction Demo",   "ป้อนข้อมูลสุขภาพและรับผลการทำนายพร้อมความน่าจะเป็นแบบ Real-time"),
]
for col, (cls, icon, title, desc) in zip([col1, col2, col3, col4], cards):
with col:
st.markdown(f"""
<div class="feature-card {cls}">
<div class="feature-icon">{icon}</div>
<div class="feature-title">{title}</div>
<div class="feature-desc">{desc}</div>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Stats row

st.markdown(”### 📈 สรุปโปรเจกต์”)
c1, c2, c3, c4 = st.columns(4)
c1.metric("📁 Datasets",    "2 ชุด",   "Heart + Diabetes")
c2.metric("🤖 ML Models",   "5 Base",  "Voting Ensemble")
c3.metric("🧠 NN Layers",   "3 Hidden","128→64→32")
c4.metric("📋 Total Pages", "4 หน้า",  "Describe + Demo")

st.markdown("—")

# Dataset overview

st.markdown("### 📂 Dataset Overview")
col_a, col_b = st.columns(2)

with col_a:
st.markdown("""
<div style='background:white; border-radius:14px; padding:1.5rem; box-shadow:0 2px 12px rgba(0,0,0,0.06); border-top:4px solid #ef4444'>
<h4 style='margin:0 0 0.75rem; color:#0f172a'>❤️ Heart Disease Dataset</h4>
<table style='width:100%; font-size:0.85rem; border-collapse:collapse'>
<tr><td style='color:#64748b; padding:0.2rem 0'>ที่มา</td><td><b>UCI Machine Learning Repository</b></td></tr>
<tr><td style='color:#64748b; padding:0.2rem 0'>Rows</td><td><b>303 records</b></td></tr>
<tr><td style='color:#64748b; padding:0.2rem 0'>Features</td><td><b>13 features</b></td></tr>
<tr><td style='color:#64748b; padding:0.2rem 0'>Target</td><td><b>target (0=ไม่เป็น, 1=เป็น)</b></td></tr>
<tr><td style='color:#64748b; padding:0.2rem 0'>ปัญหา</td><td><b>Missing values ใน trestbps, chol</b></td></tr>
</table>
</div>
""", unsafe_allow_html=True)

with col_b:
st.markdown("""
<div style='background:white; border-radius:14px; padding:1.5rem; box-shadow:0 2px 12px rgba(0,0,0,0.06); border-top:4px solid #3b82f6'>
<h4 style='margin:0 0 0.75rem; color:#0f172a'>💉 Diabetes Dataset</h4>
<table style='width:100%; font-size:0.85rem; border-collapse:collapse'>
<tr><td style='color:#64748b; padding:0.2rem 0'>ที่มา</td><td><b>Pima Indians / Kaggle</b></td></tr>
<tr><td style='color:#64748b; padding:0.2rem 0'>Rows</td><td><b>768 records</b></td></tr>
<tr><td style='color:#64748b; padding:0.2rem 0'>Features</td><td><b>8 features</b></td></tr>
<tr><td style='color:#64748b; padding:0.2rem 0'>Target</td><td><b>Outcome (0=ไม่เป็น, 1=เป็น)</b></td></tr>
<tr><td style='color:#64748b; padding:0.2rem 0'>ปัญหา</td><td><b>Zero-values แทน missing (Glucose, BMI)</b></td></tr>
</table>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.info("👈 ใช้เมนูด้านซ้ายเพื่อดูรายละเอียดโมเดลหรือทดสอบการทำนาย")
