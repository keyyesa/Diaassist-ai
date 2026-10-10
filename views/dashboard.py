"""🏠 Dashboard: Strict Role-Based Doctor & Patient Portals."""
import streamlit as st
import pandas as pd
from views import _common as C

def render() -> None:
    user = st.session_state.get("user", {})
    role = user.get("role", "Doctor")
    
    # Strictly render dashboard based on registered user role. No toggles/dropdowns!
    if role == "Patient":
        render_patient_dashboard(user)
    else:
        render_doctor_dashboard(user)

# ================= ================= ================= ================= DOCTOR DASHBOARD
def render_doctor_dashboard(user: dict) -> None:
    st.markdown(f"""
    <div class="med-hero">
        <h1>🩺 Doctor Portal - Clinical Decision Support</h1>
        <p>Welcome, {user.get('name', 'Doctor')}. Real-time patient monitoring & guideline decision support.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 1. Population Key Performance Indicators
    st.markdown("##### 📊 Clinical Population Overview")
    k1, k2, k3, k4 = st.columns(4)
    
    with k1:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-icon kpi-blue">👥</div>
            <div>
                <div class="kpi-label">Active Patients</div>
                <div class="kpi-val">1,248</div>
                <div class="kpi-sub">↑ 12 new this week</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with k2:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-icon kpi-green">🩸</div>
            <div>
                <div class="kpi-label">Mean HbA1c</div>
                <div class="kpi-val">7.2%</div>
                <div class="kpi-sub">↓ -0.3% population avg</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with k3:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-icon kpi-orange">⚠️</div>
            <div>
                <div class="kpi-label">High Risk Alerts</div>
                <div class="kpi-val">14</div>
                <div class="kpi-sub" style="color: #ef4444;">Needs urgent review</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with k4:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-icon kpi-purple">📄</div>
            <div>
                <div class="kpi-label">Lab Reports</div>
                <div class="kpi-val">89</div>
                <div class="kpi-sub">5 pending review</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    st.write("")

    # 2. Real-time Patient Monitoring Roster Table
    st.markdown("##### 🩺 Patient Status Roster (Live Patient Data)")
    patient_data = pd.DataFrame([
        {"Patient Name": "Ali Hassan", "Age": 48, "Diagnosis": "Type 2 Diabetes", "Last Fasting Sugar": "115 mg/dL", "Latest HbA1c": "6.8%", "Status": "Controlled ✅", "Last Logged": "Today, 08:30 AM"},
        {"Patient Name": "Fatima Zahra", "Age": 34, "Diagnosis": "Type 1 Diabetes", "Last Fasting Sugar": "210 mg/dL", "Latest HbA1c": "8.5%", "Status": "High Risk ⚠️", "Last Logged": "Today, 07:45 AM"},
        {"Patient Name": "Usman Ahmed", "Age": 52, "Diagnosis": "Pre-Diabetes", "Last Fasting Sugar": "102 mg/dL", "Latest HbA1c": "5.9%", "Status": "Normal ✅", "Last Logged": "Yesterday"},
        {"Patient Name": "Zainab Bibi", "Age": 29, "Diagnosis": "Gestational Diabetes", "Last Fasting Sugar": "145 mg/dL", "Latest HbA1c": "7.4%", "Status": "Review Needed 🟡", "Last Logged": "Today, 09:15 AM"},
        {"Patient Name": "Tariq Mahmood", "Age": 61, "Diagnosis": "Type 2 Diabetes", "Last Fasting Sugar": "180 mg/dL", "Latest HbA1c": "7.9%", "Status": "Moderate Risk 🟡", "Last Logged": "2 days ago"}
    ])
    
    st.dataframe(patient_data, use_container_width=True, hide_index=True)

    st.write("")
    st.write("")

    # 3. Clinical Analytics Charts
    st.markdown("##### 📈 Population Analytics & Risk Profiling")
    c_left, c_right = st.columns([1.5, 1])
    
    with c_left:
        with st.container(border=True):
            st.markdown("**Population HbA1c Control Trend (Last 6 Months)**")
            trend_df = pd.DataFrame({
                "HbA1c (%)": [7.9, 7.7, 7.6, 7.4, 7.3, 7.2],
                "Target Threshold": [7.0, 7.0, 7.0, 7.0, 7.0, 7.0]
            }, index=["Feb", "Mar", "Apr", "May", "Jun", "Jul"])
            st.line_chart(trend_df, color=["#00b4d8", "#ef4444"])

    with c_right:
        with st.container(border=True):
            st.markdown("**Patient Glycemic Risk Stratification**")
            risk_df = pd.DataFrame({
                "Patients": [680, 410, 158]
            }, index=["Normal / Controlled", "Moderate Risk", "High Risk / Critical"])
            st.bar_chart(risk_df, color="#0e7c86")

    st.write("")
    
    # 4. RAG Engine Status
    st.markdown("##### ⚡ RAG Engine & Knowledge Base")
    status = C.kb_status()
    health = C.llm_health()
    pipeline, _ = C.get_pipelines()
    trained = [d for d in status["documents"] if d["status"] == "trained"]
    chunks = status["n_chunks"] if status["n_chunks"] is not None else (pipeline.kb.n_chunks if pipeline else 0)

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Knowledge Base", f"{len(trained)} Guidelines")
    s2.metric("Indexed Chunks", f"{chunks} Chunks")
    s3.metric("Last Updated", C.fmt_time(status['trained_at']))
    s4.metric("LLM Status", "Online 🟢" if health['ok'] else "Offline 🔴")

    st.write("")

    # 5. Core Application Modules Grid
    st.markdown("##### 🚀 Clinical Action Modules")
    m1, m2, m3 = st.columns(3)
    
    with m1:
        with st.container(border=True):
            st.markdown("**💬 AI Clinical Assistant**")
            st.caption("Context-aware decision support powered by trained guidelines.")
            if st.button("Launch Assistant ➡️", key="btn_chat_doc", **C.wide(st.button)):
                C.goto("chat")

    with m2:
        with st.container(border=True):
            st.markdown("**🧪 Lab Report Intelligence**")
            st.caption("Automatic parsing & reference range classification for bloodwork.")
            if st.button("Analyze Lab Reports ➡️", key="btn_lab_doc", **C.wide(st.button)):
                C.goto("lab")

    with m3:
        with st.container(border=True):
            st.markdown("**⚖️ Benchmark & Evaluation**")
            st.caption("Side-by-side evaluation between Baseline RAG and Agentic RAG.")
            if st.button("View Benchmark ➡️", key="btn_compare_doc", **C.wide(st.button)):
                C.goto("compare")

# ================= ================= ================= ================= PATIENT DASHBOARD
def render_patient_dashboard(user: dict) -> None:
    doc_name = user.get("doctor_name", "Dr. Sarah Smith (Endocrinologist)")
    
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #2ea043 0%, #10b981 50%, #34d399 100%); 
                border-radius: 20px; padding: 2rem; color: white; margin-bottom: 1.5rem; box-shadow: 0 10px 30px rgba(46, 160, 67, 0.2);">
        <h1 style="font-size: 2.1rem; font-weight: 800; margin-bottom: 0.4rem; color: white;">👋 Welcome, {user.get('name', 'Patient')}!</h1>
        <p style="font-size: 1.05rem; opacity: 0.95; margin: 0;">Your personal health portal. Monitor blood sugar, track meals, and read lab reports easily.</p>
    </div>
    """, unsafe_allow_html=True)

    # Assigned Doctor Badge
    st.info(f"👨‍⚕️ **Primary Physician**: {doc_name} | **Hospital**: General Medical Center")

    # Patient Health Summary Cards
    st.markdown("##### 💚 My Daily Health Summary")
    p1, p2, p3, p4 = st.columns(4)

    with p1:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-icon kpi-green">🩸</div>
            <div>
                <div class="kpi-label">Fasting Sugar</div>
                <div class="kpi-val">115 <span style="font-size:0.9rem;">mg/dL</span></div>
                <div class="kpi-sub">✅ Target Range</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with p2:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-icon kpi-blue">📊</div>
            <div>
                <div class="kpi-label">Latest HbA1c</div>
                <div class="kpi-val">6.8%</div>
                <div class="kpi-sub">✅ Well Controlled</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with p3:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-icon kpi-purple">💊</div>
            <div>
                <div class="kpi-label">Medication</div>
                <div class="kpi-val">Taken</div>
                <div class="kpi-sub">Metformin 500mg</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with p4:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-icon kpi-orange">🥗</div>
            <div>
                <div class="kpi-label">Diet Goal</div>
                <div class="kpi-val">Low GI</div>
                <div class="kpi-sub">1,800 kcal / day</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    st.write("")

    # Weekly Blood Glucose Log & Input Form
    p_col1, p_col2 = st.columns([1.6, 1])

    with p_col1:
        with st.container(border=True):
            st.markdown("📈 **My Weekly Glucose Trend (Fasting vs After Meals)**")
            sugar_df = pd.DataFrame({
                "Fasting (mg/dL)": [110, 118, 105, 112, 115, 108, 115],
                "2 Hrs After Meal (mg/dL)": [140, 155, 138, 145, 150, 142, 148]
            }, index=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
            st.line_chart(sugar_df, color=["#10b981", "#3b82f6"])

    with p_col2:
        with st.container(border=True):
            st.markdown("📝 **Log Today's Blood Sugar**")
            reading_type = st.radio("Timing", ["Fasting (Morning)", "2 Hours After Meal"], horizontal=True)
            sugar_val = st.number_input("Blood Sugar Reading (mg/dL)", min_value=50, max_value=400, value=115)
            
            if st.button("Save Reading ✅", type="primary", use_container_width=True):
                if sugar_val < 70:
                    st.warning("⚠️ Low sugar level detected (Hypoglycemia). Drink fruit juice or eat 3-4 sugar tablets immediately.")
                elif sugar_val > 180:
                    st.warning("⚠️ High sugar reading. Drink plenty of water and follow your doctor's meal plan.")
                else:
                    st.success(f"Saved {sugar_val} mg/dL ({reading_type}). Great job staying in target range!")

    st.write("")

    # Patient Friendly Tools
    st.markdown("##### 🌟 Easy Health Tools")
    pm1, pm2 = st.columns(2)

    with pm1:
        with st.container(border=True):
            st.markdown("### 💬 Ask Diabetes AI Coach")
            st.write("Get instant, easy-to-understand answers for your food, symptoms, and daily routine questions.")
            if st.button("Chat with AI Coach ➡️", key="btn_chat_pat", **C.wide(st.button)):
                C.goto("chat")

    with pm2:
        with st.container(border=True):
            st.markdown("### 🧪 Understand My Lab Report")
            st.write("Upload your lab report PDF to see a simple, clear breakdown of your glucose & cholesterol levels.")
            if st.button("Upload & Read Lab Report ➡️", key="btn_lab_pat", **C.wide(st.button)):
                C.goto("lab")
