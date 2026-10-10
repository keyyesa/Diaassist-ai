"""
DiaAssist AI - Streamlit front end.
Run with: streamlit run app.py
"""
import streamlit as st

st.set_page_config(page_title="DiaAssist AI", page_icon="🩺", layout="wide", initial_sidebar_state="expanded")

from views import _common as C  # noqa: E402
from views import about, auth, chat, compare, dashboard, evaluation, knowledge_base, lab_report, settings  # noqa: E402

C.init_state()
C.inject_css()

if not st.session_state.get("logged_in", False):
    auth.render()
else:
    user_role = st.session_state.get("user", {}).get("role", "Doctor")

    PAGES = {
        "dashboard": st.Page(dashboard.render, title="Dashboard", icon="🏠", url_path="dashboard", default=True),
        "chat": st.Page(chat.render, title="AI Assistant" if user_role == "Patient" else "Chat", icon="💬", url_path="chat"),
        "lab": st.Page(lab_report.render, title="My Lab Reports" if user_role == "Patient" else "Lab Reports", icon="🧪", url_path="lab-reports"),
        "kb": st.Page(knowledge_base.render, title="Knowledge Base", icon="📚", url_path="knowledge-base"),
        "compare": st.Page(compare.render, title="Compare RAG", icon="⚖️", url_path="compare"),
        "evaluation": st.Page(evaluation.render, title="Evaluation", icon="📊", url_path="evaluation"),
        "settings": st.Page(settings.render, title="Settings", icon="⚙️", url_path="settings"),
        "about": st.Page(about.render, title="About", icon="ℹ️", url_path="about"),
    }
    st.session_state.pages = PAGES

    if user_role == "Patient":
        navigation = st.navigation({
            "Home": [PAGES["dashboard"]],
            "Patient Tools": [PAGES["chat"], PAGES["lab"]],
            "Info": [PAGES["about"]],
        })
    else:
        navigation = st.navigation({
            "Home": [PAGES["dashboard"]],
            "Assistant": [PAGES["chat"], PAGES["lab"]],
            "Knowledge": [PAGES["kb"]],
            "Analysis": [PAGES["compare"], PAGES["evaluation"]],
            "System": [PAGES["settings"], PAGES["about"]],
        })

    C.render_sidebar()
    C.show_flash()
    C.handle_training_request()  # runs a requested training (upload / Train button) with a progress bar
    C.get_pipelines()            # load the trained index + embedding model once, up front
    navigation.run()


