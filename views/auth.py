"""🔐 Auth View: Clean Login and Signup for DiaAssist AI with strict role assignment."""
import streamlit as st

def render() -> None:
    # Centered container for login card
    col1, col2, col3 = st.columns([1, 2.2, 1])
    
    with col2:
        st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem; margin-top: 1rem;">
            <h1 style="color: #0e7c86; font-size: 2.2rem; font-weight: 800; margin-bottom: 0.2rem;">🩺 DiaAssist AI</h1>
            <p style="color: #64748b; font-size: 1rem;">Clinical & Patient Care Portal</p>
        </div>
        """, unsafe_allow_html=True)

        tab_login, tab_signup = st.tabs(["🔐 Log In", "📝 Create Account"])
        
        # Seed default accounts if empty
        users = st.session_state.get("registered_users", {})
        if "doctor@diaassist.ai" not in users:
            users["doctor@diaassist.ai"] = {
                "name": "Dr. Sarah Smith",
                "email": "doctor@diaassist.ai",
                "role": "Doctor",
                "specialty": "Endocrinologist",
                "hospital": "General Medical Center",
                "password": "doctor123"
            }
        if "patient@diaassist.ai" not in users:
            users["patient@diaassist.ai"] = {
                "name": "Ali Hassan",
                "email": "patient@diaassist.ai",
                "role": "Patient",
                "type": "Type 2 Diabetes Patient",
                "doctor_name": "Dr. Sarah Smith",
                "hospital": "Home Care",
                "password": "patient123"
            }
        st.session_state.registered_users = users

        # ------------------------------------------------------------- LOGIN TAB
        with tab_login:
            with st.form("login_form"):
                st.markdown("##### Account Sign In")
                
                email = st.text_input("Email Address", placeholder="doctor@diaassist.ai or patient@diaassist.ai")
                password = st.text_input("Password", type="password")
                
                submitted = st.form_submit_button("Sign In to Portal ➡️", use_container_width=True, type="primary")
                
                if submitted:
                    if email and password:
                        user_info = st.session_state.registered_users.get(email.strip().lower())
                        
                        if user_info and user_info["password"] == password.strip():
                            st.session_state.logged_in = True
                            st.session_state.user = user_info
                            st.success(f"Welcome, {user_info['name']}!")
                            st.rerun()
                        else:
                            st.error("Invalid email or password.")
                    else:
                        st.warning("Please enter both email and password.")

            st.caption("💡 **Test Accounts**: Doctor: `doctor@diaassist.ai` / `doctor123` | Patient: `patient@diaassist.ai` / `patient123`")

        # ------------------------------------------------------------- SIGNUP TAB
        with tab_signup:
            with st.form("signup_form"):
                st.markdown("##### Create Account")
                
                account_type = st.radio("I am registering as a:", ["👨‍⚕️ Healthcare Provider / Doctor", "👤 Patient / Self-Manager"], horizontal=True)
                new_name = st.text_input("Full Name", placeholder="e.g. Dr. John Doe or Ali Khan")
                new_email = st.text_input("Email Address", placeholder="name@domain.com")
                
                if "Doctor" in account_type:
                    new_role_type = "Doctor"
                    new_sub = st.selectbox("Specialty", ["Endocrinologist", "General Physician", "Diabetes Educator", "Clinical Researcher"])
                    new_hosp = st.text_input("Hospital / Institution", placeholder="City Hospital")
                else:
                    new_role_type = "Patient"
                    new_sub = st.selectbox("Diabetes Diagnosis", ["Type 1 Diabetes", "Type 2 Diabetes", "Gestational Diabetes", "Pre-diabetes"])
                    new_hosp = "Home Care"
                    
                new_pass = st.text_input("Password", type="password")
                confirm_pass = st.text_input("Confirm Password", type="password")
                
                terms = st.checkbox("I agree to terms and privacy policy")
                
                registered = st.form_submit_button("Create Account ✨", use_container_width=True, type="primary")
                
                if registered:
                    if not terms:
                        st.error("Please accept the terms.")
                    elif not new_name or not new_email or not new_pass:
                        st.warning("Please complete all required fields.")
                    elif new_pass != confirm_pass:
                        st.error("Passwords do not match.")
                    else:
                        email_clean = new_email.strip().lower()
                        st.session_state.registered_users[email_clean] = {
                            "name": new_name,
                            "email": email_clean,
                            "role": new_role_type,
                            "specialty": new_sub,
                            "doctor_name": "Dr. Sarah Smith",
                            "hospital": new_hosp,
                            "password": new_pass.strip()
                        }
                        st.session_state.logged_in = True
                        st.session_state.user = st.session_state.registered_users[email_clean]
                        st.success("Account created successfully!")
                        st.rerun()
