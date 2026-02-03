import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from datetime import datetime

# --------------------------------------------------
# SESSION STATE INITIALIZATION
# --------------------------------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "history_saved" not in st.session_state:
    st.session_state.history_saved = False

# --------------------------------------------------
# USER AUTHENTICATION HELPERS
# --------------------------------------------------
def load_users():
    try:
        return pd.read_csv("users.csv")
    except FileNotFoundError:
        df = pd.DataFrame(columns=["username", "password"])
        df.to_csv("users.csv", index=False)
        return df


def save_user(username, password):
    df = load_users()
    df.loc[len(df)] = [username, password]
    df.to_csv("users.csv", index=False)

# --------------------------------------------------
# USER-WISE HEALTH HISTORY (SAFE)
# --------------------------------------------------
def load_history():
    try:
        df = pd.read_csv("health_history.csv")
        if df.empty or len(df.columns) == 0:
            raise pd.errors.EmptyDataError
        return df
    except (FileNotFoundError, pd.errors.EmptyDataError):
        df = pd.DataFrame(
            columns=["username", "timestamp", "stress", "sleep", "calories"]
        )
        df.to_csv("health_history.csv", index=False)
        return df


def save_history(username, stress, sleep, calories):
    df = load_history()
    df.loc[len(df)] = [
        username,
        datetime.now().strftime("%Y-%m-%d %H:%M"),
        stress,
        sleep,
        calories
    ]
    df.to_csv("health_history.csv", index=False)

# --------------------------------------------------
# LOAD ML MODELS & FEATURE LISTS
# --------------------------------------------------
@st.cache_resource
def load_artifacts():
    stress_model = joblib.load("stress_model.pkl")
    sleep_model = joblib.load("sleep_model.pkl")
    calorie_model = joblib.load("calorie_model.pkl")

    stress_features = joblib.load("stress_features.pkl")
    sleep_features = joblib.load("sleep_features.pkl")
    calorie_features = joblib.load("calorie_features.pkl")

    return (
        stress_model,
        sleep_model,
        calorie_model,
        stress_features,
        sleep_features,
        calorie_features
    )


(
    stress_model,
    sleep_model,
    calorie_model,
    stress_features,
    sleep_features,
    calorie_features
) = load_artifacts()

# --------------------------------------------------
# LOGIN PAGE
# --------------------------------------------------
def login_page():
    st.markdown(
        "<h1 style='text-align:center; color:#1F618D;'>🩺 IntelliHealth</h1>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<h5 style='text-align:center;'>Intelligent Health Monitoring System</h5>",
        unsafe_allow_html=True
    )
    st.markdown("---")

    tab1, tab2 = st.tabs(["🔐 Login", "📝 Sign Up"])
    users_df = load_users()

    with tab1:
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            user = users_df[
                (users_df["username"] == username) &
                (users_df["password"] == password)
            ]
            if not user.empty:
                st.session_state.logged_in = True
                st.session_state.username = username
                st.session_state.history_saved = False
                st.success("Login successful!")
                st.rerun()
            else:
                st.error("Invalid username or password")

    with tab2:
        new_username = st.text_input("Create Username")
        new_password = st.text_input("Create Password", type="password")

        if st.button("Create Account"):
            if new_username == "" or new_password == "":
                st.warning("Please fill all fields")
            elif new_username in users_df["username"].values:
                st.error("Username already exists")
            else:
                save_user(new_username, new_password)
                st.success("Account created successfully! Please login.")

# --------------------------------------------------
# ENFORCE LOGIN
# --------------------------------------------------
if not st.session_state.logged_in:
    login_page()
    st.stop()

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
st.sidebar.title("🩺 Intelligent Health Monitor")
st.sidebar.markdown(f"👤 Logged in as: **{st.session_state.username}**")

page = st.sidebar.radio(
    "Select Page",
    [
        "Home",
        "Stress Analysis",
        "Sleep Analysis",
        "Calorie Analysis",
        "Visualization Dashboard",
        "Final Recommendations",
        "My Health History"
    ]
)

if st.sidebar.button("🚪 Logout"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

# --------------------------------------------------
# HOME PAGE
# --------------------------------------------------
if page == "Home":
    st.markdown(
        "<h1 style='text-align:center; color:#1F618D;'>"
        "🩺 Intelligent Health Monitoring System</h1>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<h4 style='text-align:center;'>"
        "Predict • Monitor • Improve Your Health</h4>",
        unsafe_allow_html=True
    )

    st.markdown("---")

    st.subheader("👋 Welcome!")

    st.markdown("""
    This application helps you **monitor your daily health status**
    by analyzing **stress levels**, **sleep quality**, and **calorie expenditure**.

    Simply enter your daily activity and sleep details to receive
    **personalized health insights**.
    """)

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 🧠 Stress Monitoring")
        st.markdown("""
        - Analyze physiological signals  
        - Detect stress levels  
        - Support mental well-being  
        """)

    with col2:
        st.markdown("### 😴 Sleep Analysis")
        st.markdown("""
        - Evaluate sleep quality  
        - Identify poor sleep patterns  
        - Improve recovery and rest  
        """)

    with col3:
        st.markdown("### 🔥 Calorie Tracking")
        st.markdown("""
        - Estimate daily calorie burn  
        - Based on activity & movement  
        - Encourage healthy lifestyle  
        """)

    st.markdown("---")

    st.info(
        "👉 Use the menu on the left to analyze your Stress, Sleep, or Calories."
    )

    st.caption(
        "This system uses machine learning to provide decision support for personal health monitoring."
    )


# --------------------------------------------------
# STRESS ANALYSIS
# --------------------------------------------------
elif page == "Stress Analysis":
    st.header("🧠 Stress Analysis")

    col1, col2 = st.columns([3, 2])
    with col1:
        rmssd = st.number_input("RMSSD (HRV)", 10.0, 150.0, key="rmssd")
    with col2:
        st.markdown("**Allowed Range:** 10 – 150 ms")

    col1, col2 = st.columns([3, 2])
    with col1:
        nremhr = st.number_input("NREM Heart Rate (bpm)", 40.0, 120.0, key="nremhr")
    with col2:
        st.markdown("**Allowed Range:** 40 – 120 bpm")

    col1, col2 = st.columns([3, 2])
    with col1:
        resting_hr = st.number_input("Resting Heart Rate (bpm)", 40.0, 120.0, key="resting_hr")
    with col2:
        st.markdown("**Allowed Range:** 40 – 120 bpm")

    col1, col2 = st.columns([3, 2])
    with col1:
        nightly_temp = st.number_input("Nightly Temperature (°C)", 30.0, 38.0, key="nightly_temp")
    with col2:
        st.markdown("**Allowed Range:** 30 – 38 °C")

    col1, col2 = st.columns([3, 2])
    with col1:
        steps = st.number_input("Steps", 0, 30000, key="steps")
    with col2:
        st.markdown("**Allowed Range:** 0 – 30,000 steps")

    col1, col2 = st.columns([3, 2])
    with col1:
        sedentary = st.number_input("Sedentary Minutes", 0, 1440, key="sedentary")
    with col2:
        st.markdown("**Allowed Range:** 0 – 1,440 mins")

    col1, col2 = st.columns([3, 2])
    with col1:
        sleep_duration = st.number_input("Sleep Duration (hours)", 0.0, 12.0, key="sleep_duration")
    with col2:
        st.markdown("**Allowed Range:** 0 – 12 hours")

    if st.button("Predict Stress"):
        X = pd.DataFrame(
            [[
                rmssd,
                nremhr,
                resting_hr,
                nightly_temp,
                steps,
                sedentary,
                sleep_duration
            ]],
            columns=stress_features
        )

        st.session_state.stress = stress_model.predict(X)[0]

        st.metric("Stress Index", f"{st.session_state.stress:.2f}")

        # ---------------- Interpretation ----------------
        if st.session_state.stress > 70:
            st.error("Stress Level: High")
        elif st.session_state.stress > 50:
            st.warning("Stress Level: Moderate")
        else:
            st.success("Stress Level: Low")

# --------------------------------------------------
# SLEEP ANALYSIS
# --------------------------------------------------
elif page == "Sleep Analysis":
    st.header("😴 Sleep Quality Analysis")

    col1, col2 = st.columns([3, 2])
    with col1:
        sleep_duration = st.number_input(
            "Sleep Duration (hours)", 0.0, 12.0, key="sleep_duration"
        )
    with col2:
        st.markdown("**Allowed Range:** 0 – 12 hours")

    col1, col2 = st.columns([3, 2])
    with col1:
        efficiency = st.number_input(
            "Sleep Efficiency (%)", 0.0, 100.0, key="efficiency"
        )
    with col2:
        st.markdown("**Allowed Range:** 0 – 100 %")

    col1, col2 = st.columns([3, 2])
    with col1:
        deep = st.slider(
            "Deep Sleep Ratio", 0.0, 1.0, key="deep"
        )
    with col2:
        st.markdown("**Allowed Range:** 0.0 – 1.0")

    col1, col2 = st.columns([3, 2])
    with col1:
        rem = st.slider(
            "REM Sleep Ratio", 0.0, 1.0, key="rem"
        )
    with col2:
        st.markdown("**Allowed Range:** 0.0 – 1.0")

    col1, col2 = st.columns([3, 2])
    with col1:
        awake = st.number_input(
            "Minutes Awake", 0, 300, key="awake"
        )
    with col2:
        st.markdown("**Allowed Range:** 0 – 300 mins")

    col1, col2 = st.columns([3, 2])
    with col1:
        breathing = st.number_input(
            "Breathing Rate (breaths/min)", 10.0, 25.0, key="breathing"
        )
    with col2:
        st.markdown("**Allowed Range:** 10 – 25 breaths/min")

    col1, col2 = st.columns([3, 2])
    with col1:
        nremhr = st.number_input(
            "NREM Heart Rate (bpm)", 40.0, 120.0, key="nremhr"
        )
    with col2:
        st.markdown("**Allowed Range:** 40 – 120 bpm")

    if st.button("Predict Sleep Quality"):
        minutes_asleep = sleep_duration * 60
        sleep_light_ratio = max(0.0, 1.0 - (deep + rem))

        X = pd.DataFrame(
            [[
                sleep_duration,
                efficiency,
                minutes_asleep,
                awake,
                deep,
                sleep_light_ratio,
                rem,
                breathing,
                nremhr
            ]],
            columns=sleep_features
        )

        st.session_state.sleep = sleep_model.predict(X)[0]

        st.metric(
            "Sleep Quality Index",
            f"{st.session_state.sleep:.2f}"
        )

        # ---------------- Interpretation ----------------
        if st.session_state.sleep > 65:
            st.success("Sleep Quality: Good")
        elif st.session_state.sleep > 45:
            st.warning("Sleep Quality: Average")
        else:
            st.error("Sleep Quality: Poor")


# --------------------------------------------------
# CALORIE ANALYSIS
# --------------------------------------------------
elif page == "Calorie Analysis":
    st.header("🔥 Calorie Prediction")

    col1, col2 = st.columns([3, 2])
    with col1:
        steps = st.number_input("Steps", 0, 30000, key="steps")
    with col2:
        st.markdown("**Allowed Range:** 0 – 30,000 steps")

    col1, col2 = st.columns([3, 2])
    with col1:
        distance = st.number_input("Distance (km)", 0.0, 30.0, key="distance")
    with col2:
        st.markdown("**Allowed Range:** 0 – 30 km")

    col1, col2 = st.columns([3, 2])
    with col1:
        light = st.number_input("Light Activity Minutes", 0, 500, key="light")
    with col2:
        st.markdown("**Allowed Range:** 0 – 500 mins")

    col1, col2 = st.columns([3, 2])
    with col1:
        moderate = st.number_input("Moderate Activity Minutes", 0, 300, key="moderate")
    with col2:
        st.markdown("**Allowed Range:** 0 – 300 mins")

    col1, col2 = st.columns([3, 2])
    with col1:
        vigorous = st.number_input("Vigorous Activity Minutes", 0, 180, key="vigorous")
    with col2:
        st.markdown("**Allowed Range:** 0 – 180 mins")

    col1, col2 = st.columns([3, 2])
    with col1:
        sedentary = st.number_input("Sedentary Minutes", 0, 1440, key="sedentary")
    with col2:
        st.markdown("**Allowed Range:** 0 – 1,440 mins")

    col1, col2 = st.columns([3, 2])
    with col1:
        bpm = st.number_input("Average BPM", 40.0, 150.0, key="bpm")
    with col2:
        st.markdown("**Allowed Range:** 40 – 150 bpm")

    col1, col2 = st.columns([3, 2])
    with col1:
        nremhr = st.number_input("NREM Heart Rate (bpm)", 40.0, 120.0, key="nremhr")
    with col2:
        st.markdown("**Allowed Range:** 40 – 120 bpm")

    col1, col2 = st.columns([3, 2])
    with col1:
        rmssd = st.number_input("RMSSD (HRV)", 10.0, 150.0, key="rmssd")
    with col2:
        st.markdown("**Allowed Range:** 10 – 150 ms")

    col1, col2 = st.columns([3, 2])
    with col1:
        sleep_duration = st.number_input(
            "Sleep Duration (hours)", 0.0, 12.0, key="sleep_duration"
        )
    with col2:
        st.markdown("**Allowed Range:** 0 – 12 hours")

    if st.button("Predict Calories"):
        X = pd.DataFrame(
            [[
                steps,
                distance,
                light,
                moderate,
                vigorous,
                sedentary,
                bpm,
                nremhr,
                rmssd,
                sleep_duration
            ]],
            columns=calorie_features
        )

        st.session_state.calories = calorie_model.predict(X)[0]
        st.metric(
            "Predicted Calories",
            f"{int(st.session_state.calories)} kcal/day"
        )

# --------------------------------------------------
# VISUALIZATION DASHBOARD
# --------------------------------------------------
elif page == "Visualization Dashboard":
    st.header("📊 Health Overview Dashboard")

    if None in (
        st.session_state.get("stress"),
        st.session_state.get("sleep"),
        st.session_state.get("calories")
    ):
        st.warning("Please run Stress, Sleep, and Calorie predictions first.")
    else:
        fig, ax1 = plt.subplots(figsize=(7, 5))

        # LEFT Y-axis → Stress & Sleep
        indicators = ["Stress", "Sleep"]
        values = [st.session_state.stress, st.session_state.sleep]
        colors = ["red", "green"]

        ax1.bar(indicators, values, color=colors, width=0.5)
        ax1.set_ylabel("Stress / Sleep Index")
        ax1.set_ylim(0, 100)

        # Threshold reference lines
        ax1.axhline(70, color="red", linestyle="--", linewidth=1)
        ax1.axhline(65, color="green", linestyle="--", linewidth=1)

        # RIGHT Y-axis → Calories
        ax2 = ax1.twinx()
        ax2.bar(
            ["Calories"],
            [st.session_state.calories],
            color="blue",
            width=0.4
        )
        ax2.set_ylabel("Calories (kcal/day)")
        ax2.set_ylim(0, max(3500, st.session_state.calories + 300))

        ax2.axhline(2500, color="blue", linestyle=":", linewidth=1)

        ax1.set_title("Health Indicators with Reference Thresholds")

        st.pyplot(fig)

        # ----------------------------
        # HEALTH INTERPRETATION
        # ----------------------------
        st.subheader("🧾 Health Interpretation")

        # Stress interpretation
        if st.session_state.stress > 70:
            st.error(f"🧠 Stress Level: High ({st.session_state.stress:.1f})")
        elif st.session_state.stress > 50:
            st.warning(f"🧠 Stress Level: Moderate ({st.session_state.stress:.1f})")
        else:
            st.success(f"🧠 Stress Level: Low ({st.session_state.stress:.1f})")

        # Sleep interpretation
        if st.session_state.sleep > 65:
            st.success(f"😴 Sleep Quality: Good ({st.session_state.sleep:.1f})")
        elif st.session_state.sleep > 45:
            st.warning(f"😴 Sleep Quality: Average ({st.session_state.sleep:.1f})")
        else:
            st.error(f"😴 Sleep Quality: Poor ({st.session_state.sleep:.1f})")

        # Calorie interpretation
        if st.session_state.calories > 2800:
            st.warning(
                f"🔥 Calorie Expenditure: High ({int(st.session_state.calories)} kcal)"
            )
        elif st.session_state.calories > 2000:
            st.success(
                f"🔥 Calorie Expenditure: Average ({int(st.session_state.calories)} kcal)"
            )
        else:
            st.info(
                f"🔥 Calorie Expenditure: Low ({int(st.session_state.calories)} kcal)"
            )


# --------------------------------------------------
# FINAL RECOMMENDATIONS
# --------------------------------------------------
elif page == "Final Recommendations":
    st.header("✅ Personalized Health Recommendations")

    if None in (
        st.session_state.get("stress"),
        st.session_state.get("sleep"),
        st.session_state.get("calories")
    ):
        st.warning("Please complete all analyses first.")
    else:
        stress = st.session_state.stress
        sleep = st.session_state.sleep
        calories = st.session_state.calories

        # ✅ Save history ONLY ONCE per session
        if not st.session_state.history_saved:
            save_history(
                st.session_state.username,
                stress,
                sleep,
                calories
            )
            st.session_state.history_saved = True
            st.success("📁 Health data saved to your history!")

        # ---------------- Stress ----------------
        st.subheader("🧠 Stress Recommendation")
        if stress > 70:
            st.error("High stress detected. Practice relaxation techniques and reduce workload.")
        elif stress > 50:
            st.warning("Moderate stress detected. Monitor stress and maintain balance.")
        else:
            st.success("Low stress detected. Maintain your current stress-management habits.")

        # ---------------- Sleep ----------------
        st.subheader("😴 Sleep Recommendation")
        if sleep > 65:
            st.success("Good sleep quality. Maintain consistent sleep routines.")
        elif sleep > 45:
            st.warning("Average sleep quality. Improve sleep hygiene and consistency.")
        else:
            st.error("Poor sleep quality. Prioritize adequate sleep and reduce disruptions.")

        # ---------------- Calories ----------------
        st.subheader("🔥 Calorie Recommendation")
        if calories > 2800:
            st.warning("High calorie expenditure. Ensure sufficient nutrition and hydration.")
        elif calories > 2000:
            st.success("Balanced calorie expenditure. Maintain your activity levels.")
        else:
            st.info("Low calorie expenditure. Consider increasing physical activity.")

        st.markdown("""
        **Overall Advice:**
        - Maintain a consistent daily routine  
        - Balance activity, recovery, and nutrition  
        - Monitor stress and sleep regularly  
        """)


# --------------------------------------------------
# MY HEALTH HISTORY
# --------------------------------------------------
elif page == "My Health History":
    st.header("📊 My Health History")

    history_df = load_history()
    user_history = history_df[
        history_df["username"] == st.session_state.username
    ]

    if user_history.empty:
        st.info("No health records found yet.")
    else:
        st.dataframe(user_history, use_container_width=True)
        st.line_chart(
            user_history.set_index("timestamp")[["stress", "sleep", "calories"]]
        )
