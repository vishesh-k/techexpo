import io
import json
import textwrap
import numpy as np
import pandas as pd
import streamlit as st

# Optional: sklearn & joblib
try:
    import joblib
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LogisticRegression

    SKLEARN_AVAILABLE = True
except:
    SKLEARN_AVAILABLE = False

# ---------------------------
# Styling
# ---------------------------
st.set_page_config(
    page_title="Portfolio & ML Demo",
    page_icon="🧭",
    layout="wide",
    menu_items={
        "Get help": "https://docs.streamlit.io",
        "Report a bug": "https://github.com/streamlit/streamlit/issues",
        "About": "Demo app with portfolio, models, and predictions"
    },
)

PRIMARY = "#2563eb"
st.markdown(
    f"""
    <style>
      .big-title {{ font-size: 2rem; font-weight: 800; margin: 0; }}
      .pill {{ display:inline-block; padding: 2px 10px; border-radius:999px; background:{PRIMARY}15; color:{PRIMARY}; font-weight:600; font-size:.85rem; }}
      .card {{ border:1px solid #e5e7eb; border-radius:16px; padding:16px; background:white; box-shadow: 0 1px 2px rgba(0,0,0,.04); }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------
# Helper functions
# ---------------------------
def load_pickle(upload):
    if upload is None or not SKLEARN_AVAILABLE:
        return None
    try:
        return joblib.load(upload)
    except:
        try:
            upload.seek(0)
            return joblib.load(io.BytesIO(upload.read()))
        except:
            return None


def safe_predict(model, df: pd.DataFrame):
    if model is None:
        return None
    try:
        preds = model.predict(df)
        return preds
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None


# ---------------------------
# Demo models
# ---------------------------
@st.cache_resource
def build_demo_salary_pipe():
    if not SKLEARN_AVAILABLE:
        return None
    rng = np.random.default_rng(42)
    n = 600
    df = pd.DataFrame({
        "years_experience": rng.integers(0, 16, size=n),
        "education": rng.choice(["High School", "Bachelor", "Master", "PhD"], size=n, p=[.2, .45, .3, .05]),
        "job_title": rng.choice(["Data Scientist", "ML Engineer", "Software Dev", "Analyst"], size=n),
        "location": rng.choice(["Bangalore", "Delhi", "Mumbai", "Remote"], size=n),
        "company_size": rng.choice(["Startup", "SME", "MNC"], size=n, p=[.3, .4, .3]),
    })
    base = 3.5 + 0.25 * df["years_experience"]
    edu_map = {"High School": -0.3, "Bachelor": 0.0, "Master": 0.3, "PhD": 0.6}
    job_map = {"Data Scientist": 0.6, "ML Engineer": 0.5, "Software Dev": 0.35, "Analyst": 0.2}
    loc_map = {"Bangalore": 0.25, "Delhi": 0.15, "Mumbai": 0.3, "Remote": 0.0}
    size_map = {"Startup": -0.05, "SME": 0.0, "MNC": 0.2}
    y = base + df["education"].map(edu_map) + df["job_title"].map(job_map) + df["location"].map(loc_map) + df[
        "company_size"].map(size_map) + rng.normal(0, 0.25, size=n)
    y = np.maximum(2.5, y)
    num = ["years_experience"]
    cat = ["education", "job_title", "location", "company_size"]
    pre = ColumnTransformer([
        ("num", StandardScaler(), num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
    ])
    pipe = Pipeline([("pre", pre), ("model", RandomForestRegressor(n_estimators=220, random_state=42))])
    pipe.fit(df[num + cat], y)
    return pipe


@st.cache_resource
def build_demo_water_pipe():
    if not SKLEARN_AVAILABLE:
        return None
    rng = np.random.default_rng(7)
    n = 1200
    df = pd.DataFrame({
        "ph": rng.normal(7.0, 1.2, size=n).clip(0.0, 14.0),
        "Hardness": rng.normal(200, 60, size=n).clip(10, 400),
        "Solids": rng.normal(20000, 9000, size=n).clip(100, 55000),
        "Chloramines": rng.normal(7.0, 2.0, size=n).clip(0.1, 13),
        "Sulfate": rng.normal(330, 120, size=n).clip(10, 600),
        "Conductivity": rng.normal(425, 120, size=n).clip(80, 800),
        "Organic_carbon": rng.normal(10, 5, size=n).clip(1, 30),
        "Trihalomethanes": rng.normal(70, 25, size=n).clip(5, 130),
        "Turbidity": rng.normal(3.0, 1.5, size=n).clip(0.1, 9.0),
    })

    def potable_rule(r):
        score = 0
        score += 1 if 6.5 <= r.ph <= 8.5 else 0
        score += 1 if r.Hardness <= 300 else 0
        score += 1 if r.Solids <= 30000 else 0
        score += 1 if 2 <= r.Chloramines <= 10 else 0
        score += 1 if 150 <= r.Sulfate <= 400 else 0
        score += 1 if 100 <= r.Conductivity <= 700 else 0
        score += 1 if r.Organic_carbon <= 20 else 0
        score += 1 if r.Trihalomethanes <= 100 else 0
        score += 1 if r.Turbidity <= 5 else 0
        return 1 if score >= 6 else 0

    y = df.apply(potable_rule, axis=1)
    num = df.columns.tolist()
    pre = ColumnTransformer([("num", StandardScaler(), num)])
    pipe = Pipeline([("pre", pre), ("model", LogisticRegression(max_iter=200))])
    pipe.fit(df, y)
    return pipe


# Load demo models
DEMO_SALARY = build_demo_salary_pipe()
DEMO_WATER = build_demo_water_pipe()

# ---------------------------
# Sidebar Navigation (navbar style)
# ---------------------------
st.sidebar.title("VIRTUAL DEMO")
menu_options = ["Home", "Portfolio", "Salary Prediction", "Water Quality", "Team"]
choice = st.sidebar.selectbox("MODEL VIEW", menu_options)

# ---------------------------
# Main content based on selection
# ---------------------------
if choice == "Home":
    # Home.py — Streamlit Home Page for Two ML Dashboards
    # Place this file at the root of your Streamlit app.
    # If you use Streamlit's multipage structure, also create two pages:
    #   pages/1_💼_Salary_Prediction.py
    #   pages/2_💧_Water_Quality_Prediction.py
    # The buttons below use st.page_link to jump to those pages.

    import streamlit as st
    import pandas as pd
    from datetime import datetime

    st.set_page_config(
        page_title="AI Dashboards Hub",
        page_icon="🧭",
        layout="wide",
    )

    # --------------- Styling ---------------
    st.markdown(
        """
        <style>
          .hero {
            background: linear-gradient(135deg, #0ea5e9 0%, #22c55e 100%);
            color: white; padding: 28px; border-radius: 24px;
            box-shadow: 0 10px 30px rgba(0,0,0,.15);
          }
          .badge {display:inline-block; padding:6px 10px; border-radius:999px;
                  background:#eef2ff; color:#3730a3; font-weight:600; margin-right:6px;}
          .card {background: white; border-radius: 20px; padding: 18px; border:1px solid #eef2f7;
                 box-shadow: 0 2px 12px rgba(16,24,40,.06);}    
          .muted {color:#475467}
          .step {display:flex; gap:12px; margin:10px 0;}
          .stepnum {min-width:28px; height:28px; border-radius:999px; background:#111827; color:white;
                    font-weight:700; display:flex; align-items:center; justify-content:center;}
          .pill {display:inline-block; border:1px solid #e5e7eb; padding:6px 10px; border-radius:999px; margin:4px 6px 0 0;}
          .small {font-size: 13px; color:#6b7280}
          .metricbox {border:1px dashed #e5e7eb; border-radius:16px; padding:16px;}
          .tablewrap {border:1px solid #eef2f7; border-radius:12px; overflow:hidden}
        </style>
        """,
        unsafe_allow_html=True,
    )

    # --------------- Hero ---------------
    st.markdown(
        f"""
        <div class="hero">
          <h1 style="margin:0;">🧭 Unified ML Hub</h1>
          <p style="margin-top:6px; font-size:18px; opacity:.95;">
            One home page for your machine learning dashboards. Explore <b>Salary Prediction</b> and
            <b>Water Quality Prediction</b> with clear, step‑by‑step details.
          </p>
          <div class="small">Last updated: {datetime.now().strftime('%b %d, %Y %I:%M %p')}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")

    # --------------- Overview Badges ---------------
    colA, colB, colC, colD = st.columns([1, 1, 1, 1])
    with colA:
        st.markdown('<span class="badge">Framework: Streamlit</span>', unsafe_allow_html=True)
    with colB:
        st.markdown('<span class="badge">Models: RandomForest, XGBoost*</span>', unsafe_allow_html=True)
    with colC:
        st.markdown('<span class="badge">Encoders: One‑Hot, StandardScaler</span>', unsafe_allow_html=True)
    with colD:
        st.markdown('<span class="badge">Artifacts: .pkl</span>', unsafe_allow_html=True)

    st.markdown("\n")

    # --------------- SALARY PREDICTION CARD ---------------
    with st.container():
        st.markdown("### 💼 Salary Prediction — Details (Step by Step)")
        st.markdown('<div class="card">', unsafe_allow_html=True)

        c1, c2 = st.columns([1.2, 1])
        with c1:
            st.markdown("**Problem**: Predict annual salary (₹ / $) from candidate/job features.")
            st.markdown("**Task Type**: Regression (or Income Bracket Classification → then map to salary).")
            st.markdown("**Primary Model**: RandomForestRegressor (trained offline, loaded via `joblib.load`).")

            # Steps one by one
            st.markdown("#### Pipeline — One by One")
            steps = [
                ("Input", "Collect form inputs: age, education, years_experience, job_role, country, remote, skills."),
                ("Validation", "Basic range checks (e.g., age 18–70) + required fields."),
                ("Feature Eng.",
                 "One‑hot encode categoricals (job_role, country, education). Scale numeric fields if needed."),
                ("Prediction", "Load model.pkl, align columns to training schema, call model.predict(X_aligned)."),
                ("Post‑process", "Clamp to sensible bounds; format currency in INR/USD based on user selection."),
                ("Explain", "Show feature importances, partial dependence or SHAP (optional)."),
            ]
            for i, (title, desc) in enumerate(steps, start=1):
                st.markdown(
                    f"<div class='step'><div class='stepnum'>{i}</div><div><b>{title}</b><br><span class='muted'>{desc}</span></div></div>",
                    unsafe_allow_html=True)

            # Expected schema table
            st.markdown("#### Expected Input Schema")
            salary_schema = pd.DataFrame([
                {"field": "age", "type": "int", "example": 28},
                {"field": "education", "type": "category", "example": "Bachelor"},
                {"field": "years_experience", "type": "float", "example": 3.5},
                {"field": "job_role", "type": "category", "example": "Data Analyst"},
                {"field": "country", "type": "category", "example": "India"},
                {"field": "remote", "type": "bool", "example": True},
                {"field": "skills", "type": "multilabel", "example": "Python, SQL"},
            ])
            st.dataframe(salary_schema, use_container_width=True, hide_index=True)

            with st.expander("Preprocessing Components"):
                st.markdown(
                    """
                    - **One‑Hot Encoder** fit on training categories (store as `encoder.pkl`).  
                    - **Column Aligner** to match training columns (missing → 0; unseen → 'Other').  
                    - **Scaler** (optional) for numeric columns.
                    """
                )

            with st.expander("Evaluation & Monitoring"):
                st.markdown("- Offline metrics: R², MAE, RMSE on hold‑out set.")
                st.markdown("- Online checks: input drift, out‑of‑range guardrails, logging predictions.")

        with c2:
            st.markdown("#### Quick Metrics (example)")
            st.markdown('<div class="metricbox">', unsafe_allow_html=True)
            mcol1, mcol2 = st.columns(2)
            with mcol1:
                st.metric(label="R²", value="0.82")
            with mcol2:
                st.metric(label="MAE (₹)", value="₹ 58,000")
            mcol3, mcol4 = st.columns(2)
            with mcol3:
                st.metric(label="RMSE (₹)", value="₹ 96,000")
            with mcol4:
                st.metric(label="Train Rows", value="48,210")
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown("#### Open the App")
            if st.button("💼 Go to Salary Prediction"):
                st.session_state["page"] = "salary"

            if st.button("💧 Go to Water Quality Prediction"):
                st.session_state["page"] = "water"

            page = st.session_state.get("page", "home")

            if page == "salary":
                st.title("Salary Prediction")
                st.write("Show salary prediction form here.")
            elif page == "water":
                st.title("Water Quality Prediction")
                st.write("Show water quality prediction form here.")

        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("\n")

    # --------------- WATER QUALITY CARD ---------------
    with st.container():
        st.markdown("### 💧 Water Quality Prediction — Details (Step by Step)")
        st.markdown('<div class="card">', unsafe_allow_html=True)

        w1, w2 = st.columns([1.2, 1])
        with w1:
            st.markdown("**Problem**: Predict water potability / quality index from physico‑chemical parameters.")
            st.markdown("**Task Type**: Classification (Potable / Not Potable) or Regression (WQI score).")
            st.markdown("**Primary Model**: RandomForestClassifier or XGBoostClassifier (loaded via `joblib`).")

            st.markdown("#### Pipeline — One by One")
            wsteps = [
                ("Input",
                 "Collect: pH, Hardness, Solids (TDS), Chloramines, Sulfate, Conductivity, Organic Carbon, Turbidity, Trihalomethanes, etc."),
                ("Validation", "Range checks per parameter (e.g., pH 0–14; no negatives)."),
                ("Imputation", "Median/mode impute missing numeric/categorical values."),
                ("Scaling", "Standardize numeric features if model benefits."),
                ("Prediction", "Load model.pkl → predict class and probability or continuous WQI."),
                ("Explain", "Show top features; threshold slider for classification decisions."),
            ]
            for i, (title, desc) in enumerate(wsteps, start=1):
                st.markdown(
                    f"<div class='step'><div class='stepnum'>{i}</div><div><b>{title}</b><br><span class='muted'>{desc}</span></div></div>",
                    unsafe_allow_html=True)

            st.markdown("#### Expected Input Schema")
            water_schema = pd.DataFrame([
                {"field": "pH", "type": "float", "example": 7.2},
                {"field": "Hardness", "type": "float", "example": 204.9},
                {"field": "Solids(TDS)", "type": "float", "example": 18630.0},
                {"field": "Chloramines", "type": "float", "example": 7.1},
                {"field": "Sulfate", "type": "float", "example": 350.0},
                {"field": "Conductivity", "type": "float", "example": 420.0},
                {"field": "OrganicCarbon", "type": "float", "example": 10.5},
                {"field": "Trihalomethanes", "type": "float", "example": 56.0},
                {"field": "Turbidity", "type": "float", "example": 3.6},
            ])
            st.dataframe(water_schema, use_container_width=True, hide_index=True)

            with st.expander("Preprocessing Components"):
                st.markdown(
                    """
                    - **Imputer** for numeric columns (median) and optional categorical (most frequent).  
                    - **Scaler** (StandardScaler/MinMaxScaler) as needed.  
                    - **Column Aligner** to ensure inference columns match training schema.
                    """
                )
            with st.expander("Evaluation & Monitoring"):
                st.markdown("- Offline metrics: Accuracy, F1, ROC‑AUC (classification) or R²/MAE (regression).")
                st.markdown("- Calibration plot and threshold tuning (optional).")

        with w2:
            st.markdown("#### Quick Metrics (example)")
            st.markdown('<div class="metricbox">', unsafe_allow_html=True)
            a1, a2 = st.columns(2)
            with a1:
                st.metric(label="Accuracy", value="0.91")
            with a2:
                st.metric(label="ROC‑AUC", value="0.95")
            a3, a4 = st.columns(2)
            with a3:
                st.metric(label="F1", value="0.90")
            with a4:
                st.metric(label="# Samples", value="32,500")
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown("#### Open the App")
            if page == "🏠 Home":
                st.header("Welcome!")
                st.write("Choose a project from the sidebar.")

            elif page == "💼 Salary Prediction":
                st.header("💼 Salary Prediction")
                st.write("Here goes your salary prediction model UI.")

            elif page == "💧 Water Quality":
                st.header("💧 Water Quality Prediction")
                st.write("Here goes your water quality model UI.")

        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("\n")

    # --------------- Developer Notes / Integration ---------------
    with st.expander("📦 How to integrate your trained models (.pkl)"):
        st.markdown(
            """
            **Folder structure (suggested):**
            ```
            /app
            ├─ Home.py
            ├─ models/
            │   ├─ salary/
            │   │   ├─ model.pkl
            │   │   ├─ encoder.pkl
            │   │   └─ train_columns.json
            │   └─ water/
            │       ├─ model.pkl
            │       └─ train_columns.json
            └─ pages/
                ├─ 1_💼_Salary_Prediction.py
                └─ 2_💧_Water_Quality_Prediction.py
            ```

            **On each page:**
            - Load artifacts with `import joblib` → `joblib.load('models/.../model.pkl')`.  
            - Keep a list of `train_columns` and reindex incoming features to this list (fill missing with 0).  
            - For classification → if you map classes to salary (e.g., `{'<=50K': 40000, '>50K': 80000}`), do the mapping **after** `predict`.
            """
        )

    with st.expander("🧪 Minimal inference helpers (copy into the respective pages)"):
        st.code(
            """
    import json, joblib, numpy as np, pandas as pd

    def load_artifacts(model_dir):
        model = joblib.load(f"{model_dir}/model.pkl")
        cols = json.load(open(f"{model_dir}/train_columns.json"))
        enc = None
        try:
            enc = joblib.load(f"{model_dir}/encoder.pkl")
        except Exception:
            pass
        return model, cols, enc

    # Align columns to training schema
    # X is a pandas DataFrame with raw (already encoded if enc is None) features

    def align_columns(X: pd.DataFrame, train_columns: list[str]) -> pd.DataFrame:
        X_aligned = X.reindex(columns=train_columns, fill_value=0)
        missing = set(train_columns) - set(X.columns)
        if missing:
            print("Filled missing columns with 0:", sorted(missing)[:5], "...")
        return X_aligned
            """,
            language="python",
        )

    with st.expander("🧭 Navigation without pages (single‑file fallback)"):
        st.markdown(
            "If you don't want to create separate files, you can use a selectbox on this Home page to switch between mini‑UIs.")
        choice = st.selectbox("Quick preview (single‑file demo)", ["— Select —", "Salary mini‑form", "Water mini‑form"])
        if choice == "Salary mini‑form":
            age = st.number_input("Age", 18, 70, 28)
            edu = st.selectbox("Education", ["High School", "Bachelor", "Master", "PhD"])
            exp = st.number_input("Years of Experience", 0.0, 50.0, 3.5, step=0.5)
            role = st.text_input("Job Role", "Data Analyst")
            country = st.text_input("Country", "India")
            remote = st.checkbox("Remote", True)
            if st.button("Predict (demo)"):
                st.info("This is a UI demo only. Hook up your trained model on the dedicated page.")
        elif choice == "Water mini‑form":
            pH = st.number_input("pH", 0.0, 14.0, 7.2)
            hard = st.number_input("Hardness", 0.0, 1000.0, 204.9)
            tds = st.number_input("Solids (TDS)", 0.0, 100000.0, 18630.0)
            chl = st.number_input("Chloramines", 0.0, 50.0, 7.1)
            sul = st.number_input("Sulfate", 0.0, 1000.0, 350.0)
            cond = st.number_input("Conductivity", 0.0, 2000.0, 420.0)
            org = st.number_input("Organic Carbon", 0.0, 50.0, 10.5)
            thm = st.number_input("Trihalomethanes", 0.0, 200.0, 56.0)
            turb = st.number_input("Turbidity", 0.0, 100.0, 3.6)
            if st.button("Predict (demo)"):
                st.info("This is a UI demo only. Hook up your trained model on the dedicated page.")

    st.markdown("---")
    import streamlit as st
    import pandas as pd
    import plotly.express as px

    # Sample glacier location data (latitude, longitude, name)
    data = {
        "Name": ["Glacier A", "Glacier B", "Glacier C"],
        "Latitude": [61.5, 46.8, 78.9],
        "Longitude": [-149.9, 11.2, 16.0],
        "Size_km2": [120, 80, 200]
    }
    df = pd.DataFrame(data)

    st.title("World Glacier Map")

    fig = px.scatter_geo(df,
                         lat="Latitude",
                         lon="Longitude",
                         hover_name="Name",
                         size="Size_km2",
                         projection="natural earth",
                         title="Sample Glacier Locations")
    st.plotly_chart(fig)
    st.caption("Tip: Put your links (GitHub, LinkedIn, Email) in the sidebar of each page for quick team contacts.")


elif choice == "Portfolio":
    st.markdown(f"<div class='big-title'>My <span style='color:{PRIMARY}'>Portfolio</span></div>",
                unsafe_allow_html=True)
    import streamlit as st

    # Page settings
    st.set_page_config(page_title="🌟 Portfolio", page_icon="✨", layout="wide")

    # ---------- HERO / INTRO ----------
    col1, col2 = st.columns([1, 3])
    with col1:
        st.image(r"C:\Users\adity\OneDrive\Documents\WhatsApp Image 2025-07-03 at 11.55.19_00a18f75.jpg", width=180)  # Replace with your photo
    with col2:
        st.markdown("""
            ## 👋 Hi, I'm Vishesh kumar Prajapati
            **Machine Learning Enthusiast | Data Scientist | Full Stack Web Developer**

            Passionate about building intelligent systems and stunning dashboards.  
            Connect with me 👇
        """)
        st.markdown(
            """
            <a href="https://linkedin.com/in/yourusername" target="_blank">
                <img src="https://img.shields.io/badge/LinkedIn-blue?logo=linkedin&logoColor=white" height="30">
            </a>
            <a href="https://github.com/yourusername" target="_blank">
                <img src="https://img.shields.io/badge/GitHub-black?logo=github&logoColor=white" height="30">
            </a>
            <a href="mailto:youremail@gmail.com" target="_blank">
                <img src="https://img.shields.io/badge/Email-red?logo=gmail&logoColor=white" height="30">
            </a>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")
    import streamlit as st

    # --- SKILLS / BADGES SECTION ---
    st.markdown("## 🛠️ Skills & Technologies")

    st.markdown(
        """
        <div style='display: flex; flex-wrap: wrap; gap: 10px;'>
            <img src="https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white&style=for-the-badge" height="30">
            <img src="https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white&style=for-the-badge" height="30">
            <img src="https://img.shields.io/badge/Scikit--Learn-F7931E?logo=scikitlearn&logoColor=white&style=for-the-badge" height="30">
            <img src="https://img.shields.io/badge/TensorFlow-FF6F00?logo=tensorflow&logoColor=white&style=for-the-badge" height="30">
            <img src="https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=white&style=for-the-badge" height="30">
            <img src="https://img.shields.io/badge/Numpy-013243?logo=numpy&logoColor=white&style=for-the-badge" height="30">
            <img src="https://img.shields.io/badge/Plotly-3F4F75?logo=plotly&logoColor=white&style=for-the-badge" height="30">
            <img src="https://img.shields.io/badge/Matplotlib-00457C?logo=matplotlib&logoColor=white&style=for-the-badge" height="30">
            <img src="https://img.shields.io/badge/HTML5-E34F26?logo=html5&logoColor=white&style=for-the-badge" height="30">
            <img src="https://img.shields.io/badge/CSS3-1572B6?logo=css3&logoColor=white&style=for-the-badge" height="30">
            <img src="https://img.shields.io/badge/JavaScript-F7DF1E?logo=javascript&logoColor=black&style=for-the-badge" height="30">
            <img src="https://img.shields.io/badge/MySQL-4479A1?logo=mysql&logoColor=white&style=for-the-badge" height="30">
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("---")
    # ---------- PROJECTS ----------
    st.markdown("## 💼 Projects Showcase")

    projects = [
        {
            "title": "Salary Prediction App",
            "desc": "ML model with Random Forest that predicts salaries based on education, job role, and experience.",
            "link": "https://github.com/yourusername/salary-prediction"
        },
        {
            "title": "Water Quality Prediction",
            "desc": "Classifies water as drinkable or not based on chemical parameters using ML.",
            "link": "https://github.com/yourusername/water-quality"
        },
        {
            "title": "Personal Portfolio Website",
            "desc": "Modern responsive portfolio website built with HTML, CSS, and JS.",
            "link": "https://your-portfolio-link.com"
        }
    ]

    # Custom CSS for cards
    st.markdown("""
        <style>
            .card {
                background: linear-gradient(135deg, #f6f9fc, #e9f0ff);
                border-radius: 20px;
                padding: 20px;
                margin-bottom: 20px;
                box-shadow: 0px 6px 15px rgba(0,0,0,0.1);
                transition: transform 0.3s ease-in-out;
            }
            .card:hover {
                transform: translateY(-6px);
                box-shadow: 0px 12px 25px rgba(0,0,0,0.2);
            }
            .card-title {
                font-size: 20px;
                font-weight: 700;
                color: #2c3e50;
                margin-bottom: 8px;
            }
            .card-desc {
                font-size: 15px;
                color: #555;
            }
        </style>
    """, unsafe_allow_html=True)

    for project in projects:
        st.markdown(
            f"""
            <div class="card">
                <div class="card-title">{project['title']}</div>
                <div class="card-desc">{project['desc']}</div>
                <br>
                <a href="{project['link']}" target="_blank">🔗 View Project</a>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")

    # ---------- CONTACT FORM ----------
    st.markdown("## 📩 Contact Me")

    with st.form("contact_form", clear_on_submit=True):
        name = st.text_input("Your Name")
        email = st.text_input("Your Email")
        query = st.text_area("Your Message")
        uploaded_file = st.file_uploader("Upload a file/photo (optional)", type=["png", "jpg", "jpeg", "pdf"])

        submitted = st.form_submit_button("Send Message ✉️")
        if submitted:
            if name and email and query:
                st.success(f"✅ Thanks {name}! Your message has been sent successfully.")
                if uploaded_file:
                    st.info(f"📎 You uploaded: {uploaded_file.name}")
            else:
                st.error("⚠️ Please fill in all required fields (Name, Email, Message).")

    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    import streamlit as st

    st.title("❄️ Glacier Melting Tracker")

    # Upload images
    before_file = st.file_uploader("Upload BEFORE Glacier Image", type=["jpg", "png"])
    after_file = st.file_uploader("Upload AFTER Glacier Image", type=["jpg", "png"])

    if before_file and after_file:
        # Read images in grayscale
        before = cv2.imdecode(np.frombuffer(before_file.read(), np.uint8), 0)
        after = cv2.imdecode(np.frombuffer(after_file.read(), np.uint8), 0)

        # Threshold to highlight glacier regions
        _, before_mask = cv2.threshold(before, 127, 255, cv2.THRESH_BINARY)
        _, after_mask = cv2.threshold(after, 127, 255, cv2.THRESH_BINARY)

        # Calculate glacier area (white pixels)
        before_area = np.sum(before_mask == 255)
        after_area = np.sum(after_mask == 255)
        melted = (before_area - after_area) / before_area * 100

        st.success(f"❄️ Glacier Melted: {melted:.2f}%")

        # --- Plot Graphs ---
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Show images
        axes[0].imshow(before, cmap="gray")
        axes[0].set_title("Before Glacier")
        axes[0].axis("off")

        axes[1].imshow(after, cmap="gray")
        axes[1].set_title("After Glacier")
        axes[1].axis("off")

        # Bar chart of glacier area
        axes[2].bar(["Before", "After"], [before_area, after_area], color=["blue", "red"])
        axes[2].set_title("Glacier Area Comparison")
        axes[2].set_ylabel("Pixel Count (Area)")

        st.pyplot(fig)

    import streamlit as st
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier

    st.title("💧 Water Quality Prediction System")

    # Upload dataset
    uploaded_file = st.file_uploader("Upload Water Quality Dataset (CSV)", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("📊 Dataset Preview:", df.head())

        if "Potability" in df.columns:
            # Features and labels
            X = df.drop("Potability", axis=1)
            y = df["Potability"]

            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Train model
            model = RandomForestClassifier()
            model.fit(X_train, y_train)

            st.success("✅ Model trained successfully!")

            # User input for prediction
            st.subheader("🔎 Test a Water Sample")
            ph = st.number_input("pH Level", min_value=0.0, max_value=14.0, value=7.0)
            hardness = st.number_input("Hardness", min_value=0.0, value=150.0)
            solids = st.number_input("Solids", min_value=0.0, value=200.0)
            chloramines = st.number_input("Chloramines", min_value=0.0, value=7.0)
            sulfate = st.number_input("Sulfate", min_value=0.0, value=400.0)
            conductivity = st.number_input("Conductivity", min_value=0.0, value=250.0)
            organic_carbon = st.number_input("Organic Carbon", min_value=0.0, value=3.0)
            trihalomethanes = st.number_input("Trihalomethanes", min_value=0.0, value=60.0)
            turbidity = st.number_input("Turbidity", min_value=0.0, value=3.0)

            # Arrange in same order as dataset
            sample = [[ph, hardness, solids, chloramines, sulfate, conductivity, organic_carbon, trihalomethanes,
                       turbidity]]

            if st.button("🔮 Predict Potability"):
                pred = model.predict(sample)
                st.success("💧 Safe to Drink" if pred[0] == 1 else "⚠️ Not Safe to Drink")
        else:
            st.error("❌ The dataset must contain a 'Potability' column.")
    import pandas as pd
    import plotly.express as px
    import streamlit as st

    st.title("💧 Water Quality Prediction by pH Values")

    # Load dataset
    df = pd.read_csv(r"C:\Users\adity\Downloads\water_potability (1).csv")

    # Ensure dataset has required columns
    if "ph" in df.columns and "Potability" in df.columns:
        # Map Potability: 1 = Safe, 0 = Not Safe
        df["Potability"] = df["Potability"].map({1: "✅ Safe to Drink", 0: "⚠️ Not Safe"})

        # Scatter plot
        fig = px.scatter(
            df,
            x="ph",
            y="Hardness",  # you can change to another parameter (like Sulfate, Solids, etc.)
            color="Potability",
            title="pH vs Water Quality Potability",
            labels={"ph": "pH Value", "Hardness": "Hardness Level"},
            opacity=0.7
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("CSV file must contain 'ph' and 'Potability' columns.")

    import streamlit as st
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier

    st.title("💼 Salary Prediction System")

    # Upload CSV file
    uploaded_file = st.file_uploader("📂 Upload Salary Dataset (CSV)", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        st.write("📊 Dataset Preview", df.head())

        if "income" in df.columns:
            # Features & Labels
            X = pd.get_dummies(df.drop("income", axis=1))
            y = df["income"]

            # Train/Test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Train model
            model = RandomForestClassifier()
            model.fit(X_train, y_train)

            st.success("✅ Model trained successfully!")

            # Show accuracy
            accuracy = model.score(X_test, y_test)
            st.info(f"📈 Model Accuracy: {accuracy * 100:.2f}%")

            # Predict from test sample
            idx = st.number_input("Choose row index from test data", min_value=0, max_value=len(X_test) - 1, value=0)
            if st.button("🔮 Predict Salary"):
                sample = X_test.iloc[[idx]]
                pred = model.predict(sample)
                st.success(f"💰 Predicted Salary Category: {pred[0]}")
        else:
            st.error("❌ Dataset must contain an 'income' column.")
    else:
        st.warning("⬆️ Please upload a CSV file to continue.")
    import pandas as pd
    import plotly.express as px
    import streamlit as st

    st.title("💼 Salary Prediction by Sector")

    # Example dataset (you can replace with your salary_data.csv or model output)
    data = {
        "Sector": ["Student", "Intern", "Junior Engineer", "Senior Engineer", "Manager", "Data Scientist",
                   "Software Engineer"],
        "Predicted_Salary_USD": [5000, 12000, 35000, 70000, 95000, 110000, 105000]
    }

    df = pd.DataFrame(data)

    # Plot
    fig = px.bar(
        df,
        x="Sector",
        y="Predicted_Salary_USD",
        color="Predicted_Salary_USD",
        text="Predicted_Salary_USD",
        title="💰 Salary Prediction from Student to Engineer & Beyond",
        color_continuous_scale="Blues"
    )

    fig.update_traces(texttemplate='$%{text:,.0f}', textposition="outside")
    fig.update_layout(
        xaxis_title="Career Sector",
        yaxis_title="Predicted Salary (USD)",
        uniformtext_minsize=8,
        uniformtext_mode="hide"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Featured Projects")
    st.markdown("<div class='grid-3'>", unsafe_allow_html=True)
    projects = [
        {
            "title": "Salary Prediction Dashboard",
            "desc": "RandomForest pipeline with categorical encoding and CSV batch scoring.",
            "tags": "Python • scikit‑learn • Streamlit"
        },
        {
            "title": "Water Potability Classifier",
            "desc": "Logistic Regression baseline with StandardScaler; supports form & CSV input.",
            "tags": "Python • scikit‑learn • Streamlit"
        },
        {
            "title": "Glacier Change Visualizer",
            "desc": "Image differencing + portfolio UI (separate app).",
            "tags": "OpenCV • Streamlit"
        }
    ]
    for proj in projects:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown(f"**{proj['title']}**")
        st.write(proj['desc'])
        st.caption(proj['tags'])
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    import streamlit as st
    import pandas as pd
    import pydeck as pdk

    st.set_page_config(page_title="World Map", layout="wide")
    st.title("🌍 World Map (PyDeck)")

    # Countries with coordinates (approx lat/lon)
    data = pd.DataFrame([


    ])

    # Initial view centered around South Asia
    initial_view = pdk.ViewState(latitude=30, longitude=80, zoom=2)

    # Scatterplot layer
    scatter = pdk.Layer(
        "ScatterplotLayer",
        data=data,
        get_position='[lon, lat]',
        get_radius=600000,  # in meters
        get_fill_color=[255, 0, 0],  # red circles
        pickable=True,
        auto_highlight=True,
    )

    # Tooltip
    tooltip = {"text": "{country}\nLat: {lat}\nLon: {lon}"}

    # Deck
    r = pdk.Deck(
        layers=[scatter],
        initial_view_state=initial_view,
        tooltip=tooltip,
    )

    st.pydeck_chart(r)
    st.caption("Tip: Zoom, pan, and hover over the red circles to see country details.")

    st.download_button(
        label="Download README.md",
        data=textwrap.dedent(
            """
            # Portfolio — Salary & Water Quality App
            Features:
            - Salary regression (RandomForest demo)
            - Water potability classification (Logistic demo)
            - Batch CSV scoring & model uploads.
            """
        ).strip(),

        file_name="README.md"
    )

elif choice == "Salary Prediction":
    st.markdown(f"<div class='big-title'>Salary <span style='color:{PRIMARY}'>Prediction</span></div>",
                unsafe_allow_html=True)
    st.write("Use demo model or upload your own sklearn pipeline (.pkl).")
    with st.expander("Model Source", expanded=True):
        model_type = st.radio("Select Model", ["Demo (built-in)", "Upload .pkl"], horizontal=True)
        user_salary_model = None
        if model_type == "Upload .pkl":
            uploaded_model = st.file_uploader("Upload scikit-learn Pipeline (.pkl)", type=["pkl", "joblib"])
            user_salary_model = load_pickle(uploaded_model)
            if uploaded_model and user_salary_model is None:
                st.error("Failed to load pickle.")
        salary_model = user_salary_model if user_salary_model else DEMO_SALARY
        if not SKLEARN_AVAILABLE and model_type == "Demo (built-in)":
            st.warning("scikit-learn not installed. Demo unavailable.")

    # Prediction form
    # --- Single Prediction ---
    st.subheader("Single Prediction")

    col1, col2, col3 = st.columns(3)
    with col1:
        years = st.number_input("Years of Experience", 0, 40, 3)
        edu = st.selectbox("Education", ["High School", "Bachelor", "Master", "PhD"])
    with col2:
        title = st.selectbox("Job Title", ["Data Scientist", "ML Engineer", "Software Dev", "Analyst"])
        location = st.selectbox("Location", ["Bangalore", "Delhi", "Mumbai", "Remote"])
    with col3:
        size = st.selectbox("Company Size", ["Startup", "SME", "MNC"])

    df_input = pd.DataFrame([{
        "years_experience": years,
        "education": edu,
        "job_title": title,
        "location": location,
        "company_size": size,
    }])

    # 🔑 Fix: keep preds local, not global
    if st.button("Predict Salary (LPA)"):
        preds = safe_predict(salary_model, df_input)
        if preds is not None:
            st.success(f"Estimated Salary: **{preds[0]:.2f} LPA**")
            # Chart only after prediction
            st.line_chart(pd.Series([preds[0]]), height=200)

    st.divider()
    st.subheader("Batch CSV Predictions")
    csv_input = st.file_uploader("Upload CSV", type=["csv"])
    if csv_input:
        try:
            df_csv = pd.read_csv(csv_input)
            st.dataframe(df_csv.head(20))
            if st.button("Run Batch Predictions"):
                preds = safe_predict(salary_model, df_csv)
                if preds is not None:
                    out_df = df_csv.copy()
                    out_df["predicted_salary_LPA"] = preds
                    st.success("Predictions generated.")
                    st.dataframe(out_df.head(50))
                    st.download_button(
                        "Download CSV", out_df.to_csv(index=False).encode("utf-8"), "salary_predictions.csv"
                    )
        except Exception as e:
            st.error(f"Error reading CSV: {e}")

elif choice == "Water Quality":
    st.markdown(f"<div class='big-title'>Water <span style='color:{PRIMARY}'>Quality</span></div>",
                unsafe_allow_html=True)
    st.write("Use demo or upload your own sklearn pipeline (.pkl).")
    with st.expander("Model Source", expanded=True):
        model_type2 = st.radio("Select Model", ["Demo (built-in)", "Upload .pkl"], horizontal=True,
                               key="water_model_source")
        user_water_model = None
        if model_type2 == "Upload .pkl":
            uploaded_water = st.file_uploader("Upload scikit-learn Pipeline (.pkl)", type=["pkl", "joblib"],
                                              key="water_pkl")
            user_water_model = load_pickle(uploaded_water)
            if uploaded_water and user_water_model is None:
                st.error("Failed to load pickle.")
        water_model = user_water_model if user_water_model else DEMO_WATER
        if not SKLEARN_AVAILABLE and model_type2 == "Demo (built-in)":
            st.warning("scikit-learn not installed. Demo unavailable.")

    # Single sample assessment
    st.subheader("Single Sample Test")
    col1, col2, col3 = st.columns(3)
    ph = col1.number_input("pH", 0.0, 14.0, 7.0, step=0.1)
    hardness = col1.number_input("Hardness", 0.0, 1000.0, 200.0)
    solids = col1.number_input("Solids", 0.0, 100000.0, 20000.0)
    chloramines = col2.number_input("Chloramines", 0.0, 20.0, 7.0, step=0.1)
    sulfate = col2.number_input("Sulfate", 0.0, 800.0, 330.0)
    conductivity = col2.number_input("Conductivity", 0.0, 2000.0, 425.0)
    organic_carbon = col3.number_input("Organic Carbon", 0.0, 50.0, 10.0)
    trihalomethanes = col3.number_input("Trihalomethanes", 0.0, 200.0, 70.0)
    turbidity = col3.number_input("Turbidity", 0.0, 20.0, 3.0)

    sample_df = pd.DataFrame([{
        "ph": ph,
        "Hardness": hardness,
        "Solids": solids,
        "Chloramines": chloramines,
        "Sulfate": sulfate,
        "Conductivity": conductivity,
        "Organic_carbon": organic_carbon,
        "Trihalomethanes": trihalomethanes,
        "Turbidity": turbidity,
    }])

    if st.button("Assess Potability"):
        preds = safe_predict(water_model, sample_df)
        if preds is not None:
            try:
                proba = water_model.predict_proba(sample_df)[:, 1][0]
            except:
                proba = None
            label = int(np.rint(preds[0]))
            verdict = "Potable ✅" if label == 1 else "Not Potable ❌"
            if proba is not None:
                st.success(f"Prediction: **{verdict}** — Confidence: {proba:.2%}")
            else:
                st.success(f"Prediction: **{verdict}**")
    # Batch predictions
    st.divider()
    st.subheader("Batch CSV Predictions")
    csv_water = st.file_uploader("Upload water samples CSV", type=["csv"], key="water_csv")
    if csv_water:
        try:
            df_water_csv = pd.read_csv(csv_water)
            st.dataframe(df_water_csv.head(20))
            if st.button("Run Batch Water Predictions"):
                preds = safe_predict(water_model, df_water_csv)
                if preds is not None:
                    out_df = df_water_csv.copy()
                    out_df["potable_pred"] = (np.rint(preds).astype(int))
                    try:
                        probs = water_model.predict_proba(df_water_csv)[:, 1]
                        out_df["potable_confidence"] = probs
                    except:
                        pass
                    st.success("Predictions done.")
                    st.dataframe(out_df.head(50))
                    st.download_button("Download CSV", out_df.to_csv(index=False).encode("utf-8"),
                                       "water_predictions.csv")
        except Exception as e:
            st.error(f"Error reading CSV: {e}")

# ---------------------------
# Team Section with edit button
# ---------------------------
elif choice == "Team":
    st.markdown(f"<div class='big-title'>Meet the <span style='color:{PRIMARY}'>Team</span></div>",
                unsafe_allow_html=True)

    if "team" not in st.session_state:
        st.session_state.team = [
            {"name": "Vishesh Kumar Prajapati", "role": "Founder / ML Engineer",
             "email": "vishesh@example.com", "linkedin": "https://linkedin.com/in/your-id",
             "github": "https://github.com/", "photo": None},
            {"name": "Teammate 2", "role": "Data Scientist", "email": "teammate2@example.com",
             "linkedin": "https://linkedin.com", "github": "", "photo": None},
        ]
    else:
        for m in st.session_state.team:
            if "photo" not in m:
                m["photo"] = None

    import base64

    for idx, member in enumerate(st.session_state.team):
        with st.container():
            col1, col2 = st.columns([1, 4])

            with col1:
                if member["photo"]:
                    st.image(member["photo"], width=100, caption=member["name"])
                else:
                    st.image("https://via.placeholder.com/100", width=100, caption=member["name"])

            with col2:
                st.subheader(member["name"])
                st.caption(member["role"])

                # Buttons row
                c1, c2, c3, c4 = st.columns([1,1,1,3])
                with c1:
                    if member.get("email"):
                        st.link_button("📧 Email", f"mailto:{member['email']}")
                with c2:
                    if member.get("linkedin"):
                        st.link_button("💼 LinkedIn", member["linkedin"])
                with c3:
                    if member.get("github"):
                        st.link_button("💻 GitHub", member["github"])
                with c4:
                    edit_expander = st.expander("✏️ Edit Member", expanded=False)
                    with edit_expander:
                        with st.form(f"edit_form_{idx}"):
                            new_name = st.text_input("Name", member["name"])
                            new_role = st.text_input("Role", member["role"])
                            new_email = st.text_input("Email", member["email"])
                            new_linkedin = st.text_input("LinkedIn", member["linkedin"])
                            new_github = st.text_input("GitHub", member["github"])
                            new_photo = st.file_uploader("Upload Photo", type=["png", "jpg", "jpeg"],
                                                         key=f"edit_photo_{idx}")

                            update = st.form_submit_button("✅ Update")
                            if update:
                                member["name"] = new_name
                                member["role"] = new_role
                                member["email"] = new_email
                                member["linkedin"] = new_linkedin
                                member["github"] = new_github
                                if new_photo:
                                    member["photo"] = new_photo
                                st.success(f"Updated {new_name}")

    # Add new member
    with st.expander("➕ Add New Member"):
        with st.form("add_member_form"):
            st.subheader("Add a New Team Member")
            name = st.text_input("Name")
            role = st.text_input("Role")
            email = st.text_input("Email")
            linkedin = st.text_input("LinkedIn")
            github = st.text_input("GitHub")
            photo = st.file_uploader("Upload Photo", type=["png", "jpg", "jpeg"], key="new_photo")
            add_submitted = st.form_submit_button("Add Member")
            if add_submitted and name and role:
                new_member = {
                    "name": name, "role": role, "email": email,
                    "linkedin": linkedin, "github": github, "photo": photo
                }
                st.session_state.team.append(new_member)
                st.success(f"Added {name} to the team!")

    # Delete member
    with st.form("delete_member_form"):
        st.subheader("🗑️ Delete a Member")
        names = [m['name'] for m in st.session_state.team]
        delete_name = st.selectbox("Select member to delete", options=[""] + names)
        delete_submitted = st.form_submit_button("Delete")
        if delete_submitted and delete_name:
            st.session_state.team = [m for m in st.session_state.team if m['name'] != delete_name]
            st.success(f"Deleted {delete_name}")

    # Export
    st.download_button(
        "📥 Export Team JSON",
        data=json.dumps(st.session_state.team, indent=2, default=str).encode("utf-8"),
        file_name="team.json"
    )

# ---------------------------
# Additional: World Map Visualization (for Home or other pages)
# ---------------------------
if choice == "Home" or choice == "Portfolio":
    # Example: display a world map with random data points
    import pydeck as pdk

    df_map = pd.DataFrame({
        'lat': np.random.uniform(-60, 60, size=50),
        'lon': np.random.uniform(-180, 180, size=50),
        'value': np.random.rand(50)
    })
    st.pydeck_chart(pdk.Deck(
        map_style='mapbox://styles/mapbox/light-v9',
        initial_view_state=pdk.ViewState(
            latitude=0,
            longitude=0,
            zoom=1,
            pitch=0,
        ),
        layers=[
            pdk.Layer(
                'ScatterplotLayer',
                data=df_map,
                get_position='[lon, lat]',
                get_color='[255, 0, 0, 160]',
                get_radius='value * 10000',
            ),
        ],
    ))

# ---------------------------
# End of app
# ---------------------------
st.info("Tip: You can upload your own models for production. Ensure they include all preprocessing.")
