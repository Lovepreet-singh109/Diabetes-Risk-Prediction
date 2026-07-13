import json
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from abc import ABC, abstractmethod
import textwrap

st.set_page_config(page_title="Diabetes Risk Predictor", page_icon="🩺", layout="wide")


def md(html: str, **kwargs):
    """st.markdown wrapper that strips common leading indentation first.

    Streamlit's markdown parser still applies CommonMark rules before the
    unsafe_allow_html pass-through, so any line indented 4+ spaces (which
    happens naturally with multi-line f-strings written inside indented
    Python methods) gets treated as a fenced code block and shown as raw
    text instead of being rendered as HTML. Dedenting first avoids that.
    """
    st.markdown(textwrap.dedent(html).strip(), unsafe_allow_html=True, **kwargs)


# ------------------------------------------------------------------
# Design System
# ------------------------------------------------------------------
class DesignSystem:
    PALETTE = {
        "bg_a": "#03010a",
        "bg_b": "#0B0A16",
        "glass": "rgba(255,255,255,0.035)",
        "glass_border": "rgba(255,255,255,0.08)",
        "glass_hi": "rgba(255,255,255,0.06)",
        "text": "#F8FAFC",
        "muted": "#94A3B8",
        "teal": "#06B6D4",
        "teal_soft": "rgba(6,182,212,0.12)",
        "violet": "#8B5CF6",
        "violet_soft": "rgba(139,92,246,0.12)",
        "low": "#10B981",
        "low_bg": "rgba(16,185,129,0.12)",
        "moderate": "#F59E0B",
        "moderate_bg": "rgba(245,158,11,0.12)",
        "high": "#EF4444",
        "high_bg": "rgba(239,68,68,0.12)",
    }

    MODEL_ACCENTS = {
        "Logistic Regression": "#06B6D4", # Cyan
        "Linear Regression": "#8B5CF6",   # Purple
        "Random Forest": "#F59E0B",       # Amber
        "Decision Tree": "#EC4899",       # Pink
    }
    
    MODEL_ACCENTS_FADE = {
        "Logistic Regression": "#3B82F6",
        "Linear Regression": "#D946EF",
        "Random Forest": "#FBBF24",
        "Decision Tree": "#F43F5E",
    }

    @classmethod
    def apply_custom_css(cls, accent_color: str):
        # Load external stylesheet
        with open("assets/style.css") as f:
            style_content = f.read()
            
        st.markdown(f"<style>{style_content}</style>", unsafe_allow_html=True)
        st.markdown(f"""
        <style>
        :root {{
            --accent-color: {accent_color};
            --accent-glow: {accent_color}25;
            --accent-glow-strong: {accent_color}55;
        }}

        /* -------------------------------------------------- */
        /* Hover / pop micro-interactions                     */
        /* -------------------------------------------------- */
        @media (prefers-reduced-motion: no-preference) {{

            /* Buttons: lift on hover, snap down ("pop") on click */
            .stButton > button {{
                transition: transform 0.22s cubic-bezier(.2,.8,.3,1.4),
                            box-shadow 0.22s ease,
                            border-color 0.22s ease;
                will-change: transform;
            }}
            .stButton > button:hover {{
                transform: translateY(-3px) scale(1.02);
                box-shadow: 0 10px 24px var(--accent-glow-strong), 0 0 0 1px var(--accent-color) inset;
                border-color: var(--accent-color) !important;
            }}
            .stButton > button:active {{
                transform: translateY(0px) scale(0.95);
                transition: transform 0.08s ease-out;
                box-shadow: 0 2px 8px var(--accent-glow);
            }}

            /* Bordered content cards: gentle lift + glow */
            div[data-testid="stVerticalBlockBorderWrapper"] {{
                transition: transform 0.3s cubic-bezier(.2,.8,.3,1.2),
                            box-shadow 0.3s ease,
                            border-color 0.3s ease;
            }}
            div[data-testid="stVerticalBlockBorderWrapper"]:hover {{
                transform: translateY(-4px);
                box-shadow: 0 16px 32px rgba(0,0,0,0.35), 0 0 0 1px var(--accent-glow-strong);
            }}

            /* Plotly chart frames: subtle scale + glow on hover */
            div[data-testid="stPlotlyChart"] {{
                transition: transform 0.3s cubic-bezier(.2,.8,.3,1.2),
                            box-shadow 0.3s ease,
                            filter 0.3s ease;
                border-radius: 14px;
            }}
            div[data-testid="stPlotlyChart"]:hover {{
                transform: scale(1.012);
                box-shadow: 0 14px 30px rgba(0,0,0,0.4), 0 0 0 1px var(--accent-glow-strong);
                filter: brightness(1.04);
                z-index: 5;
            }}

            /* Metric widgets: pop slightly on hover */
            div[data-testid="stMetric"] {{
                transition: transform 0.25s cubic-bezier(.2,.8,.3,1.3);
                border-radius: 10px;
            }}
            div[data-testid="stMetric"]:hover {{
                transform: scale(1.04);
            }}

            /* Chips: lift + brighten */
            .chip {{
                transition: transform 0.2s ease, box-shadow 0.2s ease;
                display: inline-block;
            }}
            .chip:hover {{
                transform: translateY(-2px) scale(1.03);
                box-shadow: 0 6px 16px rgba(255,255,255,0.08);
            }}

            /* Sidebar radio options: gentle press feedback */
            div[data-testid="stSidebar"] label {{
                transition: transform 0.18s ease, opacity 0.18s ease;
            }}
            div[data-testid="stSidebar"] label:hover {{
                transform: translateX(3px);
                opacity: 0.9;
            }}

            /* Selectbox: hover glow */
            div[data-baseweb="select"] > div {{
                transition: box-shadow 0.22s ease, border-color 0.22s ease;
            }}
            div[data-baseweb="select"]:hover > div {{
                box-shadow: 0 0 0 1px var(--accent-glow-strong);
                border-color: var(--accent-color) !important;
            }}
        }}

        /* -------------------------------------------------- */
        /* Number input: fix white-on-white contrast           */
        /* -------------------------------------------------- */
        div[data-testid="stNumberInput"] input {{
            background-color: rgba(255,255,255,0.06) !important;
            color: #F8FAFC !important; 
            font-weight: 600;
            border: 1px solid {DesignSystem.PALETTE["glass_border"]} !important;
            caret-color: {DesignSystem.PALETTE["text"]};
        }}
        div[data-testid="stNumberInput"] input::placeholder {{
            color: {DesignSystem.PALETTE["muted"]} !important;
        }}
        div[data-testid="stNumberInput"] button {{
            background-color: rgba(255,255,255,0.06) !important;
            color: {DesignSystem.PALETTE["text"]} !important;
            border: 1px solid {DesignSystem.PALETTE["glass_border"]} !important;
        }}
        div[data-testid="stNumberInput"] button:hover {{
            background-color: var(--accent-glow-strong) !important;
            border-color: var(--accent-color) !important;
        }}
        div[data-testid="stNumberInput"] > div {{
            background-color: transparent !important;
        }}
        </style>
        """, unsafe_allow_html=True)


# ------------------------------------------------------------------
# Model Manager
# ------------------------------------------------------------------
@st.cache_resource
def load_cached_artifacts():
    models = {name: joblib.load(path) for name, path in ModelManager.MODEL_FILES.items()}
    scaler = joblib.load("diabetes_scaler.pkl")
    imputer = joblib.load("diabetes_imputer.pkl")
    with open("model_metrics.json") as f:
        meta = json.load(f)
    return models, scaler, imputer, meta


class ModelManager:
    MODEL_FILES = {
        "Logistic Regression": "diabetes_logreg.pkl",
        "Linear Regression": "diabetes_linreg.pkl",
        "Random Forest": "diabetes_rf.pkl",
        "Decision Tree": "diabetes_dtree.pkl",
    }

    FEATURE_NAMES = [
        "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
        "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
        "BMI_Category", "Age_Group", "High_Glucose"
    ]

    def __init__(self):
        models, scaler, imputer, meta = load_cached_artifacts()
        self.models = models
        self.scaler = scaler
        self.imputer = imputer
        self.meta = meta
        self.baseline_means = np.array(meta["baseline_means"])
        self.metrics = meta["metrics"]
        self.roc_data = meta["roc_data"]
        self.feature_importance = meta["feature_importance"]

    @staticmethod
    def bmi_category(bmi):
        if bmi < 18.5:
            return 0
        elif bmi < 25:
            return 1
        elif bmi < 30:
            return 2
        return 3

    @staticmethod
    def age_group(age):
        if age < 30:
            return 0
        elif age < 45:
            return 1
        elif age < 60:
            return 2
        return 3

    def preprocess_inputs(self, raw_data: pd.DataFrame) -> np.ndarray:
        """
        Preprocesses a raw patient record:
        1. Identifies invalid zero values in physiological fields and replaces with NaN.
        2. Imputes missing values using KNN imputer.
        3. Appends engineered features.
        4. Scales final feature matrix.
        """
        df = raw_data.copy()
        
        # Zero is physiologically impossible for these variables; treat them as missing for KNN Imputation
        cols_with_invalid_zeros = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
        df[cols_with_invalid_zeros] = df[cols_with_invalid_zeros].replace(0, np.nan)
        
        # Apply imputer
        imputed_arr = self.imputer.transform(df)
        df_imputed = pd.DataFrame(imputed_arr, columns=df.columns)
        
        # Re-derive engineered features
        df_imputed["BMI_Category"] = df_imputed["BMI"].apply(self.bmi_category)
        df_imputed["Age_Group"] = df_imputed["Age"].apply(self.age_group)
        df_imputed["High_Glucose"] = (df_imputed["Glucose"] > 140).astype(int)
        
        # Scale inputs in exact feature order
        scaled_arr = self.scaler.transform(df_imputed[self.FEATURE_NAMES])
        return scaled_arr[0]

    def predict_proba_row(self, model_name: str, row_scaled: np.ndarray) -> float:
        model = self.models[model_name]
        if model_name == "Linear Regression":
            raw = model.predict(row_scaled.reshape(1, -1))[0]
            return float(np.clip(raw, 0, 1))
        return float(model.predict_proba(row_scaled.reshape(1, -1))[0][1])

    def explain_prediction(self, model_name: str, row_scaled: np.ndarray) -> tuple[dict[str, float], float]:
        """
        Dependency-free leave-one-feature-out SHAP-like contribution chart.
        """
        full_pred = self.predict_proba_row(model_name, row_scaled)
        contributions = {}
        for i, name in enumerate(self.FEATURE_NAMES):
            perturbed = row_scaled.copy()
            perturbed[i] = self.baseline_means[i]
            contributions[name] = full_pred - self.predict_proba_row(model_name, perturbed)
        return contributions, full_pred


# ------------------------------------------------------------------
# UI Component Renderers
# ------------------------------------------------------------------
class UIComponents:
    @staticmethod
    def render_hero(title: str, subtitle: str):
        md(f"""
        <div class="hero">
            <div style="position:relative; z-index:2;">
                <h1>🩺 {title}</h1>
                <p>{subtitle}</p>
            </div>
            <!-- Scanner grid overlay -->
            <div style="position:absolute; inset:0; background:linear-gradient(90deg, rgba(255,255,255,0.015) 1px, transparent 1px), linear-gradient(0deg, rgba(255,255,255,0.015) 1px, transparent 1px); background-size:20px 20px; pointer-events:none;"></div>
            <div style="position:absolute; width:180px; height:180px; background:radial-gradient(circle, var(--accent-glow-strong) 0%, transparent 70%); top:-60px; right:-60px; pointer-events:none;"></div>
        </div>
        """)

    @staticmethod
    def eyebrow(text: str):
        md(f'<div class="eyebrow">{text}</div>')

    @staticmethod
    def chip(label: str, value: str, risk_level: str = "none") -> str:
        dot_color = "#94A3B8"
        if risk_level == "low":
            dot_color = DesignSystem.PALETTE["low"]
        elif risk_level == "moderate":
            dot_color = DesignSystem.PALETTE["moderate"]
        elif risk_level == "high":
            dot_color = DesignSystem.PALETTE["high"]

        # Built without leading indentation so it's safe to splice into a
        # larger multi-line block without triggering markdown's code-block rule.
        return (
            f'<span class="chip" style="border-left: 3px solid {dot_color} !important;">'
            f'<span style="display:inline-block; width:6px; height:6px; background:{dot_color}; '
            f'border-radius:50%; margin-right:6px; box-shadow:0 0 6px {dot_color};"></span>'
            f'{label}: <b>{value}</b></span>'
        )

    @staticmethod
    def render_ambient_particles():
        """Injects a fixed, full-viewport canvas of drifting glowing motes into the
        parent document. Reads assets/particles.html and renders it via a
        zero-height component (the script targets window.parent, not the iframe)."""
        with open("assets/particles.html") as f:
            html = f.read()
        components.html(html, height=0, width=0)

    @staticmethod
    def render_pulse_divider():
        md(f"""
        <svg viewBox="0 0 600 28" preserveAspectRatio="none" xmlns="http://www.w3.org/2000/svg"
             style="width:100%; height:26px; margin:1.4rem 0; opacity:0.4;">
            <polyline points="0,14 220,14 240,4 255,24 270,14 290,14 300,2 312,26 324,14 600,14"
                      fill="none" stroke="var(--accent-color)" stroke-width="2"
                      stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
        """)

    @staticmethod
    def render_gauge(prob_pct: float, color: str, tier: str):
        radius = 85
        circumference = 2 * 3.14159265 * radius
        color_fade = color + "20"
        color_name = tier.lower()
        
        with open("assets/gauge.html") as f:
            template = f.read()
            
        html = (template
                .replace("__PROB_PCT__", f"{prob_pct:.2f}")
                .replace("__RADIUS__", str(radius))
                .replace("__CIRCUMFERENCE__", f"{circumference:.2f}")
                .replace("__COLOR__", color)
                .replace("__COLOR_FADE__", color_fade)
                .replace("__COLOR_NAME__", color_name)
                .replace("__TIER__", tier))
                
        components.html(html, height=265)

    @staticmethod
    def render_multi_model_simulator(all_probs: dict, selected_model: str):
        rows_html = []
        for name, prob in all_probs.items():
            prob_pct = prob * 100
            model_color = DesignSystem.MODEL_ACCENTS[name]
            model_color_fade = DesignSystem.MODEL_ACCENTS_FADE[name]
            
            is_active = (name == selected_model)
            if is_active:
                row_bg = "rgba(255, 255, 255, 0.05)"
                row_border = f"1px solid {model_color}45"
                selected_badge = f'<span style="font-size:0.65rem; background:{model_color}25; color:{model_color}; border:1px solid {model_color}40; padding:1px 6px; border-radius:4px; margin-left:8px; font-weight:700; text-transform:uppercase; letter-spacing:0.05em; box-shadow:0 0 6px {model_color}20;">Active</span>'
            else:
                row_bg = "rgba(255, 255, 255, 0.015)"
                row_border = "1px solid rgba(255, 255, 255, 0.05)"
                selected_badge = ""
                
            rows_html.append(f"""
            <div style="margin-bottom:10px; position:relative; padding:10px 12px; border-radius:10px; background:{row_bg}; border:{row_border}; box-shadow:0 2px 5px rgba(0,0,0,0.15);">
              <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
                <span style="font-family:'Space Grotesk',sans-serif; font-size:0.85rem; font-weight:600; color:{model_color}; display:flex; align-items:center;">
                  {name} {selected_badge}
                </span>
                <span style="font-family:'IBM Plex Mono',monospace; font-size:0.88rem; font-weight:700; color:{model_color};">{prob_pct:.1f}%</span>
              </div>
              <div style="height:8px; background:rgba(255,255,255,0.05); border-radius:4px; overflow:hidden; position:relative;">
                <div class="sim-bar-fill" data-target="{prob_pct:.2f}" 
                     style="position:absolute; left:0; top:0; height:100%; width:0%; 
                            background:linear-gradient(90deg, {model_color_fade}, {model_color}); 
                            border-radius:4px; box-shadow: 0 0 8px {model_color}50; 
                            transition: width 1.2s cubic-bezier(.1,.85,.25,1);"></div>
              </div>
            </div>
            """)
            
        with open("assets/multi_model_simulator.html") as f:
            template = f.read()
            
        html = template.replace("__ROWS__", "".join(rows_html))
        components.html(html, height=280)

    @staticmethod
    def render_contribution_chart(contributions: dict, accent: str):
        items = sorted(contributions.items(), key=lambda kv: abs(kv[1]), reverse=True)[:8]
        max_abs = max(abs(v) for _, v in items) or 1e-9
        rows_html = []
        for i, (name, val) in enumerate(items):
            pct = abs(val) / max_abs * 100
            is_pos = val >= 0
            bar_color = DesignSystem.PALETTE["high"] if is_pos else DesignSystem.PALETTE["low"]
            side = "right" if is_pos else "left"
            left_pos = "50%" if is_pos else f"calc(50% - {pct:.2f}%)"
            
            rows_html.append(f"""
            <div style="display:flex; align-items:center; gap:12px; margin:10px 0;">
              <div style="width:180px; font-size:0.85rem; font-weight:500; color:#E2E8F0; text-align:right; font-family:'Inter',sans-serif;">{name}</div>
              <div style="flex:1; position:relative; height:18px; background:rgba(255,255,255,0.04); border-radius:9px; overflow:hidden; border:1px solid rgba(255,255,255,0.03);">
                <div class="bar-fill" data-target="{pct:.2f}" data-side="{side}"
                     style="position:absolute; top:0; left:{left_pos}; height:100%; width:0%;
                            background:{bar_color}; border-radius:9px; transition: width 0.9s cubic-bezier(.15,.85,.3,1);
                            transition-delay:{i * 50}ms; box-shadow:0 0 8px {bar_color}90;"></div>
                <div style="position:absolute; left:50%; top:0; bottom:0; width:1px; background:rgba(255,255,255,0.2);"></div>
              </div>
              <div style="width:70px; font-family:'IBM Plex Mono',monospace; font-size:0.85rem; font-weight:600; color:{bar_color}; text-align:left;">{val:+.3f}</div>
            </div>
            """)

        with open("assets/contribution_chart.html") as f:
            template = f.read()
            
        html = (template
                .replace("__LOW_COLOR__", DesignSystem.PALETTE["low"])
                .replace("__HIGH_COLOR__", DesignSystem.PALETTE["high"])
                .replace("__ROWS__", "".join(rows_html)))
                
        components.html(html, height=44 * len(items) + 50)


# ------------------------------------------------------------------
# Page Class Architecture (OOD)
# ------------------------------------------------------------------
class BasePage(ABC):
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager

    @abstractmethod
    def render(self, selected_model_name: str, accent: str):
        pass


class PredictorPage(BasePage):
    def render(self, selected_model_name: str, accent: str):
        UIComponents.render_hero(
            "Diabetes Risk Predictor",
            f"Fill in the patient's diagnostic values below to estimate risk with "
            f"<b style='color:{accent}'>{selected_model_name}</b> and compare outcomes in real time."
        )

        with st.container(border=True):
            UIComponents.eyebrow("Diagnostics Intake Form")
            col1, col2 = st.columns(2)

            with col1:
                md("<div style='font-weight:600; font-size:0.95rem; margin-bottom:0.6rem; color:#FFFFFF;'>📋 Physiological Metrics</div>")
                pregnancies = st.slider(
                    "Pregnancies",
                    min_value=0,
                    max_value=20,
                    value=1,
                    step=1
                )         
                age = st.slider("Age", 18, 100, 35)
                bmi = st.slider("BMI", 0.0, 70.0, 25.0, step=0.1, help="0.0 triggers missing value imputation")
                dpf = st.slider("Diabetes Pedigree Function (DPF)", 0.0, 2.5, 0.5, step=0.01)

            with col2:
                md("<div style='font-weight:600; font-size:0.95rem; margin-bottom:0.6rem; color:#FFFFFF;'>🔬 Laboratory Observations</div>")
                glucose = st.slider("Glucose (mg/dL)", 0, 250, 120, help="0 triggers missing value imputation")
                blood_pressure = st.slider("Blood Pressure (mm Hg)", 0, 140, 70, help="0 triggers missing value imputation")
                skin_thickness = st.slider("Skin Thickness (mm)", 0, 100, 20, help="0 triggers missing value imputation")
                insulin = st.slider("Insulin (mu U/ml)", 0, 900, 80, help="0 triggers missing value imputation")

            md("<div style='margin-top:0.8rem'></div>")
            
            # Quick summary panel with dynamic coloring
            bmi_labels = ["Underweight", "Normal", "Overweight", "Obese"]
            age_labels = ["<30", "30-44", "45-59", "60+"]
            
            bmi_cat = ModelManager.bmi_category(bmi) if bmi > 0 else 1
            age_grp = ModelManager.age_group(age)
            high_glucose_label = "Yes" if glucose > 140 else "No"
            
            bmi_risks = ["low", "low", "moderate", "high"]
            age_risks = ["low", "low", "moderate", "high"]
            glucose_risk = "high" if glucose > 140 else "low"

            # Built as a single-line string (no leading indentation on any
            # line) so Streamlit's markdown parser can't mistake part of it
            # for an indented code block.
            summary_html = (
                '<div style="margin-bottom:0.8rem;">'
                '<span style="font-size:0.85rem; color:#94A3B8; font-weight:600; '
                'text-transform:uppercase; margin-right:10px;">Classification:</span>'
                + UIComponents.chip("BMI Profile", bmi_labels[bmi_cat], bmi_risks[bmi_cat])
                + UIComponents.chip("Age Bracket", age_labels[age_grp], age_risks[age_grp])
                + UIComponents.chip("Hyperglycemia", high_glucose_label, glucose_risk)
                + '</div>'
            )
            st.markdown(summary_html, unsafe_allow_html=True)

            # Interactive Predict button
            predict_clicked = st.button("⚡ Run Diagnostic Analysis", use_container_width=True)

        # Store session state for prediction results to persist on slider interactions
        current_inputs = {
            "pregnancies": pregnancies,
            "glucose": glucose,
            "blood_pressure": blood_pressure,
            "skin_thickness": skin_thickness,
            "insulin": insulin,
            "bmi": bmi,
            "dpf": dpf,
            "age": age,
            "model_name": selected_model_name
        }

        if "last_prediction" not in st.session_state:
            st.session_state.last_prediction = None
            st.session_state.last_inputs = None

        # Execute prediction if requested, or auto-recalculate if active model changes on a previously computed state
        if predict_clicked:
            with st.spinner("🧠 Booting diagnostic analytics scanner..."):
                import time
                time.sleep(1.2)
            st.session_state.last_prediction = self._run_prediction(current_inputs)
            st.session_state.last_inputs = current_inputs
        elif st.session_state.last_prediction is not None:
            # Auto-update if the user changes the active model on the sidebar while looking at results
            if st.session_state.last_inputs and st.session_state.last_inputs["model_name"] != selected_model_name:
                st.session_state.last_inputs["model_name"] = selected_model_name
                st.session_state.last_prediction = self._run_prediction(st.session_state.last_inputs)

        # Render prediction results
        if st.session_state.last_prediction is not None:
            pred_data = st.session_state.last_prediction
            prob = pred_data["prob"]
            tier = pred_data["tier"]
            color = pred_data["color"]
            tier_class = pred_data["tier_class"]
            input_scaled = pred_data["input_scaled"]
            all_probs = pred_data.get("all_probs", {})

            UIComponents.render_pulse_divider()

            res_col1, res_col2 = st.columns([1, 1.3])

            with res_col1:
                with st.container(border=True):
                    UIComponents.eyebrow("Biometric Scan")
                    UIComponents.render_gauge(prob * 100, color, tier)

            with res_col2:
                with st.container(border=True):
                    UIComponents.eyebrow("Diagnostic Report")
                    md(f'<span class="risk-badge risk-{tier_class}">● {tier} Risk Profile</span>')
                    md("<div style='margin-top:0.9rem'></div>")
                    st.metric("Diabetes Risk Score", f"{prob * 100:.1f}%")
                    st.caption(f"Calculated with active model: **{selected_model_name}**")
                    md("<div style='margin-top:0.6rem'></div>")
                    
                    if tier == "Low":
                        st.success("Analysis suggests **low diabetes risk** under these conditions. Health metrics are within baseline limits.")
                    elif tier == "Moderate":
                        st.warning("Analysis suggests **moderate diabetes risk**. Key markers are elevated. Consultation with a medical practitioner is advised.")
                    else:
                        st.error("Analysis suggests **high diabetes risk**. Multiple physiological features are outside safety limits. Immediate medical consultation is recommended.")

            UIComponents.render_pulse_divider()
            
            # Simulator & Explainability Side by Side
            col_sim, col_contrib = st.columns([1.1, 1.3])
            
            with col_sim:
                UIComponents.eyebrow("Multi-Model Simulator")
                UIComponents.render_multi_model_simulator(all_probs, selected_model_name)
                
            with col_contrib:
                UIComponents.eyebrow("Explainability")
                md('<div style="font-family:\'Space Grotesk\', sans-serif; font-size:1.1rem; font-weight:600; color:#FFFFFF; margin-bottom:8px;">Feature Contribution Analysis</div>')
                with st.container(border=True):
                    contributions, _ = self.model_manager.explain_prediction(selected_model_name, input_scaled)
                    UIComponents.render_contribution_chart(contributions, accent)
                    st.caption(
                        "Each feature's baseline average is swapped in one at a time and the model is re-run; "
                        "the resulting change in predicted probability is that feature's contribution. "
                        "Red bars push risk up, green bars pull it down."
                    )

    def _run_prediction(self, inputs: dict) -> dict:
        # Construct raw DataFrame
        df_raw = pd.DataFrame(
            [[
                inputs["pregnancies"],
                inputs["glucose"],
                inputs["blood_pressure"],
                inputs["skin_thickness"],
                inputs["insulin"],
                inputs["bmi"],
                inputs["dpf"],
                inputs["age"]
            ]],
            columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                     'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
        )

        # Impute and scale using model_manager
        input_scaled = self.model_manager.preprocess_inputs(df_raw)
        
        # Model inference
        prob = self.model_manager.predict_proba_row(inputs["model_name"], input_scaled)
        
        # Risk classification
        if prob < 0.3:
            tier, color, tier_class = "Low", DesignSystem.PALETTE["low"], "low"
        elif prob < 0.6:
            tier, color, tier_class = "Moderate", DesignSystem.PALETTE["moderate"], "moderate"
        else:
            tier, color, tier_class = "High", DesignSystem.PALETTE["high"], "high"

        # Calculate all model probabilities for the simulator comparison
        all_probs = {}
        for name in self.model_manager.models.keys():
            all_probs[name] = self.model_manager.predict_proba_row(name, input_scaled)

        return {
            "prob": prob,
            "tier": tier,
            "color": color,
            "tier_class": tier_class,
            "input_scaled": input_scaled,
            "all_probs": all_probs
        }


class ComparisonPage(BasePage):
    def render(self, selected_model_name: str, accent: str):
        UIComponents.render_hero(
            "Model Comparison",
            "Logistic Regression, Linear Regression, Random Forest, and Decision Tree — "
            "side by side on the same held-out test set (20%, stratified split)."
        )

        UIComponents.eyebrow("Test set metrics")
        cols = st.columns(4)
        for col, name in zip(cols, self.model_manager.models.keys()):
            with col:
                with st.container(border=True):
                    md(f'<div style="font-family:\'Space Grotesk\',sans-serif; font-weight:600; '
                       f'color:{DesignSystem.MODEL_ACCENTS[name]}; font-size:0.95rem;">{name}</div>')
                    m = self.model_manager.metrics[name]
                    st.metric("Accuracy", f"{m['accuracy']:.2f}")
                    st.metric("ROC-AUC", f"{m['roc_auc']:.2f}")
                    st.caption(f"Precision {m['precision']:.2f} · Recall {m['recall']:.2f} · F1 {m['f1']:.2f}")

        UIComponents.render_pulse_divider()

        col_a, col_b = st.columns(2)

        with col_a:
            with st.container(border=True):
                UIComponents.eyebrow("ROC curves — all 4 models")
                fig = go.Figure()
                for name in self.model_manager.models.keys():
                    rd = self.model_manager.roc_data[name]
                    fig.add_trace(go.Scatter(
                        x=rd["fpr"], y=rd["tpr"], mode="lines", name=f"{name} ({self.model_manager.metrics[name]['roc_auc']:.3f})",
                        line=dict(color=DesignSystem.MODEL_ACCENTS[name], width=2.5)
                    ))
                fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                          line=dict(color="rgba(255,255,255,0.2)", dash="dash"),
                                          showlegend=False))
                fig.update_layout(
                    height=380, margin=dict(l=15, r=15, t=30, b=15),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color=DesignSystem.PALETTE["text"], family="Plus Jakarta Sans"),
                    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10, family="Plus Jakarta Sans")),
                    xaxis=dict(title="False Positive Rate", gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.1)"),
                    yaxis=dict(title="True Positive Rate", gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.1)"),
                )
                st.plotly_chart(fig, use_container_width=True)

        with col_b:
            with st.container(border=True):
                UIComponents.eyebrow("Metric comparison")
                metric_names = ["accuracy", "precision", "recall", "f1", "roc_auc"]
                fig2 = go.Figure()
                for name in self.model_manager.models.keys():
                    fig2.add_trace(go.Bar(
                        x=metric_names, y=[self.model_manager.metrics[name][mn] for mn in metric_names],
                        name=name, marker_color=DesignSystem.MODEL_ACCENTS[name]
                    ))
                fig2.update_layout(
                    barmode="group", height=380, margin=dict(l=15, r=15, t=30, b=15),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color=DesignSystem.PALETTE["text"], family="Plus Jakarta Sans"),
                    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10, family="Plus Jakarta Sans")),
                    yaxis=dict(range=[0, 1], gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.1)"),
                    xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                )
                st.plotly_chart(fig2, use_container_width=True)

        UIComponents.render_pulse_divider()

        with st.container(border=True):
            UIComponents.eyebrow("Feature importance")
            st.write("Pick a model to see which features it leans on most:")
            fi_model = st.selectbox("Model", list(self.model_manager.models.keys()), label_visibility="collapsed")
            fi = self.model_manager.feature_importance[fi_model]
            sorted_items = sorted(fi.items(), key=lambda kv: kv[1])
            fig3 = go.Figure(go.Bar(
                x=[v for _, v in sorted_items], y=[k for k, _ in sorted_items],
                orientation="h", marker_color=DesignSystem.MODEL_ACCENTS[fi_model]
            ))
            fig3.update_layout(
                height=380, margin=dict(l=15, r=15, t=20, b=15),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color=DesignSystem.PALETTE["text"], family="Plus Jakarta Sans"),
                xaxis=dict(title="|coefficient| or importance", gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.1)"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
            )
            st.plotly_chart(fig3, use_container_width=True)

        UIComponents.render_pulse_divider()

        with st.container(border=True):
            UIComponents.eyebrow("Confusion matrices")
            cm_cols = st.columns(4)
            for c, name in zip(cm_cols, self.model_manager.models.keys()):
                with c:
                    cm = np.array(self.model_manager.metrics[name]["confusion_matrix"])
                    fig4 = go.Figure(data=go.Heatmap(
                        z=cm, x=["Pred 0", "Pred 1"], y=["Actual 0", "Actual 1"],
                        colorscale=[[0, "rgba(255,255,255,0.02)"], [1, DesignSystem.MODEL_ACCENTS[name]]],
                        showscale=False, text=cm, texttemplate="%{text}",
                        textfont=dict(color=DesignSystem.PALETTE["text"], family="Plus Jakarta Sans", size=12)
                    ))
                    fig4.update_layout(
                        height=220, margin=dict(l=15, r=15, t=35, b=15),
                        title=dict(text=name, font=dict(size=12, color=DesignSystem.PALETTE["text"], family="Space Grotesk")),
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(color=DesignSystem.PALETTE["muted"], size=10, family="Plus Jakarta Sans"),
                    )
                    st.plotly_chart(fig4, use_container_width=True)


class AboutPage(BasePage):
    def render(self, selected_model_name: str, accent: str):
        UIComponents.render_hero(
            "About This Project",
            "The data, methodology, and limitations behind this predictor."
        )

        with st.container(border=True):
            UIComponents.eyebrow("What this is")
            st.markdown("""
            A set of four models that estimate diabetes risk from a handful of diagnostic
            measurements, trained on the Pima Indians Diabetes Dataset, so you can compare
            how a linear model, a probabilistic classifier, and two tree-based methods
            each read the same patient data.
            """)

        UIComponents.render_pulse_divider()

        col_a, col_b = st.columns(2)
        with col_a:
            with st.container(border=True):
                UIComponents.eyebrow("Data")
                st.markdown("""
                - Pima Indians Diabetes Database (UCI / Kaggle)
                - 768 records, 8 diagnostic features
                """)
        with col_b:
            with st.container(border=True):
                UIComponents.eyebrow("How it was built")
                st.markdown("""
                - Zero values in fields where zero isn't physically possible
                  (glucose, blood pressure, etc.) treated as missing, imputed with KNN
                - Class imbalance handled with a from-scratch SMOTE-style oversampler
                  (k-nearest-neighbor interpolation on the minority class)
                - **Logistic Regression, Linear Regression, Random Forest, and Decision
                  Tree** trained on the same preprocessed features and compared
                - Explanations generated with a dependency-free, leave-one-out-from-baseline
                  method — no SHAP/XGBoost install required to run this app
                """)

        UIComponents.render_pulse_divider()

        with st.container(border=True):
            UIComponents.eyebrow("Limitations worth knowing")
            st.markdown("""
            - The dataset is small and specific (Pima Indian women, age 21+), so it
              won't generalize well to a broader population
            - Linear Regression isn't a natural fit for a yes/no outcome — it's included
              here as a baseline comparison, not a recommended choice
            - Not validated for clinical use in any capacity
            """)
            st.markdown("---")
            st.caption(
                "This is a learning project, not a medical device. It shouldn't replace "
                "advice from an actual healthcare provider."
            )

        st.markdown("<div style='margin-top:1rem'></div>", unsafe_allow_html=True)
        st.caption("Built with scikit-learn, Plotly, and Streamlit.")


# ------------------------------------------------------------------
# Main Orchestration Class
# ------------------------------------------------------------------
class DiabetesRiskApp:
    def __init__(self):
        self.model_manager = ModelManager()
        self.pages = {
            "Predictor": PredictorPage(self.model_manager),
            "Model Comparison": ComparisonPage(self.model_manager),
            "About": AboutPage(self.model_manager)
        }

    def run(self):
        # Ambient background effect: glowing bio-signal motes drifting upward
        UIComponents.render_ambient_particles()

        # Sidebar Title & Navigation
        st.sidebar.title("🩺 Diabetes Risk")
        
        st.sidebar.markdown('<div class="nav-picker-wrapper">', unsafe_allow_html=True)
        selected_page_name = st.sidebar.radio(
            "Go to", list(self.pages.keys()), label_visibility="collapsed"
        )
        st.sidebar.markdown('</div>', unsafe_allow_html=True)

        st.sidebar.markdown("<div style='margin-top:1.2rem'></div>", unsafe_allow_html=True)
        st.sidebar.markdown('<div class="eyebrow">Active model</div>', unsafe_allow_html=True)

        # Model Selector radio wrapped in class for custom styling
        st.sidebar.markdown('<div class="model-picker-wrapper">', unsafe_allow_html=True)
        selected_model_name = st.sidebar.radio(
            "Model", list(self.model_manager.models.keys()), label_visibility="collapsed", key="model_choice"
        )
        st.sidebar.markdown('</div>', unsafe_allow_html=True)
        accent = DesignSystem.MODEL_ACCENTS[selected_model_name]

        # Inject dynamic styling matching active model accent
        DesignSystem.apply_custom_css(accent)

        # Model metrics summary card (built as a single-line string —
        # see the `md()` helper's docstring for why that matters)
        st.sidebar.markdown(
            '<div style="font-family:\'IBM Plex Mono\',monospace; font-size:0.78rem; color:' + accent + '; '
            'background:rgba(255,255,255,0.05); border:1px solid rgba(255,255,255,0.1); border-radius:10px; '
            'padding:0.5rem 0.7rem; margin-top:0.4rem; box-shadow: 0 4px 10px rgba(0,0,0,0.15);">'
            f"ROC-AUC: {self.model_manager.metrics[selected_model_name]['roc_auc']:.3f} &nbsp;|&nbsp; "
            f"F1: {self.model_manager.metrics[selected_model_name]['f1']:.3f}"
            '</div>',
            unsafe_allow_html=True,
        )

        st.sidebar.markdown("<div style='margin-top:1.5rem'></div>", unsafe_allow_html=True)
        st.sidebar.warning(
            "This tool is for educational purposes only and isn't a substitute "
            "for professional medical advice, diagnosis, or treatment."
        )

        # Route to selected page
        self.pages[selected_page_name].render(selected_model_name, accent)


if __name__ == "__main__":
    app = DiabetesRiskApp()
    app.run()