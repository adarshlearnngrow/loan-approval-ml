"""
Input Form Component - Renders the applicant data input form
"""
import streamlit as st
from typing import Callable

from ..utils.helpers import random_sample, get_category_options
from ..utils.validators import validate_ratios
from ..config.constants import SAMPLE_PROFILES


def render_input_form(categories: dict) -> dict:
    """
    Render the complete input form for applicant data
    
    Args:
        categories: Dictionary of category options for select fields
        
    Returns:
        dict: Collected input data from form
    """
    # Quick-fill buttons
    col_lbl, col_random, col_sp = st.columns([2, 1, 5])
    with col_lbl:
        st.markdown(
            "<div style='padding-top:0.5rem;color:#64748b;font-size:0.85rem;'>"
            "Quick Fill:</div>",
            unsafe_allow_html=True
        )
    with col_random:
        fill_random = st.button("Generate Random", key="btn_random")
    
    # Initialize session state
    if "D" not in st.session_state:
        st.session_state["D"] = SAMPLE_PROFILES["Low Risk"].copy()
    
    if fill_random:
        st.session_state["D"] = random_sample(categories)
        st.rerun()
    
    D = st.session_state["D"]
    
    st.markdown('<div class="thin-divider"></div>', unsafe_allow_html=True)
    
    # ── Section 1: Personal Details ──────────────────────────────────────────
    st.markdown("### Personal Details")
    with st.container():
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            gender = _selectbox("Gender", "CODE_GENDER", D["CODE_GENDER"], categories)
        with c2:
            own_car = _yes_no_box("Owns a Car", "FLAG_OWN_CAR", D["FLAG_OWN_CAR"])
        with c3:
            own_realty = _yes_no_box("Owns Real Estate", "FLAG_OWN_REALTY", D["FLAG_OWN_REALTY"])
        with c4:
            cnt_children = st.number_input(
                "Number of Children", min_value=0, max_value=20,
                value=int(D["CNT_CHILDREN"]), step=1, key="w_children"
            )
        
        c1, c2, c3 = st.columns(3)
        with c1:
            family_status = _selectbox("Family Status", "NAME_FAMILY_STATUS", 
                                       D["NAME_FAMILY_STATUS"], categories)
        with c2:
            type_suite = _selectbox("Accompanied By", "NAME_TYPE_SUITE", 
                                    D["NAME_TYPE_SUITE"], categories)
        with c3:
            add_doc = st.selectbox(
                "Additional Docs Provided", ["No", "Yes"],
                index=1 if D["ADDITIONAL_DOC_PROVIDED"] else 0,
                key="w_adddoc"
            )
            add_doc_val = add_doc == "Yes"
    
    # ── Section 2: Age & Life Events ─────────────────────────────────────────
    st.markdown("### Age & Life History")
    with st.container():
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            years_birth = _number_input("Age (years)", D["YEARS_BIRTH"], 18, 100, 0.1)
        with c2:
            years_employed = _number_input("Years Employed", D["YEARS_EMPLOYED"], -1, 60, 0.1)
        with c3:
            years_registration = _number_input("Years Registered", D["YEARS_REGISTRATION"], 0, 60, 0.1)
        with c4:
            years_id_publish = _number_input("Years Since ID", D["YEARS_ID_PUBLISH"], 0, 30, 0.1)
        with c5:
            years_phone = _number_input("Years Since Phone", D["YEARS_LAST_PHONE_CHANGE"], 0, 30, 0.1)
    
    # ── Section 3: Financial & Employment ────────────────────────────────────
    st.markdown("### Financial & Employment")
    with st.container():
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            debt_to_income = _number_input("Debt to Income", D["DEBT_TO_INCOME"], 0.0, 50.0, 0.01)
        with c2:
            income_type = _selectbox("Income Type", "NAME_INCOME_TYPE", 
                                     D["NAME_INCOME_TYPE"], categories)
        with c3:
            education = _selectbox("Education Level", "EDUCATION_LEVEL", 
                                   D["EDUCATION_LEVEL"], categories)
        with c4:
            housing = _selectbox("Housing Type", "NAME_HOUSING_TYPE", 
                                 D["NAME_HOUSING_TYPE"], categories)
        
        c1, c2, c3 = st.columns(3)
        with c1:
            org_group = _selectbox("Organisation Group", "ORG_GROUP", 
                                   D["ORG_GROUP"], categories)
        with c2:
            occupation = _selectbox("Occupation Type", "OCCUPATION_TYPE_GROUPED", 
                                    D["OCCUPATION_TYPE_GROUPED"], categories)
        with c3:
            n_docs = st.number_input(
                "Documents Provided", min_value=0, max_value=20,
                value=int(D["N_DOCUMENTS_PROVIDED"]), step=1, key="w_ndocs"
            )
    
    # ── Section 4: Credit & Region ───────────────────────────────────────────
    st.markdown("### Credit Ratings & Region")
    with st.container():
        c1, c2, c3 = st.columns(3)
        with c1:
            avg_ext_rating = st.slider(
                "External Credit Rating (0=worst, 1=best)",
                0.0, 1.0, float(D["AVERAGE_EXTERNAL_RATING"]),
                step=0.001, format="%.3f", key="w_rating"
            )
        with c2:
            region_rating = st.selectbox(
                "Region Risk (1=best, 3=worst)", [1, 2, 3],
                index=[1, 2, 3].index(int(D["REGION_RATING_CLIENT"])),
                key="w_region"
            )
        with c3:
            st.markdown("")  # spacer
    
    # ── Section 5: Previous Applications ─────────────────────────────────────
    st.markdown("### Previous Application History")
    with st.container():
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            tot_prev_app = st.number_input(
                "Total Prev. Applications", min_value=0, max_value=50,
                value=int(D["TOT_PREV_APP"]), step=1, key="w_prevapp"
            )
        with c2:
            approved_ratio = st.slider("Approved Ratio", 0.0, 1.0, 
                                       float(D["APPROVED_RATIO"]), 0.01, key="w_apr")
        with c3:
            refused_ratio = st.slider("Refused Ratio", 0.0, 1.0, 
                                      float(D["REFUSED_RATIO"]), 0.01, key="w_ref")
        with c4:
            cancelled_ratio = st.slider("Cancelled Ratio", 0.0, 1.0, 
                                        float(D["CANCELLED_RATIO"]), 0.01, key="w_canc")
        with c5:
            unused_ratio = st.slider("Unused Ratio", 0.0, 1.0, 
                                     float(D["UNUSED_RATIO"]), 0.01, key="w_unu")
    
    # Ratio validation
    is_valid, msg = validate_ratios(
        approved_ratio, refused_ratio, cancelled_ratio, unused_ratio, tot_prev_app
    )
    if not is_valid:
        st.warning(msg)
    
    # Compile input data
    return {
        "CODE_GENDER": gender,
        "FLAG_OWN_CAR": own_car,
        "FLAG_OWN_REALTY": own_realty,
        "CNT_CHILDREN": cnt_children,
        "DEBT_TO_INCOME": debt_to_income,
        "AVERAGE_EXTERNAL_RATING": avg_ext_rating,
        "N_DOCUMENTS_PROVIDED": n_docs,
        "ADDITIONAL_DOC_PROVIDED": add_doc_val,
        "TOT_PREV_APP": tot_prev_app,
        "APPROVED_RATIO": approved_ratio,
        "REFUSED_RATIO": refused_ratio,
        "CANCELLED_RATIO": cancelled_ratio,
        "UNUSED_RATIO": unused_ratio,
        "NAME_TYPE_SUITE": type_suite,
        "NAME_INCOME_TYPE": income_type,
        "NAME_FAMILY_STATUS": family_status,
        "NAME_HOUSING_TYPE": housing,
        "REGION_RATING_CLIENT": region_rating,
        "YEARS_BIRTH": years_birth,
        "YEARS_EMPLOYED": years_employed,
        "YEARS_REGISTRATION": years_registration,
        "YEARS_ID_PUBLISH": years_id_publish,
        "YEARS_LAST_PHONE_CHANGE": years_phone,
        "ORG_GROUP": org_group,
        "OCCUPATION_TYPE_GROUPED": occupation,
        "EDUCATION_LEVEL": education,
    }


def _selectbox(label: str, field: str, value: str, categories: dict) -> str:
    """Helper for category selectbox"""
    opts = get_category_options(categories, field)
    idx = opts.index(value) if value in opts else 0
    return st.selectbox(label, opts, index=idx, key=f"w_{field}")


def _yes_no_box(label: str, field: str, value: int) -> int:
    """Helper for Yes/No selectbox returning 0 or 1"""
    choice = st.selectbox(
        label, ["No", "Yes"],
        index=1 if int(value) == 1 else 0,
        key=f"w_{field}"
    )
    return 1 if choice == "Yes" else 0


def _number_input(label: str, value: float, mn: float, mx: float, 
                  step: float, fmt: str = "%.2f") -> float:
    """Helper for number input"""
    return st.number_input(
        label, value=float(value), min_value=float(mn),
        max_value=float(mx), step=step, format=fmt
    )
