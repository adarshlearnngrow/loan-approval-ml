"""
Application Constants - Sample Data and Feature Definitions
"""

# ── Sample Applicant Profiles ──────────────────────────────────────────────────
SAMPLE_PROFILES = {
    "Low Risk": {
        "CODE_GENDER": "F",
        "FLAG_OWN_CAR": 1,
        "FLAG_OWN_REALTY": 1,
        "CNT_CHILDREN": 0,
        "DEBT_TO_INCOME": 1.2,
        "AVERAGE_EXTERNAL_RATING": 0.72,
        "N_DOCUMENTS_PROVIDED": 3,
        "ADDITIONAL_DOC_PROVIDED": True,
        "TOT_PREV_APP": 3,
        "APPROVED_RATIO": 0.8,
        "REFUSED_RATIO": 0.1,
        "CANCELLED_RATIO": 0.1,
        "UNUSED_RATIO": 0.0,
        "NAME_TYPE_SUITE": "Unaccompanied",
        "NAME_INCOME_TYPE": "Working",
        "NAME_FAMILY_STATUS": "Married",
        "NAME_HOUSING_TYPE": "House / apartment",
        "REGION_RATING_CLIENT": 1,
        "YEARS_BIRTH": 42.0,
        "YEARS_EMPLOYED": 8.5,
        "YEARS_REGISTRATION": 15.0,
        "YEARS_ID_PUBLISH": 5.0,
        "YEARS_LAST_PHONE_CHANGE": 3.0,
        "ORG_GROUP": "Business",
        "OCCUPATION_TYPE_GROUPED": "WhiteCollar/Admin",
        "EDUCATION_LEVEL": "Higher Academic",
    },
    "High Risk": {
        "CODE_GENDER": "M",
        "FLAG_OWN_CAR": 0,
        "FLAG_OWN_REALTY": 0,
        "CNT_CHILDREN": 3,
        "DEBT_TO_INCOME": 8.9,
        "AVERAGE_EXTERNAL_RATING": 0.15,
        "N_DOCUMENTS_PROVIDED": 1,
        "ADDITIONAL_DOC_PROVIDED": False,
        "TOT_PREV_APP": 5,
        "APPROVED_RATIO": 0.2,
        "REFUSED_RATIO": 0.6,
        "CANCELLED_RATIO": 0.2,
        "UNUSED_RATIO": 0.0,
        "NAME_TYPE_SUITE": "Unaccompanied",
        "NAME_INCOME_TYPE": "Working",
        "NAME_FAMILY_STATUS": "Single / not married",
        "NAME_HOUSING_TYPE": "Rented apartment",
        "REGION_RATING_CLIENT": 3,
        "YEARS_BIRTH": 28.0,
        "YEARS_EMPLOYED": 0.5,
        "YEARS_REGISTRATION": 2.0,
        "YEARS_ID_PUBLISH": 0.5,
        "YEARS_LAST_PHONE_CHANGE": 0.1,
        "ORG_GROUP": "Unknown/Other",
        "OCCUPATION_TYPE_GROUPED": "Labour/LowSkill",
        "EDUCATION_LEVEL": "Medium Education",
    },
}

# ── Feature Metadata ───────────────────────────────────────────────────────────
FEATURE_GROUPS = {
    "personal": [
        "CODE_GENDER", "FLAG_OWN_CAR", "FLAG_OWN_REALTY", "CNT_CHILDREN",
        "NAME_FAMILY_STATUS", "NAME_TYPE_SUITE", "ADDITIONAL_DOC_PROVIDED"
    ],
    "age_history": [
        "YEARS_BIRTH", "YEARS_EMPLOYED", "YEARS_REGISTRATION",
        "YEARS_ID_PUBLISH", "YEARS_LAST_PHONE_CHANGE"
    ],
    "financial": [
        "DEBT_TO_INCOME", "NAME_INCOME_TYPE", "EDUCATION_LEVEL",
        "NAME_HOUSING_TYPE", "ORG_GROUP", "OCCUPATION_TYPE_GROUPED", "N_DOCUMENTS_PROVIDED"
    ],
    "credit_region": [
        "AVERAGE_EXTERNAL_RATING", "REGION_RATING_CLIENT"
    ],
    "history": [
        "TOT_PREV_APP", "APPROVED_RATIO", "REFUSED_RATIO",
        "CANCELLED_RATIO", "UNUSED_RATIO"
    ]
}

# ── All Feature Names (ordered) ────────────────────────────────────────────────
ALL_FEATURES = [
    "CODE_GENDER", "FLAG_OWN_CAR", "FLAG_OWN_REALTY", "CNT_CHILDREN",
    "DEBT_TO_INCOME", "AVERAGE_EXTERNAL_RATING", "N_DOCUMENTS_PROVIDED",
    "ADDITIONAL_DOC_PROVIDED", "TOT_PREV_APP", "APPROVED_RATIO",
    "REFUSED_RATIO", "CANCELLED_RATIO", "UNUSED_RATIO", "NAME_TYPE_SUITE",
    "NAME_INCOME_TYPE", "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE",
    "REGION_RATING_CLIENT", "YEARS_BIRTH", "YEARS_EMPLOYED",
    "YEARS_REGISTRATION", "YEARS_ID_PUBLISH", "YEARS_LAST_PHONE_CHANGE",
    "ORG_GROUP", "OCCUPATION_TYPE_GROUPED", "EDUCATION_LEVEL"
]
