"""
Helper Utilities
"""
import random
from typing import List


def random_sample(categories: dict) -> dict:
    """Generate a random but valid applicant profile"""
    return {
        "CODE_GENDER": random.choice(categories.get("CODE_GENDER", ["M", "F"])),
        "FLAG_OWN_CAR": random.choice([0, 1]),
        "FLAG_OWN_REALTY": random.choice([0, 1]),
        "CNT_CHILDREN": random.randint(0, 5),
        "DEBT_TO_INCOME": round(random.uniform(0.5, 12.0), 2),
        "AVERAGE_EXTERNAL_RATING": round(random.uniform(0.05, 0.95), 2),
        "N_DOCUMENTS_PROVIDED": random.randint(0, 5),
        "ADDITIONAL_DOC_PROVIDED": random.choice([True, False]),
        "TOT_PREV_APP": random.randint(0, 10),
        "APPROVED_RATIO": round(random.uniform(0.0, 1.0), 2),
        "REFUSED_RATIO": round(random.uniform(0.0, 1.0), 2),
        "CANCELLED_RATIO": round(random.uniform(0.0, 0.5), 2),
        "UNUSED_RATIO": round(random.uniform(0.0, 0.3), 2),
        "NAME_TYPE_SUITE": random.choice(categories.get("NAME_TYPE_SUITE", ["Unaccompanied"])),
        "NAME_INCOME_TYPE": random.choice(categories.get("NAME_INCOME_TYPE", ["Working"])),
        "NAME_FAMILY_STATUS": random.choice(categories.get("NAME_FAMILY_STATUS", ["Married"])),
        "NAME_HOUSING_TYPE": random.choice(categories.get("NAME_HOUSING_TYPE", ["House / apartment"])),
        "REGION_RATING_CLIENT": random.choice([1, 2, 3]),
        "YEARS_BIRTH": round(random.uniform(20.0, 68.0), 1),
        "YEARS_EMPLOYED": round(random.uniform(0.0, 30.0), 1),
        "YEARS_REGISTRATION": round(random.uniform(0.0, 40.0), 1),
        "YEARS_ID_PUBLISH": round(random.uniform(0.0, 20.0), 1),
        "YEARS_LAST_PHONE_CHANGE": round(random.uniform(0.0, 10.0), 1),
        "ORG_GROUP": random.choice(categories.get("ORG_GROUP", ["Business"])),
        "OCCUPATION_TYPE_GROUPED": random.choice(categories.get("OCCUPATION_TYPE_GROUPED", ["WhiteCollar/Admin"])),
        "EDUCATION_LEVEL": random.choice(categories.get("EDUCATION_LEVEL", ["Medium Education"])),
    }


def get_category_options(categories: dict, field: str) -> List[str]:
    """Get category options for a field"""
    return categories.get(field, [])
