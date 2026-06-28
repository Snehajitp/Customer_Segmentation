APP_HOST = "0.0.0.0"
APP_PORT = 8000

# Model artifacts
MODEL_PATH = "artifacts/model.pkl"
PREPROCESSOR_PATH = "artifacts/preprocessing.pkl"

# Schema file
SCHEMA_FILE_PATH = "config/schema.yaml"

# Feature schema (must match training order)
FEATURE_DTYPES = {
    "Age": float,
    "Education": float,
    "Marital Status": float,
    "Parental Status": float,
    "Children": float,
    "Income": float,
    "Total_Spending": float,
    "Days_as_Customer": float,
    "Recency": float,
    "Wines": float,
    "Fruits": float,
    "Meat": float,
    "Fish": float,
    "Sweets": float,
    "Gold": float,
    "Web": float,
    "Catalog": float,
    "Store": float,
    "Discount Purchases": float,
    "Total Promo": float,
    "NumWebVisitsMonth": float,
}

FEATURE_COLUMNS = list(FEATURE_DTYPES.keys())