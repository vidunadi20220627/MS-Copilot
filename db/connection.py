from typing import Optional
from sqlalchemy import create_engine, text
from config.settings import (
    DB_HOST, DB_PORT,
    DB_NAME, DB_USER, DB_PASSWORD
)

DATABASE_URL = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

engine = create_engine(
    DATABASE_URL,
    pool_size=5,
    pool_recycle=3600,
    pool_pre_ping=True
)

def get_engine():
    """Return engine for Vanna to use"""
    return engine

def test_connection():
    """Test DB connection on startup"""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            print("MySQL DB connected ✅")
            return True
    except Exception as e:
        print(f"MySQL DB connection failed ❌: {e}")
        return False

def get_latest_policy_wording_credentials() -> Optional[dict]:
    """
    Get access_token and policy_no from the latest
    policy wording record across ALL policies
    Used when user does NOT provide a policy number
    Filters: status='A', trans_type='NB', product='TPS'
    Returns the single most recently updated record
    """
    query = text("""
        SELECT policy_no, access_token
        FROM view_policy_wording_schedule
        WHERE status = 'A'
        AND trans_type = 'NB'
        AND product = 'TPS'
        ORDER BY timestamp_updated DESC
        LIMIT 1
    """)
    try:
        with engine.connect() as conn:
            result = conn.execute(query)
            row = result.fetchone()
            if row:
                return {
                    "policy_no": row.policy_no,
                    "access_token": row.access_token
                }
            return None
    except Exception as e:
        print(f"Error fetching latest wording credentials: {e}")
        return None

def get_policy_credentials_by_no(policy_no: str) -> Optional[dict]:
    """
    Get access_token for a SPECIFIC policy number
    Used when user provides a policy number in their question
    Filters: status='A', trans_type='NB', product='TPS'
    Returns the latest record for that specific policy
    """
    query = text("""
        SELECT policy_no, access_token
        FROM view_policy_wording_schedule
        WHERE policy_no = :policy_no
        AND status = 'A'
        AND trans_type = 'NB'
        AND product = 'TPS'
        ORDER BY timestamp_updated DESC
        LIMIT 1
    """)
    try:
        with engine.connect() as conn:
            result = conn.execute(query, {"policy_no": policy_no})
            row = result.fetchone()
            if row:
                return {
                    "policy_no": row.policy_no,
                    "access_token": row.access_token
                }
            return None
    except Exception as e:
        print(f"Error fetching credentials for policy {policy_no}: {e}")
        return None