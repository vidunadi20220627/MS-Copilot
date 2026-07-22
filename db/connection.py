from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from config.settings import DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD

DATABASE_URL = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{10.8.2.1}:{13306}/{kinsure}"

# Create engine
engine = create_engine(
    DATABASE_URL,
    pool_size=5,
    pool_recycle=3600,
    pool_pre_ping=True    
)

# Create session
SessionLocal = sessionmaker(bind=engine)

def get_db_connection():
    """Get database connection"""
    return engine.connect()

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

def get_policy_wording_credentials(policy_no: str) -> dict:
    """
    Get access_token from policy wording view
    View filters: status='A', trans_type='NB', product='TPS'
    Returns latest record based on timestamp_updated
    """
    query = text("""
        SELECT policy_no, access_token
        FROM vw_policy_wording_access
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
        print(f"Error fetching policy credentials: {e}")
        return None

def get_latest_payment_status(policy_no: str) -> dict:
    """Get current payment status — View 1"""
    query = text("""
        SELECT *
        FROM latest_payment_status
        WHERE policy_no = :policy_no
    """)
    try:
        with engine.connect() as conn:
            result = conn.execute(query, {"policy_no": policy_no})
            row = result.fetchone()
            if row:
                return dict(row._mapping)
            return None
    except Exception as e:
        print(f"Error fetching payment status: {e}")
        return None

def get_payment_method_summary(policy_no: str = None, customer_id: str = None) -> list:
    """Get payment method summary — View 2"""
    if policy_no:
        query = text("""
            SELECT * FROM payment_method_summary
            WHERE policy_no = :policy_no
        """)
        params = {"policy_no": policy_no}
    else:
        query = text("""
            SELECT * FROM payment_method_summary
            WHERE customer_id = :customer_id
        """)
        params = {"customer_id": customer_id}

    try:
        with engine.connect() as conn:
            result = conn.execute(query, params)
            return [dict(row._mapping) for row in result.fetchall()]
    except Exception as e:
        print(f"Error fetching payment method summary: {e}")
        return []

def get_payment_amount_details(policy_no: str) -> dict:
    """Get premium and charge amounts — View 3"""
    query = text("""
        SELECT * FROM payment_amount_details
        WHERE policy_no = :policy_no
    """)
    try:
        with engine.connect() as conn:
            result = conn.execute(query, {"policy_no": policy_no})
            row = result.fetchone()
            if row:
                return dict(row._mapping)
            return None
    except Exception as e:
        print(f"Error fetching payment amounts: {e}")
        return None

def get_transaction_history(policy_no: str = None, customer_id: str = None) -> list:
    """Get full transaction history — View 4"""
    if policy_no:
        query = text("""
            SELECT * FROM transaction_history
            WHERE policy_no = :policy_no
            ORDER BY payment_date DESC
        """)
        params = {"policy_no": policy_no}
    else:
        query = text("""
            SELECT * FROM transaction_history
            WHERE customer_id = :customer_id
            ORDER BY payment_date DESC
        """)
        params = {"customer_id": customer_id}

    try:
        with engine.connect() as conn:
            result = conn.execute(query, params)
            return [dict(row._mapping) for row in result.fetchall()]
    except Exception as e:
        print(f"Error fetching transaction history: {e}")
        return []

def get_failed_payments(policy_no: str = None, customer_id: str = None) -> list:
    """Get failed payment records — View 5"""
    if policy_no:
        query = text("""
            SELECT * FROM failed_payments
            WHERE policy_no = :policy_no
        """)
        params = {"policy_no": policy_no}
    else:
        query = text("""
            SELECT * FROM failed_payments
            WHERE customer_id = :customer_id
        """)
        params = {"customer_id": customer_id}

    try:
        with engine.connect() as conn:
            result = conn.execute(query, params)
            return [dict(row._mapping) for row in result.fetchall()]
    except Exception as e:
        print(f"Error fetching failed payments: {e}")
        return []

def get_payment_method_analytics(customer_id: str) -> list:
    """Get payment method analytics — View 6"""
    query = text("""
        SELECT * FROM payment_method_analytics
        WHERE customer_id = :customer_id
    """)
    try:
        with engine.connect() as conn:
            result = conn.execute(query, {"customer_id": customer_id})
            return [dict(row._mapping) for row in result.fetchall()]
    except Exception as e:
        print(f"Error fetching analytics: {e}")
        return []

def get_recent_transactions(customer_id: str) -> list:
    """Get recent transactions — View 7"""
    query = text("""
        SELECT * FROM recent_transactions
        WHERE customer_id = :customer_id
        ORDER BY payment_date DESC
    """)
    try:
        with engine.connect() as conn:
            result = conn.execute(query, {"customer_id": customer_id})
            return [dict(row._mapping) for row in result.fetchall()]
    except Exception as e:
        print(f"Error fetching recent transactions: {e}")
        return []