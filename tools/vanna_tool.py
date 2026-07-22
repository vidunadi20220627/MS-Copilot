from vanna.openai import OpenAI_Chat
from vanna.chromadb import ChromaDB_VectorStore
from config.settings import (
    OPENAI_API_KEY,
    DB_HOST,
    DB_PORT,
    DB_NAME,
    DB_USER,
    DB_PASSWORD,
    CHROMA_DB_PATH
)

class InsuranceVanna(ChromaDB_VectorStore, OpenAI_Chat):
    def __init__(self, config=None):
        ChromaDB_VectorStore.__init__(self, config=config)
        OpenAI_Chat.__init__(self, config=config)

# Initialize Vanna
vn = InsuranceVanna(config={
    "api_key": OPENAI_API_KEY,
    "model": "gpt-4o",
    "path": f"{CHROMA_DB_PATH}/vanna"
})

def setup_vanna():
    """Connect Vanna to MySQL and train with views"""
    # Connect to MySQL
    vn.connect_to_mysql(
        host=DB_HOST,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        port=DB_PORT
    )

    # Train with view documentation
    vn.train(documentation="""
        Database views available:
        
        latest_payment_status
        → Use for current payment status of a policy
        → Fields: policy_no, payment_status, payment_date,
                  payment_method, amount
        
        payment_method_summary
        → Use for payment method used per policy or customer
        → Fields: policy_no, customer_id, payment_method,
                  payment_mode, payment_status
        
        payment_amount_details
        → Use for premium, GST and charge amounts
        → Fields: policy_no, charges1 (GST), payable_gross_premium,
                  payable_net_premium, amount, payment_date
        
        transaction_history
        → Use for full historical transactions
        → Fields: policy_no, customer_id, product, trans_type,
                  payment_date, payment_method, payment_status, amount
        
        failed_payments
        → Use for failed or unsuccessful payments
        → Fields: policy_no, customer_id, payment_date,
                  payment_method, payment_status, response_code,
                  response_desc, amount
        
        payment_method_analytics
        → Use for analytics and counts by payment method
        → Fields: customer_id, product, payment_method,
                  payment_status, payment_date
        
        recent_transactions
        → Use for most recent transactions per customer
        → Fields: policy_no, customer_id, product, trans_type,
                  timestamp_created, policy_from, policy_to,
                  payment_date, payment_status, amount,
                  payment_method
    """)

    # Train with sample question SQL pairs
    vn.train(
        question="What is the payment status of policy DTPS123?",
        sql="SELECT * FROM latest_payment_status WHERE policy_no = 'DTPS123'"
    )

    vn.train(
        question="Show recent transactions for customer 123",
        sql="SELECT * FROM recent_transactions WHERE customer_id = '123' ORDER BY payment_date DESC"
    )

    vn.train(
        question="Show failed payments for policy DTPS123",
        sql="SELECT * FROM failed_payments WHERE policy_no = 'DTPS123'"
    )

    vn.train(
        question="What is the premium amount for policy DTPS123?",
        sql="SELECT * FROM payment_amount_details WHERE policy_no = 'DTPS123'"
    )

    vn.train(
        question="What payment method was used for policy DTPS123?",
        sql="SELECT * FROM payment_method_summary WHERE policy_no = 'DTPS123'"
    )

    vn.train(
        question="Show transaction history for policy DTPS123",
        sql="SELECT * FROM transaction_history WHERE policy_no = 'DTPS123' ORDER BY payment_date DESC"
    )

    print("Vanna setup complete ✅")

def answer_from_db(question: str) -> str:
    """Answer transaction questions using Vanna"""
    try:
        sql = vn.generate_sql(question)
        print(f"Generated SQL: {sql}")
        result = vn.run_sql(sql)
        if result is None or result.empty:
            return "No data found for your query."
        return result.to_string(index=False)
    except Exception as e:
        print(f"Vanna error: {e}")
        return "Sorry, I could not retrieve the transaction data."