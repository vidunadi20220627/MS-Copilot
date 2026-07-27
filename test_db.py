from db.connection import test_connection, get_policy_wording_credentials

print("Testing connection...")
test_connection()

print("\nTesting view query...")
result = get_policy_wording_credentials("DTPS26043904")  # use a real policy_no you know exists
print(result)