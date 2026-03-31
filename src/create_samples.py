import pandas as pd

# transactions (sample 5000 rows)
pd.read_csv("data/transactions.csv") \
  .sample(5000) \
  .to_csv("data/sample_transactions.csv", index=False)

# fraud labels (sample 5000 rows)
pd.read_csv("data/fraud_labels.csv") \
  .sample(5000) \
  .to_csv("data/sample_fraud_labels.csv", index=False)

# customers (small already, but sample anyway)
pd.read_csv("data/customers.csv") \
  .sample(500) \
  .to_csv("data/sample_customers.csv", index=False)

# merchants (small already)
pd.read_csv("data/merchants.csv") \
  .to_csv("data/sample_merchants.csv", index=False)

print("Sample files created!")