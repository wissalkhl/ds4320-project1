import pandas as pd

# load full relational tables
transactions = pd.read_csv("data/transactions.csv")
customers = pd.read_csv("data/customers.csv")
merchants = pd.read_csv("data/merchants.csv")
fraud_labels = pd.read_csv("data/fraud_labels.csv")

# jkoin transactions with labels first so sampling preserves both pieces
full_df = transactions.merge(fraud_labels, on="transaction_id", how="inner")

# separate fraud and non-fraud so the sample includes both classes
fraud_df = full_df[full_df["Class"] == 1]
nonfraud_df = full_df[full_df["Class"] == 0]

# Keep all frauds in the sample, plus a subset of non-frauds
sample_nonfraud = nonfraud_df.sample(n=4500, random_state=42)
sample_df = pd.concat([fraud_df, sample_nonfraud], ignore_index=True)


sample_df = sample_df.sample(frac=1, random_state=42).reset_index(drop=True)

# rebuild sample tables consistently
sample_transactions = sample_df.drop(columns=["Class"])
sample_fraud_labels = sample_df[["transaction_id", "Class"]]

sample_customer_ids = sample_transactions["customer_id"].unique()
sample_merchant_ids = sample_transactions["merchant_id"].unique()

sample_customers = customers[customers["customer_id"].isin(sample_customer_ids)]
sample_merchants = merchants[merchants["merchant_id"].isin(sample_merchant_ids)]

# save
sample_transactions.to_csv("data/sample_transactions.csv", index=False)
sample_fraud_labels.to_csv("data/sample_fraud_labels.csv", index=False)
sample_customers.to_csv("data/sample_customers.csv", index=False)
sample_merchants.to_csv("data/sample_merchants.csv", index=False)

print("Consistent sample files created successfully.")
print(sample_fraud_labels["Class"].value_counts())