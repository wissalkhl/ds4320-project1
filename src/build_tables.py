import pandas as pd

df = pd.read_csv("data/creditcard.csv") 

# create IDs
df["transaction_id"] = df.index
df["customer_id"] = df.index % 1000
df["merchant_id"] = df.index % 50

# TABLE 1: transactions 
transactions = df.drop(columns=["Class"])

# TABLE 2: fraud_labels 
fraud_labels = df[["transaction_id", "Class"]]

# TABLE 3: customers 
customers = pd.DataFrame({
    "customer_id": df["customer_id"].unique()
})

# TABLE 4: merchants 
merchants = pd.DataFrame({
    "merchant_id": df["merchant_id"].unique()
})

# save tables
transactions.to_csv("data/transactions.csv", index=False)
fraud_labels.to_csv("data/fraud_labels.csv", index=False)
customers.to_csv("data/customers.csv", index=False)
merchants.to_csv("data/merchants.csv", index=False)

print("Tables created successfully!")