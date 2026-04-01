# DS 4320 Project 1 Pipeline: Credit Card Fraud Detection

## Objective
This notebook loads the relational credit card fraud dataset into DuckDB, prepares a dataset for modeling using SQL, trains a model, and scores how well the model identifies fraud.


```python
import duckdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
```


```python
# load the CSV file
transactions = pd.read_csv("../data/sample_transactions.csv")
customers = pd.read_csv("../data/sample_customers.csv")
merchants = pd.read_csv("../data/sample_merchants.csv")
fraud_labels = pd.read_csv("../data/sample_fraud_labels.csv")

print(transactions.shape)
print(customers.shape)
print(merchants.shape)
print(fraud_labels.shape)
```

    (4992, 33)
    (997, 1)
    (50, 1)
    (4992, 2)


## Data Preparation 
The relational dataset is stored in separate CSV files. These files represent transactions, customers, merchants, and fraud labels.


```python
# load into duckdb
con = duckdb.connect()

con.register("transactions_df", transactions)
con.register("customers_df", customers)
con.register("merchants_df", merchants)
con.register("fraud_labels_df", fraud_labels)

con.execute("CREATE OR REPLACE TABLE transactions AS SELECT * FROM transactions_df")
con.execute("CREATE OR REPLACE TABLE customers AS SELECT * FROM customers_df")
con.execute("CREATE OR REPLACE TABLE merchants AS SELECT * FROM merchants_df")
con.execute("CREATE OR REPLACE TABLE fraud_labels AS SELECT * FROM fraud_labels_df")
```




    <_duckdb.DuckDBPyConnection at 0x14c9be430>




```python
con.execute("SHOW TABLES").fetchdf()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>customers</td>
    </tr>
    <tr>
      <th>1</th>
      <td>customers_df</td>
    </tr>
    <tr>
      <th>2</th>
      <td>fraud_labels</td>
    </tr>
    <tr>
      <th>3</th>
      <td>fraud_labels_df</td>
    </tr>
    <tr>
      <th>4</th>
      <td>merchants</td>
    </tr>
    <tr>
      <th>5</th>
      <td>merchants_df</td>
    </tr>
    <tr>
      <th>6</th>
      <td>transactions</td>
    </tr>
    <tr>
      <th>7</th>
      <td>transactions_df</td>
    </tr>
  </tbody>
</table>
</div>




```python
# query to prepare the solution
model_df = con.execute("""
    SELECT
        t.*,
        f.Class
    FROM transactions t
    JOIN fraud_labels f
        ON t.transaction_id = f.transaction_id
""").fetchdf()

model_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Time</th>
      <th>V1</th>
      <th>V2</th>
      <th>V3</th>
      <th>V4</th>
      <th>V5</th>
      <th>V6</th>
      <th>V7</th>
      <th>V8</th>
      <th>V9</th>
      <th>...</th>
      <th>V24</th>
      <th>V25</th>
      <th>V26</th>
      <th>V27</th>
      <th>V28</th>
      <th>Amount</th>
      <th>transaction_id</th>
      <th>customer_id</th>
      <th>merchant_id</th>
      <th>Class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>161085.0</td>
      <td>1.983253</td>
      <td>-0.176808</td>
      <td>-0.234645</td>
      <td>0.420610</td>
      <td>-0.524881</td>
      <td>-0.517922</td>
      <td>-0.426280</td>
      <td>-0.085440</td>
      <td>1.058515</td>
      <td>...</td>
      <td>1.133433</td>
      <td>-0.414366</td>
      <td>-0.684325</td>
      <td>0.039992</td>
      <td>-0.018518</td>
      <td>0.12</td>
      <td>263729</td>
      <td>729</td>
      <td>29</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>144921.0</td>
      <td>-0.668908</td>
      <td>1.401700</td>
      <td>-0.105966</td>
      <td>-0.697902</td>
      <td>0.250835</td>
      <td>-1.096017</td>
      <td>0.845909</td>
      <td>0.014705</td>
      <td>0.223832</td>
      <td>...</td>
      <td>-0.011753</td>
      <td>-0.381611</td>
      <td>0.124993</td>
      <td>0.179492</td>
      <td>-0.068217</td>
      <td>1.98</td>
      <td>227066</td>
      <td>66</td>
      <td>16</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>130476.0</td>
      <td>-0.631970</td>
      <td>1.749792</td>
      <td>-0.057029</td>
      <td>0.957674</td>
      <td>1.566792</td>
      <td>-0.287978</td>
      <td>1.778291</td>
      <td>-0.349138</td>
      <td>-1.306805</td>
      <td>...</td>
      <td>0.540801</td>
      <td>1.189802</td>
      <td>-0.228100</td>
      <td>-0.049005</td>
      <td>0.082245</td>
      <td>46.03</td>
      <td>194199</td>
      <td>199</td>
      <td>49</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>142929.0</td>
      <td>-2.130823</td>
      <td>0.266709</td>
      <td>0.297901</td>
      <td>-0.666425</td>
      <td>-0.767904</td>
      <td>-0.341857</td>
      <td>-0.258019</td>
      <td>1.031254</td>
      <td>0.299701</td>
      <td>...</td>
      <td>0.756948</td>
      <td>-0.062582</td>
      <td>0.646321</td>
      <td>0.214136</td>
      <td>-0.019755</td>
      <td>116.10</td>
      <td>222324</td>
      <td>324</td>
      <td>24</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>57976.0</td>
      <td>0.947075</td>
      <td>-0.402152</td>
      <td>1.150211</td>
      <td>0.531900</td>
      <td>-0.197880</td>
      <td>2.060615</td>
      <td>-1.035872</td>
      <td>0.797776</td>
      <td>0.600676</td>
      <td>...</td>
      <td>-0.973012</td>
      <td>-0.079638</td>
      <td>0.426819</td>
      <td>0.086370</td>
      <td>0.003452</td>
      <td>11.50</td>
      <td>79342</td>
      <td>342</td>
      <td>42</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 34 columns</p>
</div>



## Query Preparation
The transaction and fraud label tables were joined to create the final modeling dataset. This query reconstructs the outcome variable needed for supervised fraud detection.


```python
# define features and target
X = model_df.drop(columns=["transaction_id", "customer_id", "merchant_id", "Class"])
y = model_df["Class"]

print(X.shape)
print(y.value_counts())
```

    (4992, 30)
    Class
    0    4500
    1     492
    Name: count, dtype: int64



```python
# train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)
```


```python
#implement a model
model = Pipeline([
    ("scaler", StandardScaler()),
    ("logreg", LogisticRegression(max_iter=1000, class_weight="balanced"))
])

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

```


```python
# evaluate results
print(classification_report(y_test, y_pred, digits=4))
```

                  precision    recall  f1-score   support
    
               0     0.9872    0.9741    0.9806      1350
               1     0.7892    0.8851    0.8344       148
    
        accuracy                         0.9653      1498
       macro avg     0.8882    0.9296    0.9075      1498
    weighted avg     0.9677    0.9653    0.9662      1498
    



```python
#confusion matrix
cm = confusion_matrix(y_test, y_pred)
cm
```




    array([[1315,   35],
           [  17,  131]])



## Analysis Rationale
This model has good fraud detection capabilities. The recall score for the fraud class is 0.885, meaning we catch about 88.5% of all fraud. We care about this metric because we don't want to miss any fraud cases.

The precision score is lower at 0.789 for fraud. This means that while we do predict some transactions as fraud that do not turn out to be fraud, most of the transactions we flag are truly fraud. We are usually ok with this tradeoff when it comes to fraud detection. It is better to throw up a flag on a suspicious transaction than to miss the fraud entirely.

We can conclude that we can use transaction-level features to predict whether or not a transaction is high-risk for fraud. We still deal with some class imbalance. 


```python
#visualization
fig, ax = plt.subplots(figsize=(6, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(ax=ax, colorbar=False)
ax.set_title("Confusion Matrix for Credit Card Fraud Detection")
plt.tight_layout()
plt.show()
```


    
![png](pipeline_files/pipeline_14_0.png)
    


## Visualization Rationale
I used a confusion matrix to evaluate model performance because the data is very skewed. The confusion matrix allows you to see how many fraudulent transactions were caught vs how many were missed.

## Final Interpretation
This pipeline addresses the project question by loading transaction data saved in relational tables into DuckDB, using SQL to prepare a dataset fit for modeling, training a classifier, and scoring its ability to detect fraudulent transactions. Fraudulent transactions can be detected using transaction-level features, but class imbalance is still an issue. The model prioritizes recall for fraud detection, which is desirable in financial applications where missing fraudulent transactions is more costly than flagging legitimate ones.
