# Smarter Data Modeling Helps Detect Credit Card Fraud More Effectively

## Hook
Credit card fraud is a growing problem that costs financial institutions and consumers billions of dollars each year. Detecting fraudulent transactions quickly is difficult because they are rare and often hidden among thousands of legitimate purchases.

## Problem Statement
Financial institutions process massive volumes of transactions every day, making it challenging to identify suspicious activity in real time. Fraudulent transactions make up a very small percentage of all transactions, which creates an imbalanced dataset that makes detection difficult. Traditional methods often struggle to catch fraud without generating too many false alarms. This project focuses on determining whether transaction-level data can be used to effectively identify fraudulent credit card activity.

## Solution Description
This project uses a structured data pipeline to transform raw transaction data into a relational format and applies a machine learning model to classify transactions as fraudulent or legitimate. By analyzing transaction patterns such as amount, time, and behavioral features, the model is able to identify high-risk transactions. The results show that most fraudulent transactions can be detected while keeping false alerts at a manageable level, demonstrating that transaction data can support effective fraud detection systems.

## Chart
![Confusion Matrix](../pipeline/pipeline_files/pipeline_14_0.png)

