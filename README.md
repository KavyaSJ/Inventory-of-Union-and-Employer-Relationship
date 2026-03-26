📊 Inventory of Employer–Union Relationships using Spark MLlib

📌 Project Overview

This project analyzes collective agreements data from the e-Library Portal to uncover patterns in employer–union relationships across different industrial sectors. Using Apache Spark (PySpark) and MLlib, the project applies unsupervised learning techniques to group agreements based on workforce size, contract duration, and sector characteristics.

🎯 Objectives

- Clean and preprocess real-world employer and union relationship data
- Perform exploratory data analysis (EDA)
- Engineer meaningful features for machine learning
- Apply K-Means clustering to identify agreement patterns
- Evaluate clusters using Silhouette Score
- Visualize clusters using Principal Component Analysis (PCA)

🛠️ Tools & Technologies

Apache Spark (PySpark)
Spark MLlib
Python
Spark SQL
Matplotlib / Databricks Visualization
Git & GitHub

📂 Dataset Description

The dataset includes information about collective agreements such as:

Employer and Union names
Agreement start and expiry dates
Industrial sector type
Number of employees
Labour legislation details

🧹 Data Preprocessing

Cleaned inconsistent employee count formats
Removed leading/trailing spaces and special characters
Handled missing values
Converted date columns to compute contract duration (months)
Created employee size buckets for better interpretability

⚙️ Feature Engineering

Selected key features:
employee_count
contract_length_months
Encoded categorical variables using StringIndexer
Scaled features using StandardScaler
Excluded high-cardinality features (e.g., Agreement Location)

🤖 Machine Learning Approach

Applied K-Means Clustering using Spark MLlib
Determined optimal number of clusters using Silhouette Analysis (k = 8)
Built a complete ML Pipeline for preprocessing and modeling

📊 Results & Insights

Identified distinct clusters of agreements based on:
Workforce size
Contract duration
Sector type
Found that:
Majority of agreements involve small employers (0–50 employees)
Private sector dominates agreement distribution
Some clusters represent outlier agreements with unique characteristics

📉 PCA Visualization

Applied Principal Component Analysis (PCA) to reduce dimensions
Visualized clusters in 2D space for better interpretation
Observed clear separation for some clusters and overlap for similar agreement types

⚠️ Limitations

K-Means assumes spherical clusters
High-cardinality features were excluded
PCA visualization may not fully capture high-dimensional separation
