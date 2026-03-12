import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# 1️⃣ Load dataset
data = pd.read_csv("customer_segmentation_data.csv")

# 2️⃣ Display dataset
print(data.head())
print(data.info())

# 3️⃣ Select features for clustering
X = data[['income', 'spending_score']]

# 4️⃣ Create KMeans model
kmeans = KMeans(n_clusters=5, random_state=42)

# 5️⃣ Fit model and create customer segments
data['customer_segment'] = kmeans.fit_predict(X)

# 6️⃣ Display segmented data
print(data.head())

# 7️⃣ Visualize clusters
plt.scatter(
    data['income'],
    data['spending_score'],
    c=data['customer_segment']
)

plt.xlabel("Income")
plt.ylabel("Spending Score")
plt.title("Customer Segmentation")

plt.show()

# 8️⃣ Save segmented dataset
data.to_excel("segmented_customers.xlsx", index=False)

print("Segmented dataset saved successfully!")
