import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# INSTACART CUSTOMER PURCHASE BEHAVIOUR ANALYSIS
# ============================================================


# -----------------------------------------------------------
# 1. Import Dataset
# -----------------------------------------------------------

orders         = pd.read_csv("orders.csv")
products       = pd.read_csv("products.csv")
order_products = pd.read_csv("order_products__prior.csv")
departments    = pd.read_csv("departments.csv")

print("First 5 Rows")
print(orders.head())


# -----------------------------------------------------------
# 2. Basic Data
# -----------------------------------------------------------

print("\n----- BASIC DATA -----")

print("\nTotal Customers")
print(orders["user_id"].nunique())

print("\nTotal Columns")
print(len(orders.columns))

print("\nMissing Values")
print(orders.isnull().sum())


# -----------------------------------------------------------
# 3. Analysis
# -----------------------------------------------------------

print("\n----- ANALYSIS -----")

total_orders    = orders["order_id"].nunique()
total_products  = products["product_id"].nunique()
total_customers = orders["user_id"].nunique()
average_order   = total_orders / total_customers

print("\nTotal Orders =", total_orders)
print("Total Products =", total_products)
print("Total Customers =", total_customers)
print("Average Orders Per Customer =", round(average_order, 2))


# -----------------------------------------------------------
# 4. Order Comparison (Low vs High Reorder Orders)
# -----------------------------------------------------------

customer_reorder = order_products.groupby("order_id")["reordered"].mean()
low_reorder  = (customer_reorder < 0.5).sum()
high_reorder = (customer_reorder >= 0.5).sum()

print("\n----- ORDER COMPARISON -----")
print("Low Reorder Orders  :", low_reorder)
print("High Reorder Orders :", high_reorder)

plt.figure(figsize=(7, 5))
plt.bar(["Low Reorder Orders", "High Reorder Orders"],
        [low_reorder, high_reorder], color=["#4C72B0", "#4C72B0"])
plt.title("Low vs High Reorder Orders")
plt.xlabel("Category")
plt.ylabel("Number of Orders")
plt.tight_layout()
plt.show()


# -----------------------------------------------------------
# 5. Product Hotspot (Top 10 Most Ordered Products)
# -----------------------------------------------------------

top_products = order_products.merge(products, on="product_id")
top10 = top_products["product_name"].value_counts().head(10)

print("\n----- TOP 10 PRODUCTS -----")
print(top10)

plt.figure(figsize=(10, 5))
plt.bar(range(len(top10)), top10.values, color="#4C72B0")
plt.xticks(range(len(top10)), top10.index, rotation=45, ha="right")
plt.title("Top 10 Most Ordered Products")
plt.xlabel("Product")
plt.ylabel("Number of Orders")
plt.tight_layout()
plt.show()


# -----------------------------------------------------------
# 6. Purchase Behaviour Analysis
# -----------------------------------------------------------

# New vs Reordered
purchase = order_products["reordered"].value_counts()
labels = ["Reordered" if i == 1 else "New Purchase" for i in purchase.index]

print("\n----- PURCHASE BEHAVIOUR -----")
for label, val in zip(labels, purchase.values):
    print(f"{label:<20} {val}")

plt.figure(figsize=(7, 5))
plt.bar(labels, purchase.values, color=["#4C72B0", "#4C72B0"])
plt.title("Customer Purchase Behaviour")
plt.xlabel("Purchase Type")
plt.ylabel("Number of Purchases")
plt.tight_layout()
plt.show()

# Department Analysis
department_data  = products.merge(departments, on="department_id")
department_count = department_data["department"].value_counts().head(10)

print("\n----- DEPARTMENT ANALYSIS -----")
print(department_count)

plt.figure(figsize=(10, 5))
plt.bar(department_count.index, department_count.values, color="#4C72B0")
plt.title("Popular Product Departments")
plt.xlabel("Department")
plt.ylabel("Number of Products")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# -----------------------------------------------------------
# 7. Trend Chart
# -----------------------------------------------------------

# Orders by Day of Week
day_map = {0:"Sunday", 1:"Monday", 2:"Tuesday", 3:"Wednesday",
           4:"Thursday", 5:"Friday", 6:"Saturday"}
orders["day_name"] = orders["order_dow"].map(day_map)
day_order  = ["Sunday","Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"]
day_counts = orders["day_name"].value_counts().reindex(day_order)

print("\n----- ORDERS BY DAY -----")
print(day_counts)

plt.figure(figsize=(9, 5))
plt.bar(day_counts.index, day_counts.values, color="#4C72B0")
plt.title("Orders by Day of Week")
plt.xlabel("Day")
plt.ylabel("Number of Orders")
plt.tight_layout()
plt.show()

# Orders by Hour of Day
hour_counts = orders["order_hour_of_day"].value_counts().sort_index()

print("\n----- ORDERS BY HOUR -----")
print(hour_counts)

plt.figure(figsize=(10, 5))
plt.plot(hour_counts.index, hour_counts.values, color="#2E75B6", marker="o")
plt.title("Orders by Hour of Day")
plt.xlabel("Hour of Day")
plt.ylabel("Number of Orders")
plt.xticks(range(0, 24))
plt.tight_layout()
plt.show()


# -----------------------------------------------------------
# 8. Order Analysis
# -----------------------------------------------------------

order_pattern = orders["order_number"].value_counts().sort_index()

print("\n----- ORDER PATTERN -----")
print(order_pattern.head())

plt.figure(figsize=(8, 5))
plt.plot(order_pattern.index, order_pattern.values, color="#4C72B0")
plt.title("Customer Order Pattern")
plt.xlabel("Order Number")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.show()


# -----------------------------------------------------------
# 9. Heatmap
# -----------------------------------------------------------

plt.figure(figsize=(7, 5))
sns.heatmap(orders.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()


# -----------------------------------------------------------
# 10. Customer Indicators
# -----------------------------------------------------------

reorder_percentage = order_products["reordered"].mean() * 100

print("\n----- CUSTOMER INDICATORS -----")
print(f"Total Orders         {total_orders}")
print(f"Total Customers      {total_customers}")
print(f"Reorder Percentage   {round(reorder_percentage, 2)}%")