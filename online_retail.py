import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv(r"C:\Users\prash\OneDrive\Documents\PERSONAL LIBRARY\Programming\GitHub\online_retail\online_retail.csv")
# print(df.head())
# print(df.tail())
# print(df.columns)
# print(df["Invoice"].value_counts().unique())
# print(df["Invoice"].isnull().value_counts())
# print(df.dtypes)
# print(df.info())
# print("Description")
# print(df.describe(include='all'))
# print("Null Values")
# print(df.isnull().sum())
# print("Duplicate Value")
# print(df.duplicated().sum())

# duplicates = df[df.duplicated(keep=False)]
# print("Duplicated Values:",len(duplicates))
# print("Duplicated Values")
# print(duplicates.head(20))

###Identical Duplicate Rows
# print("Exact Duplicate Rows:",df.duplicated().sum())

duplicate_counts = (
    df.groupby(
        ["Invoice", "StockCode", "Description",
         "Quantity", "InvoiceDate", "Price",
         "Customer ID", "Country"],
        dropna=False
    )
    .size()
    .reset_index(name="Count")
    .sort_values("Count", ascending=False)
)

print(duplicate_counts.head(20))

###Missing Description Investigation

missing_description = df[df["Description"].isnull()]

print("Missing Description Records:",
      len(missing_description))
print(missing_description.head(20))

print(missing_description["StockCode"].nunique(),
    "unique StockCodes have missing descriptions")

###Missing Customer ID
missing_customer = df[df["Customer ID"].isnull()]
print("Missing Customer ID Records:",
      len(missing_customer))
print(missing_customer.head(10))
print(missing_customer["Country"].value_counts().head(10))
print(missing_customer["Quantity"].describe())

###Validation of Quality
print("Quantity Statistics:")
print(df["Quantity"].describe())

print("Negative Quantity:")
print((df["Quantity"] < 0).sum())

print("Zero Quantity:")
print((df["Quantity"] == 0).sum())

##Price Validation
print("Price Statistics:")
print(df["Price"].describe())

print("Negative Price:")
print((df["Price"] < 0).sum())

print("Zero Price:")
print((df["Price"] == 0).sum())


df["Invoice"] = df["Invoice"].astype(str)
cancelled = df["Invoice"].str.startswith("C")
print("Cancelled Transactions:", cancelled.sum())
print(df[cancelled].head(10))


###Copy of Dataset
df_clean = df.copy()

print("Clean dataset created successfully!")
print("Original Shape:", df.shape)
print("Clean Dataset Shape:", df_clean.shape)

df_clean.columns = (df_clean.columns.str.strip().str.replace(" ", "_"))
print(df_clean.columns)

df_clean["Customer_ID"]
df_clean["Invoice"] = df_clean["Invoice"].astype(str)
print(df_clean["Invoice"].dtype)

df_clean["StockCode"] = df_clean["StockCode"].astype(str)
print(df_clean["StockCode"].dtype)

###InvoiceDate
df_clean["InvoiceDate"] = pd.to_datetime(
    df_clean["InvoiceDate"],
    errors="coerce")
print(df_clean["InvoiceDate"].dtype)

print("Invalid InvoiceDate:",
    df_clean["InvoiceDate"].isnull().sum())

##Missing Description
description_mapping = (df_clean.dropna(subset=["Description"])
    .drop_duplicates("StockCode")
    .set_index("StockCode")["Description"])

df_clean["Description"] = (df_clean["Description"].fillna(df_clean["StockCode"].map(description_mapping)))
print("Remaining Missing Descriptions:",df_clean["Description"].isnull().sum())

##Remove Duplicate Values
before_duplicates = len(df_clean)
df_clean = df_clean.drop_duplicates()
after_duplicates = len(df_clean)
removed_duplicates = before_duplicates - after_duplicates
print("Rows Before Removing Duplicates:", before_duplicates)
print("Rows After Removing Duplicates:", after_duplicates)
print("Duplicate Rows Removed:", removed_duplicates)
print("Remaining Duplicate Rows:",df_clean.duplicated().sum())


###Cancelled Transactions
cancelled_mask = df_clean["Invoice"].str.startswith("C")

print("Cancelled Transactions:",cancelled_mask.sum())
df_clean["Is_Cancelled"] = cancelled_mask
print(df_clean["Is_Cancelled"].value_counts())

##Revenue
df_clean["Revenue"] = (df_clean["Quantity"] *df_clean["Price"])

print(df_clean[["Quantity","Price","Revenue"]].head())

df_clean["Year"] = df_clean["InvoiceDate"].dt.year
df_clean["Month"] = df_clean["InvoiceDate"].dt.month
df_clean["Month_Name"] = df_clean["InvoiceDate"].dt.month_name()
df_clean["Day"] = df_clean["InvoiceDate"].dt.day
df_clean["Day_of_Week"] = (df_clean["InvoiceDate"].dt.day_name())

df_clean["Hour"] = df_clean["InvoiceDate"].dt.hour
df_clean["Date"] = df_clean["InvoiceDate"].dt.date

print(df_clean[["InvoiceDate","Year","Month","Month_Name","Day","Day_of_Week","Hour","Date"]].head())
df_customer = df_clean[df_clean["Customer_ID"].notna()].copy()
print("Customer-level Dataset:", df_customer.shape)

###Transaction Type
df_clean["Transaction_Type"] = np.where(df_clean["Quantity"] < 0,"Return","Sale")

print(df_clean["Transaction_Type"].value_counts())

df_sales = df_clean[df_clean["Quantity"] > 0].copy()
print("Sales Dataset Shape:", df_sales.shape)

print("Zero Price Records:",(df_clean["Price"] == 0).sum())
print("Negative Price Records:",(df_clean["Price"] < 0).sum())

df_sales = df_sales[df_sales["Price"] > 0].copy()

print("Sales Dataset Shape:", df_sales.shape)
print("Minimum Price:", df_sales["Price"].min())

###Cleaning Check
print("Final Clean Dataset Shape:")
print(df_clean.shape)

print(" Values:")
print(df_clean.isnull().sum())

print("Duplicate Rows:")
print(df_clean.duplicated().sum())

print("Data Types:")
print(df_clean.dtypes)


print("df_clean:", df_clean.shape)
print("df_sales:", df_sales.shape)

print("Transaction Type:")
print(df_clean["Transaction_Type"].value_counts())

print("Cancelled:")
print(df_clean["Is_Cancelled"].value_counts())

print("Missing Values:")
print(df_clean.isnull().sum())

###Transaction Revenue

df_clean["Revenue"] = (df_clean["Quantity"] * df_clean["Price"])

print(df_clean[["Quantity","Price","Revenue"]].head())
df_clean["Transaction_Type"] = np.where(
    df_clean["Quantity"] < 0,"Return","Sale")

print(df_clean["Transaction_Type"].value_counts())
df_clean["Is_Cancelled"] = (df_clean["Invoice"].str.startswith("C"))

df_clean["Cancellation_Status"] = np.where(df_clean["Is_Cancelled"],"Cancelled","Completed")
print(df_clean["Cancellation_Status"].value_counts())

df_clean["Year"] = (df_clean["InvoiceDate"].dt.year)
print(df_clean["Year"].value_counts().sort_index())

df_clean["Month"] = (df_clean["InvoiceDate"].dt.month)
df_clean["Year_Month"] = (df_clean["InvoiceDate"].dt.to_period("M"))

print(df_clean["Year_Month"].value_counts().sort_index())

df_clean["Month_Name"] = (df_clean["InvoiceDate"].dt.month_name())

df_clean["Day_of_Week"] = (df_clean["InvoiceDate"].dt.day_name())

df_clean["Day"] = (df_clean["InvoiceDate"].dt.day)
df_clean["Hour"] = (df_clean["InvoiceDate"].dt.hour)
df_clean["Is_Weekend"] = (df_clean["InvoiceDate"].dt.dayofweek >= 5)
df_clean["Day_Type"] = np.where(df_clean["Is_Weekend"],"Weekend","Weekday")
print(df_clean["Day_Type"].value_counts())

df_customer = df_clean[df_clean["Customer_ID"].notna()].copy()
print("Customer Dataset Shape:", df_customer.shape)

customer_revenue = (df_customer.groupby("Customer_ID")["Revenue"].sum().reset_index())

customer_revenue.rename(columns={"Revenue": "Total_Customer_Revenue"},inplace=True)
customer_orders = (df_customer.groupby("Customer_ID")["Invoice"].nunique().reset_index())

customer_orders.rename(columns={"Invoice": "Total_Orders"},inplace=True)
print(customer_orders.head())


###Total Quantity Purchased
customer_quantity = (df_customer.groupby("Customer_ID")["Quantity"].sum().reset_index())
customer_quantity.rename(columns={"Quantity": "Total_Quantity"},inplace=True)

###Customer average order values
customer_summary = (customer_revenue.merge(customer_orders,on="Customer_ID",how="left").merge(customer_quantity,on="Customer_ID",how="left"))

customer_summary["Average_Order_Value"] = (customer_summary["Total_Customer_Revenue"] /customer_summary["Total_Orders"])
print(customer_summary.head())

product_summary = (df_clean.groupby("StockCode").agg(Total_Quantity=("Quantity", "sum"),Total_Revenue=("Revenue", "sum"),
    Number_of_Transactions=("Invoice", "nunique")).reset_index())
print(product_summary.head())

country_summary = (df_clean.groupby("Country")
    .agg(Total_Revenue=("Revenue", "sum"),
        Total_Quantity=("Quantity", "sum"),
        Number_of_Orders=("Invoice", "nunique")).reset_index())

print(country_summary.head())

df_sales = df_clean[(df_clean["Quantity"] > 0) & (df_clean["Price"] > 0)].copy()

print("Sales Dataset Shape:", df_sales.shape)

print(df_clean.columns.tolist())

print("Dataset Shape:")
print(df_clean.shape)

print("First 5 Rows:")
print(df_clean.head())

print("Data Types:")
print(df_clean.dtypes)

print("Missing Values:")
print(df_clean.isnull().sum())

###Feature Engineering
df_clean["Revenue"] = (df_clean["Quantity"] * df_clean["Price"])
print(df_clean[["Quantity","Price","Revenue"]].head())

###Transaction Type

df_clean["Transaction_Type"] = np.where(df_clean["Quantity"] < 0,"Return","Sale")
print(df_clean["Transaction_Type"].value_counts())

###Cancellation Status

df_clean["Is_Cancelled"] = (df_clean["Invoice"].str.startswith("C"))

df_clean["Cancellation_Status"] = np.where(
    df_clean["Is_Cancelled"],
    "Cancelled",
    "Completed")

print(df_clean["Cancellation_Status"].value_counts())

###Year

df_clean["Year"] = (df_clean["InvoiceDate"].dt.year)

print(df_clean["Year"].value_counts().sort_index())

###Month

df_clean["Month"] = (df_clean["InvoiceDate"].dt.month)
print(df_clean["Month"].value_counts().sort_index())

# 4.6 Year-Month

df_clean["Year_Month"] = (df_clean["InvoiceDate"].dt.to_period("M"))
print(df_clean["Year_Month"].value_counts().sort_index())


###Month Name
df_clean["Month_Name"] = (df_clean["InvoiceDate"].dt.month_name())

###Day
df_clean["Day"] = (df_clean["InvoiceDate"].dt.day)

###Day of Week

df_clean["Day_of_Week"] = (df_clean["InvoiceDate"].dt.day_name())
print(df_clean["Day_of_Week"].value_counts())

###Hour
df_clean["Hour"] = (df_clean["InvoiceDate"].dt.hour)
print(df_clean["Hour"].value_counts().sort_index())

df_clean["Is_Weekend"] = (df_clean["InvoiceDate"].dt.dayofweek >= 5)

df_clean["Day_Type"] = np.where(df_clean["Is_Weekend"],"Weekend","Weekday")
print(df_clean["Day_Type"].value_counts())

df_customer = df_clean[df_clean["Customer_ID"].notna()].copy()

print("Customer Dataset Shape:", df_customer.shape)

customer_summary = (df_customer.groupby("Customer_ID").agg(Total_Revenue=("Revenue", "sum"),
    Total_Quantity=("Quantity", "sum"),Total_Orders=("Invoice", "nunique")).reset_index())
print(customer_summary.head())

customer_summary["Average_Order_Value"] = (customer_summary["Total_Revenue"] /customer_summary["Total_Orders"])
print(customer_summary.head())

###Product Summary
product_summary = (df_clean.groupby("StockCode")
    .agg(Total_Quantity=("Quantity", "sum"),
    Total_Revenue=("Revenue", "sum"),
    Number_of_Orders=("Invoice", "nunique")).reset_index())
print(product_summary.head())

###Country Summary
country_summary = (df_clean.groupby("Country").agg(
    Total_Revenue=("Revenue", "sum"),
    Total_Quantity=("Quantity", "sum"),
    Number_of_Orders=("Invoice", "nunique")).reset_index())
print(country_summary.head())

###Sales Data

df_sales = df_clean[
    (df_clean["Quantity"] > 0) &
    (df_clean["Price"] > 0)].copy()
print("Sales Dataset Shape:", df_sales.shape)
print("Columns in Clean Dataset:")
print(df_clean.columns.tolist())

#### UNIVARIATE ANALYSIS

###Numerical Summary
numerical_columns = ["Quantity","Price","Revenue","Hour"]
print(df_sales[numerical_columns].describe())

###Quantity Distribution
print("Quantity Statistics:")
print(df_sales["Quantity"].describe())

###HISTOGRAM
plt.figure(figsize=(10, 5))
plt.hist(df_sales["Quantity"],bins=50)

plt.title("Distribution of Quantity")
plt.xlabel("Quantity")
plt.ylabel("Frequency")
plt.show()

###Quantity Boxplot
plt.figure(figsize=(10, 4))
plt.boxplot(df_sales["Quantity"])
plt.title("Boxplot of Quantity")
plt.ylabel("Quantity")
plt.show()

####Price Distribution
print("Price Statistics:")
print(df_sales["Price"].describe())

plt.figure(figsize=(10, 5))

plt.hist(df_sales["Price"],bins=50)

plt.title("Distribution of Unit Price")
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.show()

###Price Boxplot
plt.figure(figsize=(10, 4))
plt.boxplot(df_sales["Price"])
plt.title("Boxplot of Unit Price")
plt.ylabel("Price")
plt.show()

###Revenue Distribution
print("Revenue Statistics:")
print(df_sales["Revenue"].describe())

###Histogram
plt.figure(figsize=(10, 5))
plt.hist(df_sales["Revenue"], bins=50)

plt.title("Distribution of Revenue")
plt.xlabel("Revenue")
plt.ylabel("Frequency")
plt.show()

###Revenue Boxplot
plt.figure(figsize=(10, 4))
plt.boxplot(df_sales["Revenue"])
plt.title("Boxplot of Revenue")
plt.ylabel("Revenue")
plt.show()

###Hourly Transaction Distribution
hour_counts = (df_sales["Hour"].value_counts().sort_index())
print(hour_counts)

plt.figure(figsize=(10, 5))

plt.bar(hour_counts.index,hour_counts.values)

plt.title("Transactions by Hour")
plt.xlabel("Hour of Day")
plt.ylabel("Number of Transactions")
plt.xticks(hour_counts.index)
plt.show()

###Country Distribution
country_counts = (df_sales["Country"].value_counts())
print(country_counts)

top_countries = country_counts.head(15)

plt.figure(figsize=(12, 6))

plt.bar(top_countries.index,top_countries.values)

plt.title("Top 15 Countries by Number of Transactions")
plt.xlabel("Country")
plt.ylabel("Number of Transactions")
plt.xticks(rotation=45, ha="right")
plt.show()

###Transaction Type
transaction_counts = (df_clean["Transaction_Type"].value_counts())
print(transaction_counts)

plt.figure(figsize=(7, 5))

plt.bar(transaction_counts.index,transaction_counts.values)

plt.title("Transaction Type Distribution")
plt.xlabel("Transaction Type")
plt.ylabel("Number of Transactions")

plt.show()

###Cancellation Status
cancellation_counts = (df_clean["Cancellation_Status"].value_counts())
print(cancellation_counts)

plt.figure(figsize=(7, 5))

plt.bar(cancellation_counts.index,cancellation_counts.values)
plt.title("Cancellation Status Distribution")
plt.xlabel("Status")
plt.ylabel("Number of Transactions")
plt.show()

###Day Type
day_type_counts = (df_sales["Day_Type"].value_counts())
print(day_type_counts)

plt.figure(figsize=(7, 5))

plt.bar(day_type_counts.index,day_type_counts.values)

plt.title("Weekday vs Weekend Transactions")
plt.xlabel("Day Type")
plt.ylabel("Number of Transactions")
plt.show()

day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
day_counts = (df_sales["Day_of_Week"].value_counts().reindex(day_order))
print(day_counts)

plt.figure(figsize=(10, 5))
plt.bar(day_counts.index,day_counts.values)

plt.title("Transactions by Day of Week")
plt.xlabel("Day of Week")
plt.ylabel("Number of Transactions")
plt.xticks(rotation=30)
plt.show()

month_order = ["January","February","March","April","May","June","July","August","September","October","November",
"December"]

month_counts = (df_sales["Month_Name"].value_counts().reindex(month_order))
print(month_counts)

plt.figure(figsize=(12, 5))
plt.bar(month_counts.index,month_counts.values)

plt.title("Transactions by Month")
plt.xlabel("Month")
plt.ylabel("Number of Transactions")

plt.xticks(rotation=45)
plt.show()

###Skewness
skewness = df_sales[["Quantity", "Price", "Revenue"]].skew()
print("Skewness:")
print(skewness)

Q1 = df_sales["Revenue"].quantile(0.25)
Q3 = df_sales["Revenue"].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

revenue_outliers = df_sales[(df_sales["Revenue"] < lower_bound)|(df_sales["Revenue"] > upper_bound)]
print("Potential Revenue Outliers:",len(revenue_outliers))
outlier_percentage = (len(revenue_outliers) /len(df_sales)) * 100

print("Revenue Outlier Percentage:",outlier_percentage)

###Summary of Univariate
univariate_summary = pd.DataFrame({
    "Variable": [
        "Quantity",
        "Price",
        "Revenue"],
    "Mean": [
        df_sales["Quantity"].mean(),
        df_sales["Price"].mean(),
        df_sales["Revenue"].mean()],
    "Median": [
        df_sales["Quantity"].median(),
        df_sales["Price"].median(),
        df_sales["Revenue"].median()],
    "Std_Dev": [
        df_sales["Quantity"].std(),
        df_sales["Price"].std(),
        df_sales["Revenue"].std()],
    "Min": [
        df_sales["Quantity"].min(),
        df_sales["Price"].min(),
        df_sales["Revenue"].min()],
    "Max": [
        df_sales["Quantity"].max(),
        df_sales["Price"].max(),
        df_sales["Revenue"].max()],
    "Skewness": [
        df_sales["Quantity"].skew(),
        df_sales["Price"].skew(),
        df_sales["Revenue"].skew()]})

print(univariate_summary)

print(df_sales[[
    "Quantity",
    "Price",
    "Revenue"
]].describe())

print("\nSkewness:")
print(df_sales[[
    "Quantity",
    "Price",
    "Revenue"
]].skew())

print("Transaction Type:")
print(df_clean["Transaction_Type"].value_counts())

print("Cancellation Status:")
print(df_clean["Cancellation_Status"].value_counts())

print("Top 10 Countries:")
print(df_sales["Country"].value_counts().head(10))

print("Day of Week:")
print(day_counts)

print("Month:")
print(month_counts)

###BIVARIATE ANALYSIS

##Quantity vs Revenue
quantity_revenue_corr = df_sales[["Quantity", "Revenue"]].corr()
print(quantity_revenue_corr)

##Scatter Plot
plt.figure(figsize=(10, 6))
plt.scatter(df_sales["Quantity"],df_sales["Revenue"],alpha=0.3)
plt.title("Quantity vs Revenue")
plt.xlabel("Quantity")
plt.ylabel("Revenue")
plt.show()


##Price and Quantity
price_quantity_corr = df_sales[["Price", "Quantity"]].corr()
print(price_quantity_corr)

plt.figure(figsize=(10, 6))

plt.scatter(df_sales["Price"],df_sales["Quantity"],alpha=0.3)
plt.title("Price vs Quantity")
plt.xlabel("Unit Price")
plt.ylabel("Quantity")
plt.show()

##Price and Revenue
price_revenue_corr = df_sales[["Price", "Revenue"]].corr()
print(price_revenue_corr)

plt.figure(figsize=(10, 6))
plt.scatter(df_sales["Price"],df_sales["Revenue"],alpha=0.3)
plt.title("Price vs Revenue")
plt.xlabel("Unit Price")
plt.ylabel("Revenue")
plt.show()

##Correlation Matrix
correlation_matrix = df_sales[["Quantity", "Price", "Revenue"]].corr()
print(correlation_matrix)

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix,annot=True,fmt=".2f")

plt.title("Correlation Matrix")
plt.show()

##Country vs Revenue
country_revenue = (df_sales.groupby("Country")["Revenue"].sum().sort_values(ascending=False))
print(country_revenue.head(15))

top_country_revenue = country_revenue.head(15)

plt.figure(figsize=(12, 6))

plt.bar(top_country_revenue.index,top_country_revenue.values)

plt.title("Top 15 Countries by Revenue")
plt.xlabel("Country")
plt.ylabel("Total Revenue")

plt.xticks(rotation=45, ha="right")
plt.show()

##Country vs order
country_orders = (df_sales.groupby("Country")["Invoice"].nunique().sort_values(ascending=False))
print(country_orders.head(15))
top_country_orders = country_orders.head(15)
plt.figure(figsize=(12, 6))

plt.bar(top_country_orders.index,top_country_orders.values)
plt.title("Top 15 Countries by Number of Orders")
plt.xlabel("Country")
plt.ylabel("Number of Orders")
plt.xticks(rotation=45, ha="right")
plt.show()

###Day of week vs revenue
day_revenue = (df_sales.groupby("Day_of_Week")["Revenue"].sum().reindex(day_order))
print(day_revenue)

plt.figure(figsize=(10, 5))

plt.bar(day_revenue.index,day_revenue.values)

plt.title("Revenue by Day of Week")
plt.xlabel("Day of Week")
plt.ylabel("Total Revenue")
plt.xticks(rotation=30)
plt.show()

##weekdays vs weekend revenue
day_type_revenue = (df_sales.groupby("Day_Type")["Revenue"].sum().sort_values(ascending=False))
print(day_type_revenue)

plt.figure(figsize=(7, 5))

plt.bar(day_type_revenue.index,day_type_revenue.values)

plt.title("Revenue: Weekday vs Weekend")
plt.xlabel("Day Type")
plt.ylabel("Total Revenue")
plt.show()

##Hour vs revenue
hour_revenue = (df_sales.groupby("Hour")["Revenue"].sum().sort_index())
print(hour_revenue)
plt.figure(figsize=(10, 5))

plt.bar(hour_revenue.index,hour_revenue.values)

plt.title("Revenue by Hour")
plt.xlabel("Hour of Day")
plt.ylabel("Total Revenue")
plt.xticks(hour_revenue.index)
plt.show()

##months vs revenue
month_revenue = (df_sales.groupby("Month_Name")["Revenue"].sum().reindex(month_order))
print(month_revenue)

plt.figure(figsize=(12, 5))
plt.bar(month_revenue.index,month_revenue.values)

plt.title("Revenue by Month")
plt.xlabel("Month")
plt.ylabel("Total Revenue")
plt.xticks(rotation=45)
plt.show()

monthly_revenue = (df_sales.groupby("Year_Month")["Revenue"].sum())
print(monthly_revenue)

plt.figure(figsize=(12, 5))

plt.plot(monthly_revenue.index.astype(str),monthly_revenue.values,marker="o")
plt.title("Monthly Revenue Trend")
plt.xlabel("Year-Month")
plt.ylabel("Total Revenue")
plt.xticks(rotation=45)
plt.show()

##Transaction Type vs Revenue
transaction_revenue = (df_clean.groupby("Transaction_Type")["Revenue"].agg(["sum", "mean", "count"]))
print(transaction_revenue)

plt.figure(figsize=(7, 5))
plt.bar(transaction_revenue.index,transaction_revenue["sum"])
plt.title("Revenue by Transaction Type")
plt.xlabel("Transaction Type")
plt.ylabel("Total Revenue")

plt.show()

##Cancellation Status vs Revenue
cancellation_revenue = (df_clean.groupby("Cancellation_Status")["Revenue"].agg(["sum", "mean", "count"]))
print(cancellation_revenue)

###Product vs Revenue
product_revenue = (df_sales.groupby(["StockCode", "Description"])["Revenue"].sum().sort_values(ascending=False))
print(product_revenue.head(15))

top_products = product_revenue.head(10)
plt.figure(figsize=(12, 6))
plt.bar(top_products.index.get_level_values("StockCode"),top_products.values)

plt.title("Top 10 Products by Revenue")
plt.xlabel("Stock Code")
plt.ylabel("Total Revenue")
plt.xticks(rotation=45)
plt.show()

###Product vs Quantity
product_quantity = (df_sales.groupby(["StockCode", "Description"])["Quantity"].sum().sort_values(ascending=False))
print(product_quantity.head(15))

top_quantity_products = product_quantity.head(10)

plt.figure(figsize=(12, 6))

plt.bar(top_quantity_products.index.get_level_values("StockCode"),top_quantity_products.values)
plt.title("Top 10 Products by Quantity Sold")
plt.xlabel("Stock Code")
plt.ylabel("Total Quantity Sold")

plt.xticks(rotation=45)
plt.show()

###Country vs Average Order Value
country_aov = (df_sales.groupby("Country").agg(Total_Revenue=("Revenue", "sum"),Number_of_Orders=("Invoice", "nunique")))
country_aov["Average_Order_Value"] = (country_aov["Total_Revenue"] /country_aov["Number_of_Orders"])

country_aov = country_aov.sort_values("Average_Order_Value",ascending=False)
print(country_aov.head(15))

###Bivariate Summary
print("Quantity vs Revenue Correlation:")
print(df_sales[["Quantity", "Revenue"]].corr().iloc[0, 1])

print("\nPrice vs Quantity Correlation:")
print(df_sales[["Price", "Quantity"]].corr().iloc[0, 1])

print("\nPrice vs Revenue Correlation:")
print(df_sales[["Price", "Revenue"]].corr().iloc[0, 1])

print("Top Countries by Revenue:")
print(country_revenue.head(10))

print("Top Products by Revenue:")
print(product_revenue.head(10))

print("Revenue by Day:")
print(day_revenue)

print("Revenue by Hour:")
print(hour_revenue)

print("Revenue by Month:")
print(month_revenue)

###Multivariate Analysis
print("Sales Dataset Shape:")
print(df_sales.shape)
print("Columns:")
print(df_sales.columns.tolist())

###Quantity + Price + Revenue
sales_numeric = df_sales[["Quantity", "Price", "Revenue"]]

print(sales_numeric.describe())
print("Correlation Matrix:")
print(sales_numeric.corr())

plt.figure(figsize=(8, 6))
sns.heatmap(sales_numeric.corr(),annot=True,fmt=".2f")
plt.title("Quantity, Price and Revenue Correlation")
plt.show()

###Country Performance
country_performance = (df_sales.groupby("Country").agg(Total_Revenue=("Revenue", "sum"),
    Total_Quantity=("Quantity", "sum"),
    Number_of_Orders=("Invoice", "nunique"),
    Average_Price=("Price", "mean"))
    .reset_index())
print(country_performance.head())

country_performance["Average_Order_Value"] = (country_performance["Total_Revenue"] / country_performance["Number_of_Orders"])

country_performance["Average_Order_Value"] = (
    country_performance["Total_Revenue"] /
    country_performance["Number_of_Orders"])

country_performance = (country_performance.sort_values("Total_Revenue",ascending=False))
print(country_performance.head(15))

top_countries = country_performance.head(10)

plt.figure(figsize=(12, 6))

plt.scatter(top_countries["Number_of_Orders"],top_countries["Total_Revenue"],s=100)

for _, row in top_countries.iterrows():
    plt.annotate(row["Country"],(row["Number_of_Orders"],row["Total_Revenue"]))

plt.title("Country Performance: Orders vs Revenue")
plt.xlabel("Number of Orders")
plt.ylabel("Total Revenue")
plt.show()

top_countries = country_performance.head(10)

plt.figure(figsize=(12, 6))

plt.scatter(top_countries["Number_of_Orders"],top_countries["Total_Revenue"],s=100)

for _, row in top_countries.iterrows():plt.annotate(row["Country"],(row["Number_of_Orders"],row["Total_Revenue"]))

plt.title("Country Performance: Orders vs Revenue")
plt.xlabel("Number of Orders")
plt.ylabel("Total Revenue")
plt.show()

plt.figure(figsize=(10, 6))

plt.scatter(country_performance["Total_Quantity"],country_performance["Total_Revenue"],s=80)

plt.title("Country-Level Quantity vs Revenue")

plt.xlabel("Total Quantity Sold")
plt.ylabel("Total Revenue")
plt.show()

###Monthly Performance
monthly_performance = (
    df_sales
    .groupby("Year_Month")
    .agg(Total_Revenue=("Revenue", "sum"),
        Total_Quantity=("Quantity", "sum"),
        Number_of_Orders=("Invoice", "nunique")).reset_index())
monthly_performance["Average_Order_Value"] = (monthly_performance["Total_Revenue"] /monthly_performance["Number_of_Orders"])
print(monthly_performance)

fig, ax1 = plt.subplots(figsize=(12, 6))

ax1.plot(monthly_performance["Year_Month"].astype(str),
    monthly_performance["Total_Revenue"],
    marker="o")
ax1.set_xlabel("Year-Month")
ax1.set_ylabel("Total Revenue")
plt.title("Monthly Revenue and Order Performance")
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(12, 5))

plt.plot(monthly_performance["Year_Month"].astype(str),
    monthly_performance["Average_Order_Value"],
    marker="o")

plt.title("Monthly Average Order Value")
plt.xlabel("Year-Month")
plt.ylabel("Average Order Value")

plt.xticks(rotation=45)
plt.show()

###Day + Hour Analysis
day_hour_revenue = pd.pivot_table(
    df_sales,
    values="Revenue",
    index="Day_of_Week",
    columns="Hour",
    aggfunc="sum",
    fill_value=0)

day_hour_revenue = (day_hour_revenue.reindex(day_order))
print(day_hour_revenue)

plt.figure(figsize=(14, 6))

sns.heatmap(day_hour_revenue,cmap="YlGnBu")

plt.title("Revenue by Day of Week and Hour")

plt.xlabel("Hour")
plt.ylabel("Day of Week")

plt.show()

###Country + Month + Revenue
country_month_revenue = pd.pivot_table(
    df_sales,
    values="Revenue",
    index="Country",
    columns="Year_Month",
    aggfunc="sum",
    fill_value=0)
print(country_month_revenue.head(10))

top_10_country_names = (country_performance.head(10)["Country"])
top_country_month = (country_month_revenue.loc[top_10_country_names])
plt.figure(figsize=(14, 7))
sns.heatmap(top_country_month)
plt.title("Revenue by Country and Month")
plt.xlabel("Year-Month")
plt.ylabel("Country")
plt.xticks(rotation=45)
plt.show()
###customer Summary
print(customer_summary.describe())

top_customers = (customer_summary.sort_values("Total_Revenue",ascending=False).head(20))
print(top_customers)

plt.figure(figsize=(10, 6))
plt.scatter(customer_summary["Total_Orders"],customer_summary["Total_Revenue"],alpha=0.5)
plt.title("Customer Orders vs Total Revenue")

plt.xlabel("Total Orders")
plt.ylabel("Total Revenue")
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(customer_summary["Total_Quantity"],
    customer_summary["Total_Revenue"],
    s=customer_summary["Average_Order_Value"],
    alpha=0.5)

plt.title("Customer Quantity vs Revenue")
plt.xlabel("Total Quantity Purchased")
plt.ylabel("Total Revenue")
plt.show()

###Product performance
product_performance = (df_sales.groupby(["StockCode", "Description"]).agg(Total_Revenue=("Revenue", "sum"),
    Total_Quantity=("Quantity", "sum"),
    Number_of_Orders=("Invoice", "nunique"),
    Average_Price=("Price", "mean")).reset_index())

product_performance["Average_Order_Value"] = (product_performance["Total_Revenue"] / product_performance["Number_of_Orders"])
print(product_performance.head())
print(product_performance.shape)
print(product_performance.isnull().sum())

###top product by revenue
top_products_revenue = (product_performance.sort_values("Total_Revenue",ascending=False).head(10))
print(top_products_revenue[["StockCode","Description","Total_Revenue","Total_Quantity","Number_of_Orders","Average_Price"]])
plt.figure(figsize=(12, 6))

plt.bar(top_products_revenue["StockCode"],top_products_revenue["Total_Revenue"])

plt.title("Top 10 Products by Revenue")
plt.xlabel("Stock Code")
plt.ylabel("Total Revenue")
plt.xticks(rotation=45)
plt.show()

###top product by quantity
top_products_quantity = (product_performance.sort_values("Total_Quantity",ascending=False).head(10))
print(top_products_quantity[["StockCode","Description","Total_Quantity","Total_Revenue","Number_of_Orders","Average_Price"]])
###top product by revenue
plt.figure(figsize=(12, 6))

plt.bar(top_products_quantity["StockCode"],top_products_quantity["Total_Quantity"])
plt.title("Top 10 Products by Quantity Sold")
plt.xlabel("Stock Code")
plt.ylabel("Total Quantity Sold")
plt.xticks(rotation=45)
plt.show()

###Pq vs revenue
plt.figure(figsize=(10, 6))

plt.scatter(product_performance["Total_Quantity"],product_performance["Total_Revenue"],alpha=0.5)
plt.title("Product Quantity vs Revenue")
plt.xlabel("Total Quantity Sold")
plt.ylabel("Total Revenue")
plt.show()

top_products = (product_performance.sort_values("Total_Revenue",ascending=False).head(50))

plt.figure(figsize=(10, 6))
plt.scatter(top_products["Total_Quantity"],top_products["Total_Revenue"],s=top_products["Average_Price"] * 10,alpha=0.5)

plt.title("Product Quantity vs Revenue ""(Bubble Size = Average Price)")

plt.xlabel("Total Quantity Sold")
plt.ylabel("Total Revenue")
plt.show()

quantity_median = (product_performance["Total_Quantity"].median())
revenue_median = (product_performance["Total_Revenue"].median())

print("Quantity Median:", quantity_median)
print("Revenue Median:", revenue_median)

def classify_product(row):
    if (row["Total_Quantity"] >= quantity_median
        and
        row["Total_Revenue"] >= revenue_median):
        return "High Volume - High Revenue"

    elif (row["Total_Quantity"] >= quantity_median
        and
        row["Total_Revenue"] < revenue_median):
        return "High Volume - Low Revenue"

    elif (row["Total_Quantity"] < quantity_median
        and
        row["Total_Revenue"] >= revenue_median):
        return "Low Volume - High Revenue"
    else:
        return "Low Volume - Low Revenue"

product_performance["Product_Segment"] = (
    product_performance.apply(classify_product,axis=1))

print(product_performance["Product_Segment"].value_counts())
segment_counts = (product_performance["Product_Segment"].value_counts())

plt.figure(figsize=(10, 6))
plt.bar(segment_counts.index,segment_counts.values)

plt.title("Product Performance Segments")
plt.xlabel("Product Segment")
plt.ylabel("Number of Products")

plt.xticks(rotation=30, ha="right")
plt.show()

###product_performance

product_performance = (
    df_sales
    .groupby(["StockCode", "Description"])
    .agg(Total_Revenue=("Revenue", "sum"),
        Total_Quantity=("Quantity", "sum"),
        Number_of_Orders=("Invoice", "nunique"),
        Average_Price=("Price", "mean")).reset_index())

product_performance["Average_Order_Value"] = (product_performance["Total_Revenue"] / product_performance["Number_of_Orders"])
print(product_performance.head())
print(product_performance.shape)
print(product_performance.isnull().sum())


###top product by revenue

top_products_revenue = (product_performance.sort_values("Total_Revenue",ascending=False).head(10))

print(
    top_products_revenue[
        ["StockCode",
        "Description",
        "Total_Revenue",
        "Total_Quantity",
        "Number_of_Orders",
        "Average_Price"]])

plt.figure(figsize=(12, 6))

plt.bar(
    top_products_revenue["StockCode"],
    top_products_revenue["Total_Revenue"]
)

plt.title("Top 10 Products by Revenue")
plt.xlabel("Stock Code")
plt.ylabel("Total Revenue")
plt.xticks(rotation=45)
plt.show()

### Top product by quantity

top_products_quantity = (
    product_performance
    .sort_values(
        "Total_Quantity",
        ascending=False
    )
    .head(10)
)

print(
    top_products_quantity[
        ["StockCode",
        "Description",
        "Total_Quantity",
        "Total_Revenue",
        "Number_of_Orders",
        "Average_Price"]])

plt.figure(figsize=(12, 6))
plt.bar(top_products_quantity["StockCode"],top_products_quantity["Total_Quantity"])
plt.title("Top 10 Products by Quantity Sold")
plt.xlabel("Stock Code")
plt.ylabel("Total Quantity Sold")

plt.xticks(rotation=45)
plt.show()

###product quantity by revenue

plt.figure(figsize=(10, 6))
plt.scatter(product_performance["Total_Quantity"],product_performance["Total_Revenue"],alpha=0.5)
plt.title("Product Quantity vs Revenue")
plt.xlabel("Total Quantity Sold")
plt.ylabel("Total Revenue")
plt.show()


### product quantity vs revenue vs price

plt.figure(figsize=(10, 6))

plt.scatter(product_performance["Total_Quantity"],
    product_performance["Total_Revenue"],
    alpha=0.5)
plt.title("Product Quantity vs Revenue")
plt.xlabel("Total Quantity Sold")
plt.ylabel("Total Revenue")
plt.show()

###product segmentation
quantity_median = (product_performance["Total_Quantity"].median())
revenue_median = (product_performance["Total_Revenue"].median())
print("Quantity Median:", quantity_median)
print("Revenue Median:", revenue_median)

def classify_product(row):
    if (row["Total_Quantity"] >= quantity_median
        and
        row["Total_Revenue"] >= revenue_median):
        return "High Volume - High Revenue"
    elif (row["Total_Quantity"] >= quantity_median
        and
        row["Total_Revenue"] < revenue_median):
        return "High Volume - Low Revenue"
    elif (row["Total_Quantity"] < quantity_median
        and
        row["Total_Revenue"] >= revenue_median):
        return "Low Volume - High Revenue"
    else:
        return "Low Volume - Low Revenue"

product_performance["Product_Segment"] = (product_performance.apply(classify_product,axis=1))
print(product_performance["Product_Segment"].value_counts())
segment_counts = (product_performance["Product_Segment"].value_counts())

plt.figure(figsize=(10, 6))
plt.bar(segment_counts.index,segment_counts.values)

plt.title("Product Performance Segments")
plt.xlabel("Product Segment")
plt.ylabel("Number of Products")
plt.xticks(rotation=30, ha="right")
plt.show()

high_performing_products = (
    product_performance[
        product_performance["Product_Segment"] == "High Volume - High Revenue"].sort_values("Total_Revenue",
        ascending=False))
print(
    high_performing_products[
        ["StockCode",
        "Description",
        "Total_Revenue",
        "Total_Quantity",
        "Number_of_Orders",
        "Average_Price"]].head(20))

premium_products = (
    product_performance[product_performance["Product_Segment"] == "Low Volume - High Revenue"].sort_values("Total_Revenue",ascending=False))

print(
    premium_products[
        [
            "StockCode",
            "Description",
            "Total_Revenue",
            "Total_Quantity",
            "Number_of_Orders",
            "Average_Price"]].head(20))

volume_products = (product_performance[product_performance["Product_Segment"] == "High Volume - Low Revenue"].sort_values("Total_Quantity",ascending=False))

print(volume_products[
        ["StockCode",
        "Description",
        "Total_Revenue",
        "Total_Quantity",
        "Number_of_Orders",
        "Average_Price"]].head(20))

###revenue contribution 
product_performance = (product_performance.sort_values("Total_Revenue",ascending=False).reset_index(drop=True))
product_performance["Revenue_Contribution_%"] = (product_performance["Total_Revenue"] / product_performance["Total_Revenue"].sum()) * 100
product_performance["Cumulative_Revenue_%"] = (product_performance["Revenue_Contribution_%"].cumsum())
print(
    product_performance[
        [
            "StockCode",
            "Description",
            "Total_Revenue",
            "Revenue_Contribution_%",
            "Cumulative_Revenue_%"
        ]
    ].head(20))

plt.figure(figsize=(12, 6))

plt.plot(range(1, len(product_performance) + 1),product_performance["Cumulative_Revenue_%"])
plt.axhline(80,linestyle="--")

plt.title("Pareto Analysis - Product Revenue")
plt.xlabel("Number of Products")
plt.ylabel("Cumulative Revenue (%)")
plt.show()

###Customer + Revenue + Orders + Quantity
print(customer_summary.head())
print("Customer Summary Statistics:")
print(customer_summary.describe())

###Customers by Revenue
top_customers = (customer_summary.sort_values("Total_Revenue",ascending=False).head(20))
print(top_customers)
print(top_customers[
        ["Customer_ID",
        "Total_Revenue",
        "Total_Quantity",
        "Total_Orders",
        "Average_Order_Value"]])

###customer order vs revenue
plt.figure(figsize=(10, 6))
plt.scatter(customer_summary["Total_Orders"],customer_summary["Total_Revenue"],alpha=0.5)
plt.title("Customer Orders vs Revenue")
plt.xlabel("Number of Orders")
plt.ylabel("Total Revenue")
plt.show()

customer_order_median = (customer_summary["Total_Orders"].median())
customer_revenue_median = (customer_summary["Total_Revenue"].median())
print("Order Median:",customer_order_median)
print("Revenue Median:",customer_revenue_median)

customer_order_median = (customer_summary["Total_Orders"].median())
customer_revenue_median = (customer_summary["Total_Revenue"].median())
print("Order Median:",customer_order_median)
print("Revenue Median:",customer_revenue_median)

customer_summary["Customer_Segment"] = (
    customer_summary.apply("missing_customer",axis=1))

print(customer_summary["Customer_Segment"].value_counts())

customer_segment_counts = (customer_summary["Customer_Segment"].value_counts())

plt.figure(figsize=(10, 6))
plt.bar(customer_segment_counts.index,customer_segment_counts.values)

plt.title("Customer Segmentation")
plt.xlabel("Customer Segment")
plt.ylabel("Number of Customers")

plt.xticks(rotation=30, ha="right")
plt.show()

customer_segment_revenue = (customer_summary.groupby("Customer_Segment")["Total_Revenue"].sum().sort_values(ascending=False))
print(customer_segment_revenue)

plt.figure(figsize=(10, 6))

plt.bar(customer_segment_revenue.index,customer_segment_revenue.values)

plt.title("Revenue by Customer Segment")
plt.xlabel("Customer Segment")
plt.ylabel("Total Revenue")
plt.xticks(rotation=30, ha="right")

plt.show()

plt.figure(figsize=(10, 6))

plt.scatter(customer_summary["Total_Quantity"],
    customer_summary["Total_Revenue"],
    s=customer_summary["Average_Order_Value"],
    alpha=0.5)

plt.title("Customer Quantity vs Revenue ""(Bubble Size = AOV)")
plt.xlabel("Total Quantity Purchased")
plt.ylabel("Total Revenue")

plt.show()


###customer evenue cooncentration

customer_performance = (
    customer_summary.sort_values("Total_Revenue",ascending=False).reset_index(drop=True))
customer_performance["Revenue_Contribution_%"] = (customer_performance["Total_Revenue"] / customer_performance["Total_Revenue"].sum()) * 100

customer_performance["Cumulative_Revenue_%"] = (
    customer_performance["Revenue_Contribution_%"].cumsum())
print(
    customer_performance[
        ["Customer_ID",
        "Total_Revenue",
        "Revenue_Contribution_%",
        "Cumulative_Revenue_%"]].head(20))

###country+customer+revenue
country_customer = (df_sales.groupby("Country").agg(Total_Revenue=("Revenue", "sum"),
        Number_of_Customers=("Customer_ID", "nunique"),
        Number_of_Orders=("Invoice", "nunique")).reset_index())
country_customer["Revenue_per_Customer"] = (country_customer["Total_Revenue"] / country_customer["Number_of_Customers"])
country_customer = (country_customer.sort_values("Total_Revenue",ascending=False))
print(country_customer.head(15))

top_country_analysis = (
    country_customer
    .head(15)
)

plt.figure(figsize=(10, 6))

plt.scatter(top_country_analysis["Number_of_Customers"],
    top_country_analysis["Total_Revenue"],
    s=top_country_analysis["Revenue_per_Customer"] / 10,
    alpha=0.6)
plt.title("Country Performance: Customers vs Revenue")
plt.xlabel("Number of Customers")
plt.ylabel("Total Revenue")
plt.show()

###kpisummary

total_revenue = df_sales["Revenue"].sum()
total_quantity = df_sales["Quantity"].sum()
total_orders = df_sales["Invoice"].nunique()
total_customers = df_sales["Customer_ID"].nunique()
total_products = df_sales["StockCode"].nunique()

overall_aov = (total_revenue / total_orders)
kpi_summary = pd.DataFrame({"Metric": [
        "Total Revenue",
        "Total Quantity Sold",
        "Total Orders",
        "Total Customers",
        "Total Products",
        "Overall Average Order Value"],
    "Value": [total_revenue,total_quantity,total_orders,total_customers,total_products,overall_aov]})
print(kpi_summary)

print("df_clean:", df_clean.shape)
print("df_sales:", df_sales.shape)
print("customer_summary:", customer_summary.shape)
print("product_performance:", product_performance.shape)
print("country_performance:", country_performance.shape)

### business & Kpi insight


total_revenue = df_sales["Revenue"].sum()
total_quantity = df_sales["Quantity"].sum()
total_orders = df_sales["Invoice"].nunique()
total_customers = df_sales["Customer_ID"].nunique()
total_products = df_sales["StockCode"].nunique()

average_order_value = (total_revenue / total_orders)
average_units_per_order = (total_quantity / total_orders)
kpi_summary = pd.DataFrame({
    "KPI": [
        "Total Revenue",
        "Total Quantity Sold",
        "Total Orders",
        "Total Customers",
        "Total Products",
        "Average Order Value",
        "Average Units per Order"
    ],
    "Value": [total_revenue,
        total_quantity,
        total_orders,
        total_customers,
        total_products,
        average_order_value,
        average_units_per_order]})
print(kpi_summary)

###Revenue Performance
print("Total Revenue:", total_revenue)
print("Average Order Value:", average_order_value)
print("Average Units per Order:", average_units_per_order)

monthly_revenue = (df_sales.groupby("Year_Month")["Revenue"].sum().sort_index())
print(monthly_revenue)

best_month = monthly_revenue.idxmax()
best_month_revenue = monthly_revenue.max()

print("Best Month:", best_month)
print("Revenue:", best_month_revenue)

worst_month = monthly_revenue.idxmin()
worst_month_revenue = monthly_revenue.min()

print("Lowest Month:", worst_month)
print("Revenue:", worst_month_revenue)

###monthly revenue growth
monthly_growth = (monthly_revenue.pct_change() * 100)
print(monthly_growth)

monthly_growth_table = pd.DataFrame({"Revenue": monthly_revenue,"Growth_%": monthly_growth})
print(monthly_growth_table)

###performance by country
country_business = (
    df_sales
    .groupby("Country")
    .agg(
        Revenue=("Revenue", "sum"),
        Orders=("Invoice", "nunique"),
        Customers=("Customer_ID", "nunique"),
        Quantity=("Quantity", "sum")
    )
    .reset_index()
)

country_business["AOV"] = (
    country_business["Revenue"] /
    country_business["Orders"]
)

country_business["Revenue_per_Customer"] = (
    country_business["Revenue"] /
    country_business["Customers"]
)

country_business = (
    country_business
    .sort_values("Revenue", ascending=False)
)

print(country_business.head(15))

print(
    country_business.head(10)
)

print(
    country_business.tail(10)
)

country_business["Revenue_Contribution_%"] = (
    country_business["Revenue"]
    / country_business["Revenue"].sum()
) * 100

print(
    country_business[
        [
            "Country",
            "Revenue",
            "Revenue_Contribution_%"
        ]
    ].head(15)
)

# ============================================
# 8.6 CUSTOMER PERFORMANCE
# ============================================

customer_business = (
    df_sales
    .groupby("Customer_ID")
    .agg(
        Revenue=("Revenue", "sum"),
        Orders=("Invoice", "nunique"),
        Quantity=("Quantity", "sum")
    )
    .reset_index()
)

customer_business["AOV"] = (
    customer_business["Revenue"] /
    customer_business["Orders"]
)

customer_business = (
    customer_business
    .sort_values("Revenue", ascending=False)
)

print(customer_business.head(20))

top_10_customer_revenue = (
    customer_business
    .head(10)["Revenue"]
    .sum()
)

top_10_customer_percentage = (
    top_10_customer_revenue /
    customer_business["Revenue"].sum()
) * 100

print(
    "Top 10 Customers Revenue Contribution:",
    top_10_customer_percentage,
    "%"
)

###product performance

product_business = (
    df_sales
    .groupby(["StockCode", "Description"])
    .agg(
        Revenue=("Revenue", "sum"),
        Quantity=("Quantity", "sum"),
        Orders=("Invoice", "nunique"),
        Average_Price=("Price", "mean")
    )
    .reset_index()
)

product_business["AOV"] = (
    product_business["Revenue"] /
    product_business["Orders"]
)

product_business = (
    product_business
    .sort_values("Revenue", ascending=False)
)
print(product_business.head(20))

top_10_products = product_business.head(10)

print(
    top_10_products[
        [
            "StockCode",
            "Description",
            "Revenue",
            "Quantity",
            "Orders",
            "Average_Price"
        ]
    ]
)

### SALES VS RETURNS

transaction_summary = (
    df_clean
    .groupby("Transaction_Type")
    .agg(
        Revenue=("Revenue", "sum"),
        Quantity=("Quantity", "sum"),
        Orders=("Invoice", "nunique")
    )
)

print(transaction_summary)

return_orders = (
    df_clean[
        df_clean["Transaction_Type"] == "Return"
    ]["Invoice"]
    .nunique()
)

total_all_orders = df_clean["Invoice"].nunique()
return_rate = (return_orders / total_all_orders) * 100

print("Return Order Rate:",return_rate,"%")

returned_quantity = abs(df_clean.loc[df_clean["Transaction_Type"] == "Return","Quantity"].sum())
sold_quantity = (df_sales["Quantity"].sum())
print("Sold Quantity:", sold_quantity)
print("Returned Quantity:", returned_quantity)

quantity_return_rate = (returned_quantity / sold_quantity) * 100

print(
    "Quantity Return Rate:",
    quantity_return_rate,
    "%"
)

cancellation_summary = (df_clean.groupby("Cancellation_Status").agg(Revenue=("Revenue", "sum"),
        Quantity=("Quantity", "sum"),
        Orders=("Invoice", "nunique")))
print(cancellation_summary)
cancelled_orders = (df_clean[df_clean["Cancellation_Status"] == "Cancelled"]["Invoice"].nunique())

cancellation_rate = (cancelled_orders / total_all_orders) * 100
print("Cancellation Rate:",cancellation_rate,"%")

###day performance

day_business = (
    df_sales
    .groupby("Day_of_Week")
    .agg(
        Revenue=("Revenue", "sum"),
        Orders=("Invoice", "nunique"),
        Quantity=("Quantity", "sum")
    )
    .reindex(day_order)
)

day_business["AOV"] = (
    day_business["Revenue"] /
    day_business["Orders"]
)
print(day_business)
best_day = day_business["Revenue"].idxmax()
print("Best Revenue Day:",best_day)

# ============================================
# 8.15 HOURLY PERFORMANCE
# ============================================

hour_business = (
    df_sales
    .groupby("Hour")
    .agg(
        Revenue=("Revenue", "sum"),
        Orders=("Invoice", "nunique"),
        Quantity=("Quantity", "sum")
    )
    .sort_index()
)

hour_business["AOV"] = (
    hour_business["Revenue"] /
    hour_business["Orders"]
)

print(hour_business)
best_hour = hour_business["Revenue"].idxmax()

print(
    "Best Revenue Hour:",
    best_hour
)
day_hour_business = pd.pivot_table(
    df_sales,
    values="Revenue",
    index="Day_of_Week",
    columns="Hour",
    aggfunc="sum",
    fill_value=0
)

day_hour_business = (
    day_hour_business
    .reindex(day_order)
)

print(day_hour_business)
plt.figure(figsize=(14, 6))

sns.heatmap(
    day_hour_business,
    annot=False
)

plt.title(
    "Revenue by Day and Hour"
)

plt.xlabel("Hour")
plt.ylabel("Day")

plt.show()
# ============================================
# 8.17 COUNTRY + MONTH PERFORMANCE
# ============================================

country_month = pd.pivot_table(
    df_sales,
    values="Revenue",
    index="Country",
    columns="Year_Month",
    aggfunc="sum",
    fill_value=0
)

top_countries = (
    country_business
    .head(10)
    ["Country"]
)

country_month_top10 = (
    country_month
    .loc[top_countries]
)

plt.figure(figsize=(14, 7))

sns.heatmap(
    country_month_top10
)

plt.title(
    "Top 10 Countries - Monthly Revenue"
)

plt.xlabel("Year-Month")
plt.ylabel("Country")

plt.show()

###KPI Calcualtion
business_kpis = {
    "Total Revenue": total_revenue,
    "Total Quantity Sold": total_quantity,
    "Total Orders": total_orders,
    "Total Customers": total_customers,
    "Total Products": total_products,
    "Average Order Value": average_order_value,
    "Average Units per Order": average_units_per_order,
    "Return Order Rate (%)": return_rate,
    "Quantity Return Rate (%)": quantity_return_rate,
    "Cancellation Rate (%)": cancellation_rate,
    "Top 10 Customer Revenue Share (%)": top_10_customer_percentage}

for metric, value in business_kpis.items():
    print(f"{metric}: {value:,.2f}")

###Business Insight Tabel

insight_summary = pd.DataFrame({
    "Analysis": ["Highest Revenue Country",
        "Highest Revenue Product",
        "Best Revenue Day",
        "Best Revenue Hour",
        "Best Revenue Month",
        "Lowest Revenue Month"],
    "Result": [country_business.iloc[0]["Country"],
        product_business.iloc[0]["Description"],
        best_day,
        best_hour,
        best_month,
        worst_month]})
print(insight_summary)

###PARETO ANALYSIS

product_count = len(product_business)

top_20_percent_count = int(
    product_count * 0.20
)

top_20_revenue_share = (
    product_business
    .head(top_20_percent_count)["Revenue"]
    .sum()
    /
    product_business["Revenue"].sum()) * 100

print("Number of Products:",product_count)

print("Top 20% Product Count:",top_20_percent_count)
print("Revenue Generated by Top 20%:",top_20_revenue_share,"%")


###REVENUE INSIGHTS

print("Total Revenue:", f"{total_revenue:,.2f}")
print("Average Order Value:", f"{average_order_value:,.2f}")
print("Average Units per Order:", f"{average_units_per_order:,.2f}")

print("\nBest Revenue Month:", best_month)
print("Best Month Revenue:", f"{best_month_revenue:,.2f}")

print("\nLowest Revenue Month:", worst_month)
print("Lowest Month Revenue:", f"{worst_month_revenue:,.2f}")

# 9.3 REVENUE GROWTH

growth_clean = monthly_growth.dropna()

best_growth_month = growth_clean.idxmax()
best_growth_rate = growth_clean.max()

worst_growth_month = growth_clean.idxmin()
worst_growth_rate = growth_clean.min()

print(
    "Highest Monthly Growth:",
    best_growth_month,
    f"({best_growth_rate:.2f}%)"
)

print(
    "Lowest Monthly Growth:",
    worst_growth_month,
    f"({worst_growth_rate:.2f}%)"
)

# 9.4 CUSTOMER INSIGHTS

print(
    "Total Customers:",
    total_customers
)

print(
    "Top 10 Customer Revenue Share:",
    f"{top_10_customer_percentage:.2f}%"
)

print("\nTop 10 Customers:")

print(
    customer_business[
        [
            "Customer_ID",
            "Revenue",
            "Orders",
            "Quantity",
            "AOV"
        ]
    ].head(10)
)


####Customer Insights

print(
    "Total Customers:",
    total_customers
)

print(
    "Top 10 Customer Revenue Share:",
    f"{top_10_customer_percentage:.2f}%"
)

print("\nTop 10 Customers:")

print(
    customer_business[
        [
            "Customer_ID",
            "Revenue",
            "Orders",
            "Quantity",
            "AOV"
        ]
    ].head(10)
)
final_segment_analysis = (
    final_segment_analysis.sort_values("Revenue",ascending=False)
)

print(final_segment_analysis)

# ============================================
# 9.6 PRODUCT INSIGHTS
# ============================================

print(
    "Total Products:",
    total_products
)

print(
    "Top 10 Product Revenue Share:",
    f"{top_10_product_percentage:.2f}%"
)

print(
    "Top 20% Product Revenue Share:",
    f"{top_20_revenue_share:.2f}%"
)
print(
    product_business[
        [
            "StockCode",
            "Description",
            "Revenue",
            "Quantity",
            "Orders",
            "Average_Price"
        ]
    ].head(10)
)
# ============================================
# 9.7 PRODUCT SEGMENTS
# ============================================

product_segment_summary = (
    product_performance
    .groupby("Product_Segment")
    .agg(
        Products=("StockCode", "count"),
        Revenue=("Total_Revenue", "sum"),
        Quantity=("Total_Quantity", "sum"),
        Orders=("Number_of_Orders", "sum")
    )
    .reset_index()
)

product_segment_summary["Revenue_Share_%"] = (
    product_segment_summary["Revenue"]
    / product_segment_summary["Revenue"].sum()
) * 100

print(product_segment_summary)

# ============================================
# 9.8 COUNTRY INSIGHTS
# ============================================

print(
    country_business[
        [
            "Country",
            "Revenue",
            "Orders",
            "Customers",
            "Quantity",
            "AOV",
            "Revenue_per_Customer"
        ]
    ].head(15)
)

top_country = country_business.iloc[0]

print("Top Revenue Country:", top_country["Country"])
print(
    "Revenue:",
    f"{top_country['Revenue']:,.2f}"
)
print(
    "Customers:",
    f"{top_country['Customers']:,.0f}"
)
print(
    "AOV:",
    f"{top_country['AOV']:,.2f}"
)

# ============================================
# 9.9 TOP 5 COUNTRY CONTRIBUTION
# ============================================

top_5_country_share = (
    country_business
    .head(5)["Revenue"]
    .sum()
    /
    country_business["Revenue"].sum()
) * 100

print(
    "Top 5 Countries Revenue Share:",
    f"{top_5_country_share:.2f}%"
)

# ============================================
# 9.10 RETURN INSIGHTS
# ============================================

print(
    "Return Order Rate:",
    f"{return_rate:.2f}%"
)

print(
    "Quantity Return Rate:",
    f"{quantity_return_rate:.2f}%"
)

print(
    "Returned Quantity:",
    f"{returned_quantity:,.0f}"
)

returned_products = (
    df_clean[
        df_clean["Transaction_Type"] == "Return"
    ]
    .groupby(
        ["StockCode", "Description"]
    )
    ["Quantity"]
    .sum()
    .abs()
    .sort_values(
        ascending=False
    )
    .head(10)
)

print(returned_products)
# ============================================
# 9.11 CANCELLATION INSIGHTS
# ============================================

print(
    "Cancellation Rate:",
    f"{cancellation_rate:.2f}%"
)

print(
    "Cancelled Orders:",
    cancelled_orders
)
# ============================================
# 9.12 BEST DAY-HOUR COMBINATION
# ============================================

day_hour_long = (
    day_hour_business
    .stack()
    .reset_index()
)

day_hour_long.columns = [
    "Day_of_Week",
    "Hour",
    "Revenue"
]

best_day_hour = (
    day_hour_long
    .sort_values(
        "Revenue",
        ascending=False
    )
    .iloc[0]
)

print(
    "Best Day:",
    best_day_hour["Day_of_Week"]
)

print(
    "Best Hour:",
    best_day_hour["Hour"]
)

print(
    "Revenue:",
    f"{best_day_hour['Revenue']:,.2f}"
)
# ============================================
# 9.13 FINAL FINDINGS TABLE
# ============================================

final_findings = pd.DataFrame({
    "Area": [
        "Revenue",
        "Revenue Growth",
        "Customer",
        "Product",
        "Market",
        "Operations",
        "Sales Timing"
    ],
    
    "Finding": [
        f"Highest revenue month: {best_month}",
        
        f"Highest monthly growth: "
        f"{best_growth_month} ({best_growth_rate:.2f}%)",
        
        f"Top 10 customers contribute "
        f"{top_10_customer_percentage:.2f}% of revenue",
        
        f"Top 20% products contribute "
        f"{top_20_revenue_share:.2f}% of revenue",
        
        f"Highest revenue market: "
        f"{top_country['Country']}",
        
        f"Return order rate: "
        f"{return_rate:.2f}%",
        
        f"Best sales period: "
        f"{best_day_hour['Day_of_Week']} "
        f"at {best_day_hour['Hour']}:00"
    ]
})

print(final_findings)

# Select features for customer segmentation

segmentation_df = customer_df[
    [
        "Recency",
        "Frequency",
        "Monetary",
        "Avg_Order_Value",
        "Total_Quantity"
    ]
].copy()

segmentation_df.head()
print("Shape:", segmentation_df.shape)

print("\nMissing Values:")
print(segmentation_df.isnull().sum())

print("\nData Types:")
print(segmentation_df.dtypes)

import matplotlib.pyplot as plt

segmentation_df.hist(
    figsize=(12, 8),
    bins=30
)

plt.suptitle("Customer Segmentation Feature Distributions")
plt.tight_layout()
plt.show()

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

segmentation_scaled = scaler.fit_transform(segmentation_df)

segmentation_scaled = pd.DataFrame(
    segmentation_scaled,
    columns=segmentation_df.columns,
    index=segmentation_df.index
)

segmentation_scaled.head()

from sklearn.cluster import KMeans

inertia = []

for k in range(2, 11):
    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )
    
    kmeans.fit(segmentation_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))

plt.plot(
    range(2, 11),
    inertia,
    marker="o"
)

plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.title("Elbow Method for Optimal Number of Clusters")
plt.xticks(range(2, 11))
plt.grid(True)

plt.show()

optimal_k = 4

kmeans = KMeans(
    n_clusters=optimal_k,
    random_state=42,
    n_init=10
)

segmentation_df["Cluster"] = kmeans.fit_predict(
    segmentation_scaled
)

segmentation_df.head()

cluster_summary = segmentation_df.groupby("Cluster").agg({
    "Recency": "mean",
    "Frequency": "mean",
    "Monetary": "mean",
    "Avg_Order_Value": "mean",
    "Total_Quantity": "mean"
}).round(2)

cluster_summary

plt.figure(figsize=(9, 6))

for cluster in sorted(segmentation_df["Cluster"].unique()):
    cluster_data = segmentation_df[
        segmentation_df["Cluster"] == cluster
    ]
    
    plt.scatter(
        cluster_data["Frequency"],
        cluster_data["Monetary"],
        label=f"Cluster {cluster}",
        alpha=0.6
    )

plt.xlabel("Purchase Frequency")
plt.ylabel("Monetary Value")
plt.title("Customer Segmentation: Frequency vs Monetary Value")
plt.legend()
plt.grid(True)

plt.show()

cluster_size = (
    segmentation_df["Cluster"]
    .value_counts()
    .sort_index()
)

print(cluster_size)

cluster_percentage = (
    segmentation_df["Cluster"]
    .value_counts(normalize=True)
    .sort_index()
    .mul(100)
    .round(2)
)

print(cluster_percentage)
cluster_counts = (
    segmentation_df["Cluster"]
    .value_counts()
    .sort_index()
)

plt.figure(figsize=(8, 5))

plt.bar(
    cluster_counts.index.astype(str),
    cluster_counts.values
)

plt.xlabel("Customer Cluster")
plt.ylabel("Number of Customers")
plt.title("Customer Distribution by Cluster")

plt.show()

plt.figure(figsize=(8, 5))

cluster_summary["Monetary"].plot(
    kind="bar"
)

plt.xlabel("Customer Cluster")
plt.ylabel("Average Monetary Value")
plt.title("Average Customer Spending by Cluster")
plt.xticks(rotation=0)

plt.show()

plt.figure(figsize=(8, 5))

cluster_summary["Frequency"].plot(
    kind="bar"
)

plt.xlabel("Customer Cluster")
plt.ylabel("Average Purchase Frequency")
plt.title("Average Purchase Frequency by Cluster")
plt.xticks(rotation=0)

plt.show()

plt.figure(figsize=(8, 5))

cluster_summary["Recency"].plot(
    kind="bar"
)

plt.xlabel("Customer Cluster")
plt.ylabel("Average Recency")
plt.title("Average Recency by Customer Cluster")
plt.xticks(rotation=0)

plt.show()

rfm_cluster_summary = segmentation_df.groupby("Cluster").agg(
    Customers=("Cluster", "count"),
    Avg_Recency=("Recency", "mean"),
    Avg_Frequency=("Frequency", "mean"),
    Avg_Monetary=("Monetary", "mean"),
    Avg_Order_Value=("Avg_Order_Value", "mean"),
    Avg_Quantity=("Total_Quantity", "mean")
).round(2)

rfm_cluster_summary
cluster_revenue = (
    segmentation_df.groupby("Cluster")["Monetary"]
    .sum()
    .sort_values(ascending=False)
)

cluster_revenue

cluster_revenue_percentage = (
    cluster_revenue /
    cluster_revenue.sum() * 100
).round(2)

cluster_revenue_percentage

plt.figure(figsize=(8, 5))

plt.bar(
    cluster_revenue_percentage.index.astype(str),
    cluster_revenue_percentage.values
)

plt.xlabel("Customer Cluster")
plt.ylabel("Revenue Contribution (%)")
plt.title("Revenue Contribution by Customer Cluster")

plt.show()

final_segment_analysis = segmentation_df.groupby("Cluster").agg(
    Customers=("Cluster", "count"),
    Avg_Recency=("Recency", "mean"),
    Avg_Frequency=("Frequency", "mean"),
    Avg_Monetary=("Monetary", "mean"),
    Avg_Order_Value=("Avg_Order_Value", "mean"),
    Avg_Quantity=("Total_Quantity", "mean")
).round(2)

final_segment_analysis["Customer_%"] = (
    final_segment_analysis["Customers"] /
    final_segment_analysis["Customers"].sum() * 100
).round(2)

final_segment_analysis["Revenue_%"] = (
    segmentation_df.groupby("Cluster")["Monetary"].sum() /
    segmentation_df["Monetary"].sum() * 100
).round(2)

final_segment_analysis

segmentation_df.to_csv(
    "customer_segmentation.csv",
    index=True
)

print("Customer segmentation file saved successfully.")

final_segment_analysis

# Overall Business KPIs

total_revenue = df_clean["Revenue"].sum()
total_quantity = df_clean["Quantity"].sum()
total_orders = df_clean["Invoice"].nunique()
total_customers = df_clean["Customer ID"].nunique()
total_products = df_clean["StockCode"].nunique()

average_order_value = total_revenue / total_orders
average_quantity_per_order = total_quantity / total_orders
average_revenue_per_customer = total_revenue / total_customers

print("===== Overall Business KPIs =====")

print(f"Total Revenue: £{total_revenue:,.2f}")
print(f"Total Quantity Sold: {total_quantity:,}")
print(f"Total Orders: {total_orders:,}")
print(f"Total Customers: {total_customers:,}")
print(f"Total Products: {total_products:,}")
print(f"Average Order Value: £{average_order_value:,.2f}")
print(f"Average Quantity per Order: {average_quantity_per_order:,.2f}")
print(f"Average Revenue per Customer: £{average_revenue_per_customer:,.2f}")

df_clean["InvoiceDate"] = pd.to_datetime(
    df_clean["InvoiceDate"]
)

df_clean["YearMonth"] = (
    df_clean["InvoiceDate"]
    .dt.to_period("M")
)

monthly_revenue = (
    df_clean.groupby("YearMonth")["Revenue"]
    .sum()
    .sort_index()
)

monthly_revenue

plt.figure(figsize=(12, 5))

plt.plot(
    monthly_revenue.index.astype(str),
    monthly_revenue.values,
    marker="o"
)

plt.xlabel("Month")
plt.ylabel("Revenue (£)")
plt.title("Monthly Revenue Trend")
plt.xticks(rotation=45)
plt.grid(True)

plt.show()

monthly_orders = (
    df_clean.groupby("YearMonth")["Invoice"]
    .nunique()
    .sort_index()
)

monthly_orders

plt.figure(figsize=(12, 5))

plt.plot(
    monthly_orders.index.astype(str),
    monthly_orders.values,
    marker="o"
)

plt.xlabel("Month")
plt.ylabel("Number of Orders")
plt.title("Monthly Order Trend")
plt.xticks(rotation=45)
plt.grid(True)

plt.show()

monthly_aov = (
    monthly_revenue / monthly_orders
).round(2)

monthly_aov

plt.figure(figsize=(12, 5))

plt.plot(
    monthly_aov.index.astype(str),
    monthly_aov.values,
    marker="o"
)

plt.xlabel("Month")
plt.ylabel("Average Order Value (£)")
plt.title("Monthly Average Order Value")
plt.xticks(rotation=45)
plt.grid(True)

plt.show()

top_products_revenue = (
    df_clean.groupby("StockCode")["Revenue"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

top_products_revenue

plt.figure(figsize=(10, 6))

plt.barh(
    top_products_revenue.index.astype(str),
    top_products_revenue.values
)

plt.xlabel("Revenue (£)")
plt.ylabel("Stock Code")
plt.title("Top 10 Products by Revenue")
plt.gca().invert_yaxis()

plt.show()

top_products_quantity = (
    df_clean.groupby("StockCode")["Quantity"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

top_products_quantity

plt.figure(figsize=(10, 6))

plt.barh(
    top_products_quantity.index.astype(str),
    top_products_quantity.values
)

plt.xlabel("Quantity Sold")
plt.ylabel("Stock Code")
plt.title("Top 10 Products by Quantity Sold")
plt.gca().invert_yaxis()

plt.show()

country_revenue = (
    df_clean.groupby("Country")["Revenue"]
    .sum()
    .sort_values(ascending=False)
)

country_revenue.head(10)

top_countries = country_revenue.head(10)

plt.figure(figsize=(10, 6))

plt.barh(
    top_countries.index,
    top_countries.values
)

plt.xlabel("Revenue (£)")
plt.ylabel("Country")
plt.title("Top 10 Countries by Revenue")
plt.gca().invert_yaxis()

plt.show()

top_customers = (
    df_clean.groupby("Customer ID")["Revenue"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

top_customers

plt.figure(figsize=(10, 6))

plt.barh(
    top_customers.index.astype(str),
    top_customers.values
)

plt.xlabel("Revenue (£)")
plt.ylabel("Customer ID")
plt.title("Top 10 Customers by Revenue")
plt.gca().invert_yaxis()

plt.show()

customer_revenue = (
    df_clean.groupby("Customer ID")["Revenue"]
    .sum()
    .sort_values(ascending=False)
)

top_10_customer_revenue = customer_revenue.head(10).sum()

top_10_customer_percentage = (
    top_10_customer_revenue /
    total_revenue
) * 100

print(
    f"Top 10 customers contribute "
    f"{top_10_customer_percentage:.2f}% of total revenue."
)

customer_revenue_df = (
    customer_revenue
    .reset_index()
)

customer_revenue_df["Cumulative_Revenue"] = (
    customer_revenue_df["Revenue"].cumsum()
)

customer_revenue_df["Cumulative_Revenue_%"] = (
    customer_revenue_df["Cumulative_Revenue"] /
    total_revenue * 100
)

customer_revenue_df.head()

customers_for_80_percent = (
    customer_revenue_df["Cumulative_Revenue_%"] <= 80
).sum()

print(
    f"{customers_for_80_percent:,} customers "
    f"generate approximately 80% of revenue.")

product_performance["Revenue_Per_Unit"] = (
    product_performance["Total_Revenue"] /
    product_performance["Total_Quantity"]
).round(2)

product_performance.head()

high_demand_products = (
    product_performance
    .sort_values("Total_Quantity", ascending=False)
    .head(10)
)

high_demand_products

high_revenue_products = (
    product_performance
    .sort_values("Total_Revenue", ascending=False)
    .head(10)
)

high_revenue_products

print("Top Products by Quantity:")
print(high_demand_products.index.tolist())

print("\nTop Products by Revenue:")
print(high_revenue_products.index.tolist())

plt.figure(figsize=(10, 6))

plt.scatter(
    product_performance["Total_Quantity"],
    product_performance["Total_Revenue"],
    alpha=0.5
)

plt.xlabel("Total Quantity Sold")
plt.ylabel("Total Revenue (£)")
plt.title("Product Demand vs Revenue")

plt.grid(True)
plt.show()

quantity_threshold = product_performance["Total_Quantity"].median()
revenue_threshold = product_performance["Total_Revenue"].median()

print("Quantity Threshold:", quantity_threshold)
print("Revenue Threshold:", revenue_threshold)

product_performance["Performance_Category"] = "Other"

product_performance.loc[
    (product_performance["Total_Quantity"] >= quantity_threshold) &
    (product_performance["Total_Revenue"] >= revenue_threshold),
    "Performance_Category"
] = "High Demand - High Revenue"

product_performance.loc[
    (product_performance["Total_Quantity"] >= quantity_threshold) &
    (product_performance["Total_Revenue"] < revenue_threshold),
    "Performance_Category"
] = "High Demand - Low Revenue"

product_performance.loc[
    (product_performance["Total_Quantity"] < quantity_threshold) &
    (product_performance["Total_Revenue"] >= revenue_threshold),
    "Performance_Category"
] = "Low Demand - High Revenue"

product_performance.loc[
    (product_performance["Total_Quantity"] < quantity_threshold) &
    (product_performance["Total_Revenue"] < revenue_threshold),
    "Performance_Category"
] = "Low Demand - Low Revenue"

product_performance["Performance_Category"].value_counts()

performance_counts = (
    product_performance["Performance_Category"]
    .value_counts()
)

plt.figure(figsize=(10, 5))

plt.bar(
    performance_counts.index,
    performance_counts.values
)

plt.xlabel("Product Performance Category")
plt.ylabel("Number of Products")
plt.title("Product Performance Distribution")
plt.xticks(rotation=25)
plt.show()

high_demand_low_revenue = product_performance[
    product_performance["Performance_Category"] ==
    "High Demand - Low Revenue"
].sort_values(
    "Total_Quantity",
    ascending=False
)

high_demand_low_revenue.head(10)

low_demand_high_revenue = product_performance[
    product_performance["Performance_Category"] ==
    "Low Demand - High Revenue"
].sort_values(
    "Total_Revenue",
    ascending=False
)

low_demand_high_revenue.head(10)

product_performance["Revenue_Contribution_%"] = (
    product_performance["Total_Revenue"] /
    total_revenue * 100
).round(2)

product_performance.sort_values(
    "Revenue_Contribution_%",
    ascending=False
).head(10)

product_pareto = (
    product_performance
    .sort_values(
        "Total_Revenue",
        ascending=False
    )
    .copy()
)

product_pareto["Cumulative_Revenue"] = (
    product_pareto["Total_Revenue"].cumsum()
)

product_pareto["Cumulative_Revenue_%"] = (
    product_pareto["Cumulative_Revenue"] /
    total_revenue * 100
)

product_pareto.head()

products_for_80_percent = (
    product_pareto["Cumulative_Revenue_%"] <= 80
).sum()

total_product_count = product_performance.shape[0]

print(
    f"{products_for_80_percent:,} out of "
    f"{total_product_count:,} products generate "
    f"approximately 80% of total revenue."
)

business_kpis = {
    "Total Revenue": total_revenue,
    "Total Quantity Sold": total_quantity,
    "Total Orders": total_orders,
    "Total Customers": total_customers,
    "Total Products": total_products,
    "Average Order Value": average_order_value,
    "Average Quantity per Order": average_quantity_per_order,
    "Average Revenue per Customer": average_revenue_per_customer
}

business_kpis
for kpi, value in business_kpis.items():
    if "Revenue" in kpi or "Value" in kpi:
        print(f"{kpi}: £{value:,.2f}")
    else:
        print(f"{kpi}: {value:,.2f}")

kpi_summary = pd.DataFrame({
    "Metric": [
        "Total Revenue",
        "Total Orders",
        "Total Customers",
        "Total Products",
        "Average Order Value",
        "Total Quantity Sold"
    ],
    "Value": [
        total_revenue,
        total_orders,
        total_customers,
        total_products,
        average_order_value,
        total_quantity
    ]
})

kpi_summary

plt.figure(figsize=(12, 5))

plt.plot(
    monthly_revenue.index.astype(str),
    monthly_revenue.values,
    marker="o"
)

plt.xlabel("Month")
plt.ylabel("Revenue (£)")
plt.title("Monthly Revenue Trend")

plt.xticks(rotation=45)
plt.grid(True)

plt.show()

plt.figure(figsize=(12, 5))

plt.plot(
    monthly_orders.index.astype(str),
    monthly_orders.values,
    marker="o"
)

plt.xlabel("Month")
plt.ylabel("Orders")
plt.title("Monthly Order Trend")

plt.xticks(rotation=45)
plt.grid(True)

plt.show()

top_10_products = (
    product_performance
    .sort_values("Total_Revenue", ascending=False)
    .head(10)
)

plt.figure(figsize=(10, 6))

plt.barh(
    top_10_products.index.astype(str),
    top_10_products["Total_Revenue"]
)

plt.xlabel("Revenue (£)")
plt.ylabel("Stock Code")
plt.title("Top 10 Products by Revenue")

plt.gca().invert_yaxis()

plt.show()

top_10_countries = (
    df_clean.groupby("Country")["Revenue"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

plt.figure(figsize=(10, 6))

plt.barh(
    top_10_countries.index,
    top_10_countries.values
)

plt.xlabel("Revenue (£)")
plt.ylabel("Country")
plt.title("Top 10 Countries by Revenue")

plt.gca().invert_yaxis()

plt.show()

segment_counts = (
    segmentation_df["Cluster"]
    .value_counts()
    .sort_index()
)

plt.figure(figsize=(8, 5))

plt.bar(
    segment_counts.index.astype(str),
    segment_counts.values
)

plt.xlabel("Customer Cluster")
plt.ylabel("Number of Customers")
plt.title("Customer Distribution by Segment")

plt.show()

segment_revenue = (
    segmentation_df.groupby("Cluster")["Monetary"]
    .sum()
    .sort_index()
)

plt.figure(figsize=(8, 5))

plt.bar(
    segment_revenue.index.astype(str),
    segment_revenue.values
)

plt.xlabel("Customer Cluster")
plt.ylabel("Revenue (£)")
plt.title("Revenue Contribution by Customer Segment")

plt.show()

plt.figure(figsize=(10, 6))

plt.scatter(
    product_performance["Total_Quantity"],
    product_performance["Total_Revenue"],
    alpha=0.5
)

plt.xlabel("Quantity Sold")
plt.ylabel("Revenue (£)")
plt.title("Product Demand vs Revenue")

plt.grid(True)

plt.show()

plt.figure(figsize=(9, 5))

plt.hist(
    df_clean["Revenue"],
    bins=50
)

plt.xlabel("Revenue per Transaction (£)")
plt.ylabel("Frequency")
plt.title("Distribution of Transaction Revenue")

plt.show()

plt.figure(figsize=(9, 5))

plt.hist(
    segmentation_df["Monetary"],
    bins=40
)

plt.xlabel("Customer Monetary Value (£)")
plt.ylabel("Number of Customers")
plt.title("Distribution of Customer Spending")

plt.show()

correlation_columns = [
    "Quantity",
    "UnitPrice",
    "Revenue"
]

correlation_matrix = (
    df_clean[correlation_columns]
    .corr()
)

correlation_matrix

plt.figure(figsize=(7, 5))

plt.imshow(
    correlation_matrix,
    interpolation="nearest"
)

plt.colorbar()

plt.xticks(
    range(len(correlation_matrix.columns)),
    correlation_matrix.columns
)

plt.yticks(
    range(len(correlation_matrix.columns)),
    correlation_matrix.columns
)

plt.title("Correlation Matrix")

plt.show()

dashboard_summary = {
    "Total Revenue": total_revenue,
    "Total Orders": total_orders,
    "Total Customers": total_customers,
    "Total Products": total_products,
    "Total Quantity": total_quantity,
    "Average Order Value": average_order_value,
    "Average Quantity per Order": average_quantity_per_order,
    "Average Revenue per Customer": average_revenue_per_customer
}

dashboard_summary

dashboard_kpis = pd.DataFrame(
    list(dashboard_summary.items()),
    columns=["KPI", "Value"]
)

dashboard_kpis.to_csv(
    "dashboard_kpis.csv",
    index=False
)

product_performance.to_csv(
    "product_performance.csv",
    index=True
)

final_segment_analysis.to_csv(
    "customer_segments.csv",
    index=True
)

monthly_revenue.to_csv(
    "monthly_revenue.csv"
)

print("Dashboard datasets exported successfully.")

final_kpis = pd.DataFrame({
    "KPI": [
        "Total Revenue",
        "Total Orders",
        "Total Customers",
        "Total Products",
        "Total Quantity Sold",
        "Average Order Value",
        "Average Quantity per Order",
        "Average Revenue per Customer"
    ],
    "Value": [
        total_revenue,
        total_orders,
        total_customers,
        total_products,
        total_quantity,
        average_order_value,
        average_quantity_per_order,
        average_revenue_per_customer
    ]
})

final_kpis

best_revenue_month = monthly_revenue.idxmax()
best_revenue_value = monthly_revenue.max()

print(f"Best Revenue Month: {best_revenue_month}")
print(f"Revenue: £{best_revenue_value:,.2f}")

best_country = country_revenue.idxmax()
best_country_revenue = country_revenue.max()

print(f"Top Revenue Country: {best_country}")
print(f"Revenue: £{best_country_revenue:,.2f}")

best_product = product_performance["Total_Revenue"].idxmax()
best_product_revenue = product_performance["Total_Revenue"].max()

print(f"Top Revenue Product: {best_product}")
print(f"Revenue: £{best_product_revenue:,.2f}")

best_customer = customer_revenue.idxmax()
best_customer_revenue = customer_revenue.max()

print(f"Top Customer: {best_customer}")
print(f"Revenue: £{best_customer_revenue:,.2f}")

print(
    f"Top 10 customers contribute "
    f"{top_10_customer_percentage:.2f}% "
    f"of total revenue."
)
print(
    f"{products_for_80_percent:,} products generate "
    f"approximately 80% of total revenue."
)

final_segment_analysis



