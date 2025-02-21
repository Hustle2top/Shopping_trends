import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import us
plt.ion()

df = pd.read_csv(r"C:\Users\Ravz\Desktop\Python\Kaggle_Shopping\shopping_trends.csv")
# print(df.info())
# print(df.head(10))

# print(df.isnull().sum())
# print(df.duplicated().sum())
# print(df.dtypes)

cat_columns = df.select_dtypes(include=['object']).columns
# print(cat_columns)

num_columns = df.select_dtypes(include = ['int64', 'float64']).columns
num_columns = num_columns.drop('Customer ID', errors='ignore') #-- This has dropped the Customer ID column
# print(num_columns)

# print(df["Gender"].unique())

# for col in cat_columns:
#     print(f"Unique values in {col}: {df[col].unique()}")  #-- This for loop gives all unique values for all categorical columns

# print(df.describe())

for col in num_columns:
    plt.figure(figsize=(6,3))
    sns.boxplot(x=df[col])
    plt.title(f"Box plot of {col}")

# plt.show()  #-- This shows box plots in different pop up images for different columns,one at a time and also without outlier dots

plt.figure(figsize=(15, 8))
sns.boxplot(data=df[num_columns], showfliers=True)
plt.yscale("log")
plt.title("Box Plots of Numerical Columns")
# plt.show(block=True)

# for col in cat_columns:
#     print(df[col].value_counts())
#     print("\n" + "-"*50 + "\n")   #--This will show the counts for unique values in each column, seperated by a line for every column

df_numeric = df[num_columns]
plt.figure(figsize=(15,5))
sns.heatmap(df_numeric.corr(), annot= True, cmap="coolwarm")
plt.title("Correlation Heatmap")
# plt.show(block=True)

df['Age Group'] = pd.cut(df['Age'], bins=[18, 25, 35, 45, 55, 65, 100], labels=['18-25', '26-35', '36-45', '46-55', '56-65', '65+'], include_lowest=True)
# print(df['Age Group'])

freq_mapping = { 'Weekly' : 7, 'Bi-Weekly' : 14, 'Fortnightly' : 14, 'Monthly' : 30, 'Quarterly' : 90, 'Every 3 Months' : 90, 'Annually' : 365}
df['Purchase days frequency'] = df['Frequency of Purchases'].map(freq_mapping)
# print(df['Purchase days frequency'].describe())

plt.figure(figsize= (10, 6))
sns.histplot(df['Purchase days frequency'],bins= 10, kde=True)
plt.title('Distribution of purchase frequency in Days')
# plt.show(block=True)


def categorize_frequency(days):
    if days <= 7:
        return "Frequent"
    elif days <= 30:
        return "Regular"
    elif days <= 90:
        return "Ocassional"
    else:
        return "Rare"

df["Customer Type"] = df['Purchase days frequency'].apply(categorize_frequency)
# print(df["Customer Type"])  #--the above function created categories of buyers and then we created a new column names Customer type

segments = df["Customer Type"].value_counts()
# print(segments)

plt.figure(figsize=(12, 6))
sns.barplot(x=segments.index, y=segments.values)
plt.title('Customer Segmentation by Purchase Frequency')
plt.xlabel('Customer Type')
plt.ylabel('Number of Customers')
# plt.show(block=True) # this generates a bar chart with customer type in X axis and number of customers in Y axis

segment_revenue = df.groupby('Customer Type')['Purchase Amount (USD)'].sum().reset_index()
# print(segment_revenue)  #--This is customer segmentation based on type of customer i.e. frequency of purchases

plt.figure(figsize=(10,7))
plt.pie(segment_revenue['Purchase Amount (USD)'], labels=segment_revenue['Customer Type'],autopct='%1.1f%%', colors=['Blue', 'orange', 'red', 'green'])
plt.pie(segment_revenue['Purchase Amount (USD)'], labels=segment_revenue['Customer Type'],autopct=lambda p: f'${p * segment_revenue["Purchase Amount (USD)"].sum() / 100:,.0f}', colors=['Pink', 'orange', 'red', 'green'])
plt.title('Revenue Contribution by Customer Segment')
# plt.show(block=True)  #--This creates a pie chart for total contribution by each segment(plt.pie has 2 lines above, use any ONLY one out of them)

customer_purchases = df.groupby('Customer ID').size().reset_index(name= 'Purchase Count')
# print(customer_purchases)

top_items = df["Item Purchased"].value_counts().head(10)
# print(top_items) #--This will give the list of top-10 items purchased by customers

top_categories = df['Category'].value_counts()
# print(top_categories)  #--This will give the list of topcategories in which items were purchased by the customers & total counts of items

plt.figure(figsize=(10,6))
plt.pie(top_categories, labels=top_categories.index, autopct ='%1.1f%%', colors=['Yellow', 'orange', 'red', 'green', 'purple'] )
plt.title("Top selling categories")
# plt.show(block=True)  #--This will create a pie chart for top selling categories

category_items = df.groupby('Category')['Item Purchased'].value_counts().groupby(level=0).head(3)
# print(category_items) #--This gives the list of top 3 item purchased in each category and its count.

location_sales = df['Location'].value_counts()
# print(location_sales)  #--This returns purchase counts by location and gives the list for all cities with count of purchases from that city

#-Identify the Most Profitable Customer Locations
location_revenue = df.groupby('Location')['Purchase Amount (USD)'].sum().reset_index()
top_locations = location_revenue.sort_values(by='Purchase Amount (USD)', ascending=False).head(10)
top_locations.index = range(1, 11)
top_locations["State Code"] = top_locations["Location"].apply(lambda x: us.states.lookup(x).abbr if us.states.lookup(x) else None)
top_locations = top_locations.dropna(subset=["State Code"])
df["State Code"] = df["Location"].apply(lambda x: us.states.lookup(x).abbr if us.states.lookup(x) else None)
# print(top_locations)
# print(top_locations.head(10))

low_revenue_regions = df.groupby("State Code")["Purchase Amount (USD)"].sum().reset_index()
low_revenue_regions = low_revenue_regions.sort_values(by="Purchase Amount (USD)", ascending=True)
# print(low_revenue_regions.head(5))

total_revenue = df["Purchase Amount (USD)"].sum()
total_customers = df["Customer ID"].nunique()
avg_purchase_per_customer = total_revenue / total_customers
# print(avg_purchase_per_customer)


#Plotting a map to show top 10 locations with most purchase amount

fig = px.choropleth(
    top_locations,
    locations="State Code",
    locationmode="USA-states",
    color="Purchase Amount (USD)",  # Determines the shade of green
    color_continuous_scale="Greens",  # Shades of green
    scope="usa",
    title="Top 10 Locations by Revenue"
)

fig.add_trace(go.Scattergeo(
    locations=top_locations["State Code"],
    locationmode="USA-states",
    text=top_locations["State Code"],  # Show state codes
    mode="text",
    showlegend=False
))

fig.write_html("usa_map_colored_states.html")

print("Map saved as usa_map_colored_states.html. Open it manually in a browser.")

df.to_csv("cleaned_shopping_trends.csv", index=False)


































