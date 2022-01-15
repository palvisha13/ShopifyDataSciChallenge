## Shopify Data Science Internship Challenge Summer 2022


## Answers: 
-----
## Question 1
My approach to this question was to perform an exporatory data analysis on the dataset provided to better understand its features and the issue with the AOV calculation given in the question. My work was done on a Jupyter Notebook on Google Colab: [Shopify 2022 Summer Challenge](https://colab.research.google.com/drive/16BDvMPM5h5sTrEkixBL8F8hhzO_wKS7M?usp=sharing).

 **a**. The AOV amount provided in the question is $3145.13. The AOV is calculated by taking the total revenue and dividing by the total number of orders. The reason that this  AOV calculation is much higher than the expected cost of shoes is because the average does not represent the central tendency of the data. As well, given that the highest order total is $704 000  and because there are multiple orders of this total value, the AOV is highly skewed due to these data points. This maximum order amount greatly offsets the AOV from the cost of individual shoes.

**b**. As per my analysis above, the best metric to report for a skewed continuous distribution will be the median.

**c**. The median determined in my EDA is $ 284.00, this median represents the central tendency of the data better than the mean, and is a good metric to report for the AOV.
            
------


## Question 2 


 

To access the data set, and determine the total number of orders, the initial query run was: 

```SQL
SELECT * FROM Orders;
```
Which selects all data from the table "Orders" in the data set. This table displays the order IDs, customer IDs, and order dates of orders shipped by Speedy Express.

**a.** There are 196 orders shipped by Speedy Express.

```SQL
SELECT COUNT(OrderID)
FROM Orders
```
This query selects and counts all OrderIDs.
More analysis and checks were preformed on this data, as shown below.

**b.** The employee with the most orders has the last name **Peacock**. 

```SQL
SELECT EmployeeID, COUNT(*) 
FROM Orders 
GROUP BY EmployeeID
ORDER BY COUNT(*) DESC
LIMIT 1
```
This returns the most frequent EmployeeID in the Orders table. It groups unique EmployeeIDs and counts how many of each there are, then orders the records by the 
most to least frequent EmployeeID in the Orders table.
The most frequent EmployeeID in the Orders column is returned as 4. 

To find the last name associated with this ID, the EmployeeID is the primary key for the Employee table, so I can use this 
key to find the last name of the employee. 

```SQL
SELECT LastName FROM Employees 
WHERE EmployeeID = 4
```
This query selects the last name from the employees table with an employee ID of 4.

The result says that the Last Name of the most frequent EmployeeID is Peacock.

**c.** The most common product ordered in Germany is Gorgonzola Telino.

```SQL
SELECT Customers.Country, Orders.OrderID, OrderDetails.ProductID, Products.ProductName, COUNT(*)
FROM Customers
INNER JOIN Orders ON Customers.CustomerID = Orders.CustomerID 
INNER JOIN OrderDetails ON OrderDetails.OrderID = Orders.OrderID
INNER JOIN Products ON OrderDetails.ProductID = Products.ProductID
WHERE Country= "Germany"
GROUP BY Products.ProductID
ORDER BY COUNT(*) DESC
LIMIT 1
```
With this query, I am joining all of the necessary tables together and returning the orderIDs, productIDs, and ProductNames for all orders placed in Germany. Then, I am grouping all of the unique products purchased in Germany by their Product IDs, and ordering the records by the most to least frequent Product ID. The most frequent product ID is returned, along with its name. The most frequent Product ID is associated with the Product: Gorgonzola Telino. 

--------



### Detailed Thought Process

----

 ## Question 1

```python

"""
This notebook preforms an analysis of the data on shoe orders 
from 100 Shopify stores.
The purpose is to explore any trends 
and distributions within the data.
--------------------------
Author: Palvisha Sharma 
"""
```
```python
# import libraries
# since I am working on google colab, lines 7 and 8 refer to the 
# drive being mounted so that I can work with the sample data 

# import libraries
# since I am working on google colab, lines 7 and 8 refer to the 
# drive being mounted so that I can work with the sample data 

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import statistics
import numpy as np
from google.colab import drive
from google.colab import files
drive.mount("/content/drive")
sns.set()

```
```python
# importing and printing the data header

df = pd.read_csv("/content/drive/My Drive/Colab Notebooks/data/ShopifyChallengeData.csv")
print(df.head())
```
```python
  order_id  shop_id  user_id  ...  total_items  payment_method           created_at
0         1       53      746  ...            2            cash  2017-03-13 12:36:56
1         2       92      925  ...            1            cash  2017-03-03 17:38:52
2         3       44      861  ...            1            cash   2017-03-14 4:23:56
3         4       18      935  ...            1     credit_card  2017-03-26 12:43:37
4         5       18      883  ...            1     credit_card   2017-03-01 4:35:11

[5 rows x 7 columns]
```

```python
# ensure that there are no null values in the data set
df.isnull().sum()

order_id          0
shop_id           0
user_id           0
order_amount      0
total_items       0
payment_method    0
created_at        0
dtype: int64
```
```python
# determining the columns in the data
df.columns

Index(['order_id', 'shop_id', 'user_id', 'order_amount', 'total_items',
       'payment_method', 'created_at'],
      dtype='object')
```
```python
# determining the shape of our data, this can be used to determine the 
# number of features and amount of data 
df.shape

(5000, 7)

# since the point of focus is the order amount,
# I will preform an analysis of the data below
print(df["order_amount"])

0       224
1        90
2       144
3       156
4       156
       ... 
4995    330
4996    234
4997    351
4998    354
4999    288
Name: order_amount, Length: 5000, dtype: int64

```
```python

df["order_amount"].max()

704000

# minimum value in the data set
df["order_amount"].min()

90

# mean of the order_amount 
df["order_amount"].mean()

3145.128

# median of the order_amount
df["order_amount"].median()

284.0

# mode of the order_amount
df["order_amount"].mode()

0    153
dtype: int64

# standard deviation of my data 
df["order_amount"].std()

41282.539348788196


```
Looking at the maxiumum and minimum amounts, the maximum amount is unreasonably high. Extreme data points such as this, skew the mean of the data, which is why the average is much higher than expected.

The standard deviation of my data is also very high, suggesting that most data points do not agree with the AOV. This confirms that the mean is not an appropriate metric for the AOV.

```python
# I can represent the data points relative to the median through a box plot
# to get a better understanding of the general data that I have, and any outliers

plt.figure(figsize=(20,10))
plt.yscale("log")
plt.boxplot(df["order_amount"])
plt.show()

```
![image of a boxplot of shopify shoe orders](docs/assets/boxplot.png)

I can see that my data is skewed, and does not follow a normal distribution since the median is not centered withing the boxplot. I can also see that the interquartile range is within an order of magnitude  10<sup>2</sup>, as are the maximum and minimum values. This suggests that the order amounts generally fall within the $ 100s range. The outliers are lie in the thousands. Therefore, a good order of magnitude estimate for the AOV would be 10<sup>2</sup>. 

I can analyze my data's distribution better with a histogram.

```python
# plot the distribution of order amounts 

plt.figure(figsize=(20,10))
plt.hist(df["order_amount"])
plt.title("Order Amount Histogram")
plt.xlabel("Order Amount")
plt.ylabel("Number of Orders")
plt.show()
```
![image of an incorrect histogram of shopify shoe data distribution](docs/assets/hist1.png)

My histogram here fails to show both the smaller and larger values due to extreme difference in costs, with the maximum amount being close to 700 000, compared to the minimum amount of 90. 

```python
# another attempt at plotting my histogram by scaling with logarithmic scaling.

plt.figure(figsize=(20,10))
plt.yscale("log")
plt.hist(df["order_amount"])
plt.title("Order Amount Histogram")
plt.xlabel("Order Amount")
plt.ylabel("Frequency")
plt.show()

```
![image of a correct histogram of shoe data distribution](docs/assets/hist2.png)

This is much better! Now, I have a histogram of the data in the order_ammount column that is readable. I did not normalize this histogram since the y axis was scaled logarithmically, which would make the y axis less intuitive to understand if the histogram was normalized before the yaxis was scaled. The shape of the expected histogram is preserved. The data is skewed to the right since it is an asymmetric distribution with a right ended tail, confirming that the mean should not be used.

Since this represents a skew in continuous data (not categorical), I will report the median of the `order_amount`, rather than the mode.The median better represents continuous data that does not follow a normal distribution, and the mode better represents the distribution of categorical (non-continuous) data. 

The median of the order amounts is $ 284.00 hence, I can expect the AOV from the shopify stores to be around $ 284.00.



Looking at my EDA above, it is evident that the mistake made in calculating the AOV by taking the average of values was that the distribution of the data was not considered. The mean best represents data that follows a normal distrbution as it is an appropriate measure of central tendency in normal distrbutions. However, the data for the order amounts did not follow a normal distribution, and a better measure of central tendency was determined to be the median. 

I decided to check one more thing with the data and calculate the average cost of each shoe below.

```python
# average cost of each shoe
# the average cost of each shoe is the total amount 
#of money spent on shoes across all orders
# divided by the total number of shoes purchased across all orders 

avg_shoe_cost = np.sum(df["order_amount"]/ np.sum(df["total_items"]))
print(avg_shoe_cost)

357.9215222141296
```
------


## Question 2 

The OrderID is the primary key in this data set, there are no duplicate values for a primary key, and I can confirm that the orderIDs are all unique by checking for duplicate values through this query: 

It groups and counts orderIDs within a group. Any group with more than a single entry represents a duplicate OrderID.

```SQL
SELECT OrderID, COUNT(*)
FROM Orders
GROUP BY OrderID
HAVING COUNT(*) > 1
```
There are no duplicates found.

I also want to make sure that there are no null records within the data set (non zero and non integer). Any null records means that a provided order record is incomplete.

Any records with null values in the data were determined by the query below:

```SQL
SELECT OrderID
FROM Orders
WHERE CustomerID IS NULL
or EmployeeID IS NULL
or OrderDate IS NULL
or ShipperID IS NULL;
```
Here, all OrderIDs are returned if one of the values in their records is null.
The output of this query confirmed that there are no null values in the data set. All records are complete.

Although the total number of records (which are confirmed as being unique) are already reported above the table, I can confirm the total number of orders by 
finding the sum of the number of orderIDs. The IDs are unique so each ID represents a single order, hence the sum of the number of OrderIDs will give the number of
individual orders shipped by Speedy Express. 

**NOTE:** I do not want to add the OrderID number together, I want to count the total number of OrderIDs, hence

I can find this total through this query: 


To get the last name of the employee with the most orders, I still need to access the information on inividual and unique orders, this table also includes EmployeeIDs. Each EmployeeID would refer to a single employee. 
Calling the query below for employee information confirms that the EmployeeID is a primary key for the Employee Information, and is therefore, unique to each employee. 

```SQL
SELECT * FROM Employees
```

Back to determining the Employee with the maximum number of orders. 
There are two parts to this. 
First, I need to determine, in the Order information, which EmployeeID shows up the most often. 
And then I need to use this EmployeeID to find the last name of the Employee it refers to in the Employee table. 

The most efficient way is to combine both steps. Rather than look through two different tables (Orders and Employees) separately, I can find the most frequent EmployeeID that shows up in my Orders table, and compare it to the Employee information table to find its corresponding last name.


To do this, I ran the following query first : 
```SQL
SELECT * FROM Orders 
```
To first accquire the entire table. 
Then, I ran the query: 

```SQL
SELECT EmployeeID, COUNT(*)
FROM Orders
GROUP BY EmployeeID
```
This query is similary to a query above. It groups the same EmployeeIDs together, and counts how many are in each group. Then, in a column beside each unique employeeID, it returns the frequency of that ID.
From here, the following query can be ran to return the most frequent of the EmployeeIDs. 

First, I can sort my data from the most to least frequent -> descending order. 
Then, since I need the most frequent, I just need the first/top row.

To put the data into descening order by the calculated frequency, I just add on this query: 

```SQL
ORDER BY COUNT(*) DESC
```
Then, to get the top value, I can always read it off by the results of my above query, but to do it completely through SQL, 
I will select just the top row now by adding the query to the bottom of everything above: 

```SQL
LIMIT 1
```
Combining all of these, my query will look like the at the end: 

```SQL
SELECT EmployeeID, COUNT(*) 
FROM Orders 
GROUP BY EmployeeID
ORDER BY COUNT(*) DESC
LIMIT 1
```
There I have it, the most frequent EmployeeID is "4". 
Now, I need to find the last name associated with this ID. The EmployeeID is the primary key for the Employee table, so I can use this 
key to find the last name of the employee. 

```SQL
SELECT LastName FROM Employees 
WHERE EmployeeID = 4
```
This query selects the last name from the employees table with an employee ID of 4.

The result says that the Last Name of the most frequent EmployeeID is Peacock. 

An alternative way to do this would have been to use SQL JOINS but this is still straightforward.

Now, to find the most common prouct ordered from Germany,  I need to find all of the customers from Germany and match them with their Orders. 

There are 4 tables I need to consider: 
`Products` 
`Orders`
`OrderDetails`
`Customers` 
To find the most purchased product from customers in Germany, I need to first inner join all of those tables so I can connect them based on their common columns. 
The  Customers table has Customer IDs and Customer Countries, the Orders have Customer IDs, and Order IDs, the OrderDetails have Order IDs and Product IDs, and the
Products table has  Product IDs and Product Names.

```SQL
SELECT Customers.Country, Orders.OrderID, OrderDetails.ProductID, Products.ProductName
FROM Customers
INNER JOIN Orders ON Customers.CustomerID = Orders.CustomerID 
INNER JOIN OrderDetails ON OrderDetails.OrderID = Orders.OrderID
INNER JOIN Products ON OrderDetails.ProductID = Products.ProductID
```
This returns joins all of the important tables and returns the information I need in order to find the most common purchase from Germany (ie. Product names, and Country). 

Now, I need to run a query to filter for just the records with Country = Germany. 
So, I add this line to my above query: 

`WHERE Country= "Germany"`

Now that I have all of the product IDs for German orders, I need to group by the Prouct IDs, count the number of each ProductID, and return the highest value. 
I added the following lines to the end of my SQL query above: 

```SQL
GROUP BY Products.ProductID
ORDER BY COUNT(*) DESC
LIMIT 1
```

and added the following to the end of my selection.

```SQL
,COUNT(*)
```
My SQL query became: 

```SQL
SELECT Customers.Country, Orders.OrderID, OrderDetails.ProductID, Products.ProductName, COUNT(*)
FROM Customers
INNER JOIN Orders ON Customers.CustomerID = Orders.CustomerID 
INNER JOIN OrderDetails ON OrderDetails.OrderID = Orders.OrderID
INNER JOIN Products ON OrderDetails.ProductID = Products.ProductID
WHERE Country= "Germany"
GROUP BY Products.ProductID
ORDER BY COUNT(*) DESC
LIMIT 1
```
So, I am essentially returning the selected columns and the count of the Product IDs after joining the three tables I require, and filtering based on the country.
The LIMIT 1 and DESC sort the records by the most to least frequent ProductIDs and the LIMIT 1 returns the top of that descending list- ie. the record with the most frequent product ID. The most frequent Product ID is associated with the Product: Gorgonzola Telino

**c** The most common product ordered in Germany is Gorgonzola Telino.

----
