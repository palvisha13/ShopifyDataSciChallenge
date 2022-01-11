## Shopify Data Science Internship Challenge Summer 2022



### Question 1
My approach to this question was to perform an exporatory data analysis on the dataset provided to better understand its features and the issue with the AOV calculation given in the question. My work was done on a Jupyter Notebook on Google Colab: [Shopify 2022 Summer Challenge](https://colab.research.google.com/drive/16BDvMPM5h5sTrEkixBL8F8hhzO_wKS7M?usp=sharing).


 **EDA**

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
# maximum value in the data set 
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

This is much better! Now, I have a histogram of the data in the order_ammount column that is readable. I did not normalize this histogram since the y axis was scaled logarithmically, which would make the y axis less intuitive to understand if the histogram was normalized before the yaxis was scaled. The shape of the expected histogram is preserved.




Looking at this histogram, the data is skewed to the right since it is an asymmetric distribution with a right ended tail. This means that the mean is higher than the median and mode values, confirming that it is not a good metric to describe central tendency.




The histogram also shows a skew in the distribution of continuous data rather than categorical, as such I will opt to report the median of the `order_amount`, rather than the mode. The median better represents continuous data that does not follow a normal distribution, and the mode better represents the distribution of categorical (non-continuous) data. Although, both can be reported for a continuous data set.



The median of the order amounts is $ 284.00 hence, I can expect the AOV and general cost of shoes from the shopify stores to be around $ 284.00.



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


 **a**. The AOV amount provided in the question is $3145.13. The AOV is calculated by taking the total revenue and dividing by the total number of orders. The reason that this  AOV calculation is much higher than the expected cost of shoes is because the average does not represent the central tendency of the data. As well, given that the highest order total is $704 000 (definitely closer to being the price of a house than a shoe), and because there are multiple orders of this total value, the AOV is highly skewed due to these data points. This maximum order amount greatly offsets the AOV from the cost of individual shoes.



Since the AOV is highly skewed by such data points, it is not an accurate representation of the general cost of shoes amongst the 100 stores. Due to this, it also does not provide much information about the stores and shoe purchases aside from the average amount of money spent on shoes over this particular 30 day period. For substantial information on the dataset and associated costs, rather than the AOV, the average or median cost (if there is a highly skewed distribution) of shoes must be reported.  



**b**. As per my analysis above, the best metric to report for a skewed continuous distribution will be the median.

**c**. The median determined in my EDA is $ 284.00, this median represents the central tendency of the data better than the mean, and is a good metric to report fofr the AOV.

_____


### Question 2

