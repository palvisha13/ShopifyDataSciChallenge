## Shopify Data Science Internship Challenge Summer 2022

This is my submission with my analysis included, for the Shopify data science challenge.

### Question 1
My approach to this question was to first perform exporatory data analysis on the dataset provided to better understand its features and the issue with the AOV calculation provided in the question. My work was done on a Jupyter Notebook on Google Colab: [Shopify 2022 Summer Challenge](https://colab.research.google.com/drive/16BDvMPM5h5sTrEkixBL8F8hhzO_wKS7M?usp=sharing).

```python

"""
This notebook preforms an analysis of the data on shoe orders from 100 Shopify stores.
The purpose is to explore any trends and distributions within the data.
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

  order_id  shop_id  user_id  ...  total_items  payment_method           created_at
0         1       53      746  ...            2            cash  2017-03-13 12:36:56
1         2       92      925  ...            1            cash  2017-03-03 17:38:52
2         3       44      861  ...            1            cash   2017-03-14 4:23:56
3         4       18      935  ...            1     credit_card  2017-03-26 12:43:37
4         5       18      883  ...            1     credit_card   2017-03-01 4:35:11

[5 rows x 7 columns]
```

a. The AOV amount provided in the question is $3145.13. The AOV is calculated by taking the total revenue and dividing by the total number of orders. The reason that this  AOV calculation is much higher than the expected cost of shoes is because the average does not represent the central tendency of the data. As well, given that the highest order total is $704 000 (definitely closer to being the price of a house than a shoe), and because there are multiple orders of this total value, the AOV is highly skewed due to these data points. This maximum order amount greatly offsets the AOV from the cost of individual shoes.

Since the AOV is highly skewed by such data points, it is not an accurate representation of the general cost of shoes amongst the 100 stores. Due to this, it also does not provide much information about the stores and shoe purchases aside from the average amount of money spent on shoes over this particular 30 day period. For substantial information on the dataset and associated costs, rather than the AOV, the average or median cost (if there is a highly skewed distribution) of shoes must be reported.  

## EDA


```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [Basic writing and formatting syntax](https://docs.github.com/en/github/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/palvisha13/ShopifyDataSciChallenge/settings/pages). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://docs.github.com/categories/github-pages-basics/) or [contact support](https://support.github.com/contact) and we’ll help you sort it out.
