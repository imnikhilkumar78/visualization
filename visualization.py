# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 17:36:18 2021

@author: Administrator
"""

import pandas as pd                       #to perform data manipulation and analysis
import numpy as np                        #to cleanse data
from datetime import datetime             #to manipulate dates
import re
import matplotlib.pyplot as plt

df=pd.read_csv("yes1.csv")
print(df[6:10])


#
t = df["Description"]
t = t.apply(lambda x: x.lower())

# Removing numbers and special characters
text = t.replace(to_replace="[0-9]", value="", regex=True).apply(lambda x: x.replace("/", "").replace("\\", "").replace(":", "").replace("\n", " ").replace("-", " ").replace("/", " "))

# Removing extra spaces created due to the above step
for i in range(len(text)):
    x = text[i].split()
    for j in range(len(x)):
        x[j] = x[j].strip()
    text[i] = " ".join(x) 
    
labels = {"imps": "imps", "rrn": "imps", "loan": "loan", "emi": "emi", "amazon": "shopping", "flipkart": "shopping",
              "mutualfund": "invest", "txn paytm": "trf", "restaurant": "food", "paytm": "trf",
              "atd": "atm", "atm": "atm", "net txn": "nettxn", "cash": "cash", "funds trf": "trf", "neft": "neft",
              "interest": "interest",
              "metro": "travel", "swiggy": "food", "faasos": "food", "zomato": "food", "upi": "trf", "ola": "travel",
              "refund": "refund",
              "charge": "bank_charges", "pca": "trf", "loan": "loan", "credit":"card"}

labs = []

# Labelling the transaction according to the dictionary defined

for i in text:
        f = 0
        for j in list(labels.keys()):
            if j in i:
                labs.append(labels[j])
                f = 1
                break
        if f == 0:
            labs.append("miscellaneous")
df["Label"] = pd.DataFrame(labs)
x = df.Description.apply(lambda x: re.findall(r'[\w\.-]+@[\w\.-]+', x))
df["Remark"] = pd.DataFrame(x)

labels = df["Label"].unique()
counts = df.groupby("Label").size()

sums = df.groupby("Label").sum()

plt.figure(figsize=(16, 10))
plt.bar(counts.index, counts)
plt.show()

plt.figure(figsize=(16, 10))
plt.bar(sums.index,sums["Debit"])
plt.show()

plt.figure(figsize=(16, 10))
plt.pie(counts, labels=counts.index)
plt.show()

plt.figure(figsize=(16, 10))
plt.pie(sums["Debit"], labels=sums.index)
plt.show()




    
