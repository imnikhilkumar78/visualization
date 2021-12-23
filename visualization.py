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



# def convert_date(date):
#     converted_date=datetime.strftime(datetime.strptime(date, "%d/%m/%Y"), "%m/%d/%Y")
#     return(converted_date)

def get_balance_in_time(data):
    df=data
    #for i in df.index:
       #df["Transaction Date"][i] = convert_date(df["Transaction Date"][i]) 
    df["Transaction Date"] = pd.to_datetime(df["Transaction Date"])
    test = pd.DataFrame(columns=["Transaction Date", "day", "month", "year", "Balance"], index=df.index)
    for i in df.index:
        test["Transaction Date"][i] = df["Transaction Date"][i]
        test["day"][i] = df["Transaction Date"][i].day
        test["month"][i] = df["Transaction Date"][i].month
        test["year"][i] = df["Transaction Date"][i].year
        test["Balance"][i] = df["Balance"][i]
    bal = pd.DataFrame(columns=["day", "month", "year", "week", "Balance"])
    
    for i in range(len(test.index) - 1):
        if test["day"][i] != test["day"][i + 1]:
            rng = pd.date_range(test["Transaction Date"][i], test["Transaction Date"][i + 1])
            t = pd.DataFrame(columns=bal.columns, index=rng)
            for j in rng:
                t["day"][j] = j.day
                t["month"][j] = j.month
                t["year"][j] = j.year
                t["week"][j] = j.week
                t["Balance"][j] = test["Balance"][i]
            
            bal = pd.concat([bal, t], axis=0)
    bal = bal[~bal.index.duplicated(keep='first')]
    return(bal)
    
def calculate_balance_avg(data):
    
    bal = get_balance_in_time(data)
    
    weekly_volume = bal.groupby("week").sum().Balance
    weekly_volume_avg = sum(weekly_volume) // len(weekly_volume)
    #weekly_volume_avg_text="Weekly Volume Agerage"+str(weekly_volume_avg)
    
    monthly_volume = bal.groupby("month").sum().Balance
    monthly_volume_avg = sum(monthly_volume) // len(monthly_volume)
    #monthly_volume_avg_text="Monthly Volume Average"+str(monthly_volume_avg)
    #print(monthly_volume)
    
    daily_avg = sum(bal.Balance) // len(bal)
    daily_volume = bal.groupby("day").sum().Balance
    #print(daily_volume)
    
    #plt.figure(figsize=(16, 10))
    plt.bar(weekly_volume.index, weekly_volume)
    #plt.text(-2,-1.5, weekly_volume_avg_text, {'color': 'C0', 'fontsize': 13})
    plt.show()
    
    #plt.figure(figsize=(16, 10))
    plt.bar(monthly_volume.index, monthly_volume)
    #plt.text(-2,-1.5, monthly_volume_avg_text, {'color': 'C0', 'fontsize': 13})
    plt.show() 
    
    #plt.figure(figsize=(16, 10))
    plt.plot(daily_volume.index, daily_volume)
    plt.show()
            

  
def categorise_trans(data): 
    df=data
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
        
    labels = {"Bank-Transfer": "Bank-Transfer", "rrn": "shopping", "loan": "loan", "emi": "loan", "amazon": "shopping", "flipkart": "shopping",
                  "mutualfund": "invest", "txn paytm": "shopping", "restaurant": "food", "paytm": "paytm",
                  "atd": "atm", "atm": "atm", "net txn": "Bank-Tranfer", "cash": "cash", "funds trf": "Transfers", "neft": "Bank-Transfer",
                  "interest": "interest",
                  "metro": "travel", "ola":"travel", "uber":"travel", 
                  "swiggy": "food", "faasos": "food", "zomato": "food", "upi": "shopping", "ola": "travel",
                  "refund": "refund",
                  "charge": "bank_charges", "pca": "Bank-Transfer", "loan": "loan", "credit":"card"}
    
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
    plt.bar(sums.index,sums["Debit"])
    plt.show()
    
    
    plt.figure(figsize=(16, 10))
    plt.pie(sums["Debit"], labels=sums.index, autopct='%1.2f%%', shadow=True)
    plt.show()

def diff_credit_debit(data):
    df1=data["Credit"]
    sum_credit = np.sum(df1)
    
    df2=data["Debit"]
    sum_debit = np.sum(df2)
    
    debt=np.round(np.subtract(sum_debit,sum_credit),2)
    savings=np.abs(debt)
    
    inflow_labels=["Credit", "Debit"]
    total_amt=[sum_credit,sum_debit]
    debt_text=r"You are in debt of "+str(debt)
    savings_text=r"You have saved overall "+str(savings)+" in savings."
    
    #plt.figure(figsize=(16, 10))
    plt.pie(total_amt, labels=inflow_labels, autopct='%1.2f%%', shadow=True)
    if(debt>0):
        plt.text(-2,-1.5, debt_text, {'color': 'C0', 'fontsize': 13})
    else:
        plt.text(-2,-1.5, savings_text, {'color': 'C0', 'fontsize': 13})
    plt.show()
    
def Prediction(data):
    bal = get_balance_in_time(data)
    bal['balance1'] = bal['Balance'].groupby(bal["month"]).transform('sum')
    #print(bal)
    monthly=bal[["month","balance1"]]
    #print(monthly)
    
    monthly.drop_duplicates(keep = 'first', inplace = True)
    print(monthly)
    print(monthly.astype(float).corr(method="kendall").abs())
    
    X = monthly.iloc[ :, :-1].values
    y = monthly.iloc[ :, 1].values
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
    
    #training model
    from sklearn.linear_model import LinearRegression
    regressor1 = LinearRegression()
    regressor1.fit(X_train, y_train)
    print('Model trained successfully')
    
    #Plotting the test data using the previously trained test data.
    line = regressor1.coef_*X+regressor1.intercept_  
    plt.scatter(X, y)
    plt.plot(X, line)
    plt.show()
    
    print(X_test)
    y_pred = regressor1.predict(X_test)
    predict = pd.DataFrame({'Months': [i[0] for i in X_test], 'Predicted Expense': [k for k in y_pred]})
    print(predict)
    
    #Comparing the actual values vs predicted values
    
    compare = pd.DataFrame({'Actual_Expense': y_test, 'Predicted_Expense': y_pred})
    print(compare)
    
    compare.plot.bar(rot =15,title="Actual Expense v/s Predicted Expense", figsize=(10,6));
    plt.show();
    
    #predicting expense of user
    
    month = int(input("Enter month for which you want to predict expenses"))
    test = np.array([month])
    test = test.reshape(-1,1)
    predicted_score= regressor1.predict(test)
    #print(predicted_score)
    print('Month = {}'.format(month))
    print('Predicted Expense = {}'.format(round(predicted_score[0],3)))
    
    
    
    
    
    
    
    
    




    
