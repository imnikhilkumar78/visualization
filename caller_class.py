# -*- coding: utf-8 -*-
"""
Created on Wed Dec 22 20:43:38 2021

@author: Administrator
"""

import pandas as pd
from visualization import *

df=pd.read_csv('yes1.csv')
categorise_trans(df)
calculate_balance_avg(df)
diff_credit_debit(df)
Prediction(df)


    
    