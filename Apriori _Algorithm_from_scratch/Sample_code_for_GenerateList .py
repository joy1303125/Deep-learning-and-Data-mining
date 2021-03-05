# -*- coding: utf-8 -*-
"""
Created on Sun Sep  6 19:23:52 2020

@author: Muyeed Ahmed
"""

import random
from random import randrange

items = ["Bread", "Butter", "Cold Drinks", "Chips", "Milk", "Nutella", "Peanut Butter", "Cereal", "Soya Sauce", "Oil", "Salt", "Suger", "Coffee", 
         "Chocolate", "Sauce", "Yogurt", "Cheese", "Egg", "Oregano", "Paprika", "Fish", "Beef", "Chicken", "Honey", "Onion", "Garlic", "Cookies", "Jam", "Flour", "Tea"]

for file in range(5):
    fileName = "InputFile" + str(file+1) + ".csv"
    f = open(fileName, "a")
    transactions = ""
    for i1 in range(20):
        itemN = random.randint(5,15)
        r = random.sample(range(0, 30), itemN)
        for i2 in range(itemN):
            if i2 == itemN-1:
                print(items[r[i2]])
                transactions += items[r[i2]] + "\n"
                break
            print(items[r[i2]], end=", ")
            transactions += items[r[i2]] + ", "
    f.write(transactions)
    f.close()

    



'''
items = ["Bread", "Butter", "Pepsi", "Chips", "Pretzels", "Nutella", "Peanut Butter", "Corn Flacks", "Soap", "Oil"]

for i1 in range(20):
    itemN = random.randint(2,7)
    r = random.sample(range(1, 10), itemN)
    for i2 in range(itemN):
        if i2 == itemN-1:
            print(items[r[i2]])
            break
        print(items[r[i2]], end=", ")
        
        '''