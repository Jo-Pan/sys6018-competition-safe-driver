# sys6018-competition-safe-driver

# Team Member
|    Name         | Computing ID  |    Role       |
| -------------   | ------------- | ------------- |
| Teresa Sun      |  js6sm        | lm,step-wise            | 
| Huitong(Jo) Pan |  hp4zw        | data cleaning, boosting, random forest| 
| Sai             |  lbs7aa       | decision tree, bagging, random forest | 

# Models Summary
|    Model Name   | Data used to train   |    Parameter     | Gini kaggle score  |
| -------------   | -------------------  | -------------    |------------------- |
| Linear Model    |  balanced data       |   ?? var         |                    | 
| Random Forest   |  balanced data       | best:mtry=7      |    0.245           | 

# Write up
**Who might care about it and why?**       
Insurance Companies will care about it because identifying risky customers can reduce their cost.

**Why might it be a challenging problem to address?**          
1) We don't have any description on the variables and we have no idea what do they mean.
2) We are trying to use past data to inform future claim patterns. 
3) Insurance is a case-by-case problem. If we want to use data to predict, it may cause problem of high false positive rate.
4) Insurance data is highly biased toward non-claim target. Predicting with a biased data can be problemetic.

**What other problems resemble this problem?**       
1) Any other insurance problem      
2) Cancer prediction problem in terms of data bias. 

# Data cleaning 
We dropped a couple of columns that contained too many NA's and we imputed the NA values with mean for all the other columns for rows which had NA's. We also factorized the relavent columns and had to remove a column out of predictors since it had 104 levels despite being a factor variable (R doesn't allow beyond 32 levels for factor variables).

# Regression Model (Teresa)
Since we have such a large number of observations, instead of using all the observations, we randomly chose 10,000 observations from train data to select variables and build the model. 

Using stepwise regression, we were able to choose (ps_ind_15 + ps_ind_17_bin + ps_reg_03 + ps_car_01_cat + ps_car_09_cat + ps_car_11 + ps_car_12 + ps_car_13 + ps_car_14 + ps_car_15 + ps_calc_02 + ps_calc_09 + ps_ind_09_bin) as parameters to build logistic regression model.

# Tree based methods (Sai, Jo)

Based on Decision trees and bagging approach, it was clear to us that class imbalance was going to be a critical issue to tackle in this data set. We experimented with both randomforest and caret packages. For random forests, with cross validation approach, first we tried to identify the ideal number of trees to grow each tree with (Due to computing power constraints, we decided to set mtry to 7, close to square root of 55).

To figure out the ideal balance ratio of data set, we ran a for loop and realized that the gini ratios are best for data sets with a class ratio of close to 1:1. Hence, we trained a random forest model (1000 trees) with roughly 43000 rows (equal number of 1's and 0's) and applied this model on the provided testing set and obtained a gini score of 0.245.

# K-NN, Boosting

We have also tried K-NN and Bossting approaches in addition to above and couldn't secure a higher gini score than random forests and linear regression.
