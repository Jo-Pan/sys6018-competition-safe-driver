# sys6018-competition-safe-driver

# Team Member
|    Name         | Computing ID  |    Role       |
| -------------   | ------------- | ------------- |
| Teresa Sun      |  js6sm        | lm,step-wise            | 
| Huitong(Jo) Pan |  hp4zw        | data cleaning, boosting, random forest| 
| Sai             |  lbs7aa       | decision tree, bagging, random forest | 

# Models Summary
|    Model Name   | Data used to train   | Gini kaggle score  |
| -------------   | -------------------  |------------------- |
| Linear Model    |  balanced data       |    0.247           | 
| Random Forest   |  balanced data       |    0.245           | 

# Write up
**Who might care about it and why?**       
Insurance Companies generally will care about it because correctly identifying risky customers can reduce their cost. Also customers who are seeking insurance plan will care about the problem because It doesn’t seem fair that you have to pay so much if you’ve been cautious on the road for years. These two sides are both hoping this model could save them some costs.

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
Since we have such a large number of observations and have class imbalance issue, instead of using all the observations, we
first balance the data with a class ratio of close to 1:1 (roughly with 30,000 to 30,000 observations). Since the data is too large to build the model, we randomly chose 10,000 from the balanced data to build the model.

Using stepwise regression, we were able to choose ps_ind_01 + ps_ind_02_cat + ps_ind_03 + ps_ind_04_cat + ps_ind_05_cat + ps_ind_07_bin + ps_ind_08_bin + ps_ind_13_bin + ps_ind_15 + ps_ind_17_bin + ps_reg_01 + ps_reg_02 + ps_car_01_cat + ps_car_04_cat + ps_car_07_cat + ps_car_08_cat + ps_car_13 + ps_car_14 + ps_calc_18_bin + ps_calc_19_bin as parameters to build logistic regression model. The gini on kaggle for this linear regression model is 0.247.

# Tree based methods (Sai, Jo)

Based on Decision trees and bagging approach, it was clear to us that class imbalance was going to be a critical issue to tackle in this data set. We experimented with both randomforest and caret packages. For random forests, with cross validation approach, first we tried to identify the ideal number of trees to grow each tree with (Due to computing power constraints, we decided to set mtry to 7, close to square root of 55).

To figure out the ideal balance ratio of data set, we ran a for loop and realized that the gini ratios are best for data sets with a class ratio of close to 1:1. Hence, we trained a random forest model (1000 trees) with roughly 43000 rows (equal number of 1's and 0's) and applied this model on the provided testing set and obtained a gini score of 0.245.

# K-NN, Boosting

We have also tried K-NN and Bossting approaches in addition to above and couldn't secure a higher gini score than random forests and linear regression.
