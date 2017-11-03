# sys6018-competition-safe-driver

# Team Member
|    Name         | Computing ID  |    Role       |
| -------------   | ------------- | ------------- |
| Teresa Sun      |  js6sm        | lm,step-wise            | 
| Huitong(Jo) Pan |  hp4zw        | data cleaning, boosting, random forest| 
| Sai             |               | decision tree, random forest | 

# Models Summary
|    Model Name   | Data used to train  |   Parameter     | Gini | Kaggle Score  |
| -------------   | ------------- | ------------- |------------- |------------- |
| Linear Model    |unbalanced data |   ?? var     |              |              | 
| Linear Model    |  balanced data |   ?? var     |              |              | 
| Random Forest   |unbalanced data(sampled 10K) | best:mtry=2     |    0.1673073   |              | 
| Random Forest   |  balanced data(sampled 10K) | best:mtry=7     |    0.3834081   |              | 
| Random Forest   |  balanced data(51K) | best:mtry=7     |    0.9997988   | 0.219         | 
| Boosting   |unbalanced data(sampled 10K) | best:ntree=100,in.depth=5,shrinkage=0.01   |    0.2027495   |              | 
| Boosting   |unbalanced data(sampled 10K) | best:ntree=2000,in.depth=5,shrinkage=0.001   |   0.2143287   |              | 
| Boosting   |unbalanced data(sampled 10K) | best:ntree=5000,in.depth=5,shrinkage=0.001   |   0.2182747   |              | 
| Boosting   |  balanced data(sampled 10K) | best:ntree=5000,in.depth=5,shrinkage=0.001   |   0.2692118   |              | 
| Boosting   |  balanced data(51K)) | ntree=5000,in.depth=5,shrinkage=0.001   |        |        0.219       | 

# Write up
Who might care about it and why?       
Insurance Companies will care about it because identifying risky customers can reduce their cost.

Why might it be a challenging problem to address?  
1) We don't have any description on the variables and we have no idea what do they mean.
2) We are trying to use past data to inform future claim patterns. 
3) Insurance is a case-by-case problem. If we want to use data to predict, it may cause problem of high false positive rate.
4) Insurance data is highly biased toward non-claim target. Predicting with a biased data can be problemetic.

What other problems resemble this problem?       
1) Any other insurance problem      
2) Cancer prediction problem in terms of data bias. 

# Data cleaning 
We imputed the NA values with mean.

# Regression Model (Teresa)
Since we have such a large number of observations, instead of using all the observations, we randomly chose 10,000 observations from train data to select variables and build the model. 

Using stepwise regression, we were able to choose (ps_ind_15 + ps_ind_17_bin + ps_reg_03 + ps_car_01_cat + ps_car_09_cat + ps_car_11 + ps_car_12 + ps_car_13 + ps_car_14 + ps_car_15 + ps_calc_02 + ps_calc_09 + ps_ind_09_bin) as parameters to build logistic regression model.


、
