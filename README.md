# sys6018-competition-safe-driver

# Team Member
Teresa Sun  
Huitong(Jo) Pan   --- hp4zw        
Sai  

# Write up
Who might care about it and why?       

Why might it be a challenging problem to address?       
What other problems resemble this problem?       

# Regression Model (Teresa)
Since we have such a large number of observations, instead of using all the observations, we randomly chose 10,000 observations from train data to select variables and build the model. 

Using stepwise regression, we were able to choose (ps_ind_15 + ps_ind_17_bin + ps_reg_03 + ps_car_01_cat + ps_car_09_cat + ps_car_11 + ps_car_12 + ps_car_13 + ps_car_14 + ps_car_15 + ps_calc_02 + ps_calc_09 + ps_ind_09_bin) as parameters to build logistic regression model.


# Random Forest
