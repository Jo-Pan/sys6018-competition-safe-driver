#competition 4 safe driver
#https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/data
#https://github.com/Jo-Pan/sys6018-competition-safe-driver
setwd("~/Desktop")
library(caret) #data cleaning, knn
library(class) #knn
library(dplyr) #sample_n for sampling rows
library(MASS)

train<-read.csv("train.csv") #892816,59
test<-read.csv("test.csv") #595212,59

#Combine train and test files
test$target <- NA
comb <- rbind(train, test)

################  Data Cleaning  ################## (Jo)
#columns with 20+% nas 
which(colSums(comb==-1) > .2*nrow(comb))
  #ps_car_03_cat, ps_car_05_cat 

#drop the col with too many na
comb<-comb[ , -which(names(comb) %in% c("ps_car_03_cat", "ps_car_05_cat" ))]

#impute na with median
comb[comb==-1]<-NA
for(i in 3:ncol(comb)){
  comb[is.na(comb[,i]), i] <- median(comb[,i], na.rm = TRUE)}

#collect names of all categorical variables 
cat_vars <- names(comb)[grepl('_cat$', names(comb))]

#convert cat_vars to factor
comb[cat_vars]<-lapply(comb[cat_vars],factor)

#split train & test sets
train<-comb[is.na(comb$target)==FALSE,]
test<-comb[is.na(comb$target)==TRUE,]

all1<-train[train$target==1,]
################  Gini Index  ##################
#' Calculates unnormalized Gini index from ground truth and predicted probabilities.
#' param ground.truth Ground-truth scalar values (e.g., 0 and 1)
#' param predicted.probabilities Predicted probabilities for the items listed in ground.truth
#' return Unnormalized Gini index.
unnormalized.gini.index = function(ground.truth, predicted.probabilities) {
  
  if (length(ground.truth) !=  length(predicted.probabilities))
  {
    stop("Actual and Predicted need to be equal lengths!")
  }
  
  # arrange data into table with columns of index, predicted values, and actual values
  gini.table = data.frame(index = c(1:length(ground.truth)), predicted.probabilities, ground.truth)
  
  # sort rows in decreasing order of the predicted values, breaking ties according to the index
  gini.table = gini.table[order(-gini.table$predicted.probabilities, gini.table$index), ]
  
  # get the per-row increment for positives accumulated by the model 
  num.ground.truth.positivies = sum(gini.table$ground.truth)
  model.percentage.positives.accumulated = gini.table$ground.truth / num.ground.truth.positivies
  
  # get the per-row increment for positives accumulated by a random guess
  random.guess.percentage.positives.accumulated = 1 / nrow(gini.table)
  
  # calculate gini index
  gini.sum = cumsum(model.percentage.positives.accumulated - random.guess.percentage.positives.accumulated)
  gini.index = sum(gini.sum) / nrow(gini.table) 
  return(gini.index)
}

#' Calculates normalized Gini index from ground truth and predicted probabilities.
#' param ground.truth Ground-truth scalar values (e.g., 0 and 1)
#' param predicted.probabilities Predicted probabilities for the items listed in ground.truth
#' return Normalized Gini index, accounting for theoretical optimal.
normalized.gini.index = function(ground.truth, predicted.probabilities) {
  
  model.gini.index = unnormalized.gini.index(ground.truth, predicted.probabilities)
  optimal.gini.index = unnormalized.gini.index(ground.truth, ground.truth)
  return(model.gini.index / optimal.gini.index)
}

################  Linear Model  ################## (Teresa)
# -------- separate train for cross validation ------ #
# originally intend to separate data into training set and validation set
# But the number of obervations is too huge

set.seed(1)
sub <- sample(1:nrow(train),size=nrow(train)*0.7)
data.train <- train[sub,]     # Select subset for cross-validation
data.valid <- train[-sub,]

# ---------- model building --------- #
# since observations are too big, random sample 10,000 from it to build the model
random.train1 <- sample_n(data.train,10000) # randomly choose 10,000 observations from training set

# ------- using logistic regression ------- #
glm1 <- glm(as.factor(target)~.-id, data=random.train1, family = binomial(link = 'logit'))

# stepwise regression first run
step <- stepAIC(glm1, direction="both")
step$anova
# Stepwise Model Path 
# Analysis of Deviance Table
# 
# Initial Model:
#   as.factor(target) ~ (id + ps_ind_01 + ps_ind_02_cat + ps_ind_03 + 
#                          ps_ind_04_cat + ps_ind_05_cat + ps_ind_06_bin + ps_ind_07_bin + 
#                          ps_ind_08_bin + ps_ind_09_bin + ps_ind_10_bin + ps_ind_11_bin + 
#                          ps_ind_12_bin + ps_ind_13_bin + ps_ind_14 + ps_ind_15 + ps_ind_16_bin + 
#                          ps_ind_17_bin + ps_ind_18_bin + ps_reg_01 + ps_reg_02 + ps_reg_03 + 
#                          ps_car_01_cat + ps_car_02_cat + ps_car_04_cat + ps_car_06_cat + 
#                          ps_car_07_cat + ps_car_08_cat + ps_car_09_cat + ps_car_10_cat + 
#                          ps_car_11_cat + ps_car_11 + ps_car_12 + ps_car_13 + ps_car_14 + 
#                          ps_car_15 + ps_calc_01 + ps_calc_02 + ps_calc_03 + ps_calc_04 + 
#                          ps_calc_05 + ps_calc_06 + ps_calc_07 + ps_calc_08 + ps_calc_09 + 
#                          ps_calc_10 + ps_calc_11 + ps_calc_12 + ps_calc_13 + ps_calc_14 + 
#                          ps_calc_15_bin + ps_calc_16_bin + ps_calc_17_bin + ps_calc_18_bin + 
#                          ps_calc_19_bin + ps_calc_20_bin) - id
# 
# Final Model:
#   as.factor(target) ~ ps_ind_15 + ps_ind_17_bin + ps_reg_03 + ps_car_01_cat + 
#   ps_car_09_cat + ps_car_11 + ps_car_12 + ps_car_13 + ps_car_14 + 
#   ps_car_15 + ps_calc_02 + ps_calc_09 + ps_ind_09_bin
# 
# 
# Step  Df     Deviance Resid. Df Resid. Dev      AIC
# 1                                         9800   2627.963 3027.963
# 2       - ps_ind_14   0 0.000000e+00      9800   2627.963 3027.963
# 3   - ps_ind_09_bin   0 0.000000e+00      9800   2627.963 3027.963
# 4   - ps_car_11_cat 103 1.407541e+02      9903   2768.717 2962.717
# 5   - ps_car_06_cat  17 1.088532e+01      9920   2779.603 2939.603
# 6   - ps_car_04_cat   8 1.004742e+01      9928   2789.650 2933.650
# 7   - ps_ind_02_cat   3 1.104909e+00      9931   2790.755 2928.755
# 8   - ps_car_10_cat   2 1.808518e+00      9933   2792.563 2926.563
# 9      - ps_calc_03   1 3.809930e-03      9934   2792.567 2924.567
# 10     - ps_calc_04   1 6.489372e-03      9935   2792.574 2922.574
# 11  - ps_ind_12_bin   1 5.228311e-02      9936   2792.626 2920.626
# 12  - ps_car_02_cat   1 6.673179e-02      9937   2792.693 2918.693
# 13     - ps_calc_14   1 6.990004e-02      9938   2792.763 2916.763
# 14 - ps_calc_16_bin   1 9.715303e-02      9939   2792.860 2914.860
# 15 - ps_calc_19_bin   1 1.219903e-01      9940   2792.982 2912.982
# 16     - ps_calc_06   1 1.399949e-01      9941   2793.122 2911.122
# 17  - ps_car_07_cat   1 1.435384e-01      9942   2793.265 2909.265
# 18  - ps_ind_10_bin   1 1.878630e-01      9943   2793.453 2907.453
# 19      - ps_reg_02   1 2.107397e-01      9944   2793.664 2905.664
# 20     - ps_calc_10   1 2.905952e-01      9945   2793.954 2903.954
# 21 - ps_calc_20_bin   1 3.028715e-01      9946   2794.257 2902.257
# 22     - ps_calc_01   1 3.113766e-01      9947   2794.569 2900.569
# 23     - ps_calc_13   1 4.161463e-01      9948   2794.985 2898.985
# 24 - ps_calc_18_bin   1 5.485398e-01      9949   2795.533 2897.533
# 25     - ps_calc_07   1 5.922330e-01      9950   2796.126 2896.126
# 26  - ps_ind_11_bin   1 6.358114e-01      9951   2796.761 2894.761
# 27      - ps_ind_03   1 6.744946e-01      9952   2797.436 2893.436
# 28  - ps_car_08_cat   1 7.598733e-01      9953   2798.196 2892.196
# 29      - ps_reg_01   1 7.315064e-01      9954   2798.927 2890.927
# 30      - ps_ind_01   1 7.545534e-01      9955   2799.682 2889.682
# 31  - ps_ind_04_cat   1 8.232782e-01      9956   2800.505 2888.505
# 32     - ps_calc_12   1 9.324971e-01      9957   2801.438 2887.438
# 33  - ps_ind_18_bin   1 9.704450e-01      9958   2802.408 2886.408
# 34  - ps_ind_16_bin   1 3.574361e-01      9959   2802.766 2884.766
# 35  - ps_ind_13_bin   1 1.056522e+00      9960   2803.822 2883.822
# 36  - ps_ind_05_cat   6 1.112935e+01      9966   2814.951 2882.951
# 37     - ps_calc_05   1 1.202294e+00      9967   2816.154 2882.154
# 38     - ps_calc_08   1 1.207225e+00      9968   2817.361 2881.361
# 39     - ps_calc_11   1 1.269683e+00      9969   2818.631 2880.631
# 40 - ps_calc_15_bin   1 1.585592e+00      9970   2820.216 2880.216
# 41 - ps_calc_17_bin   1 1.614129e+00      9971   2821.830 2879.830
# 42  - ps_ind_06_bin   1 1.759509e+00      9972   2823.590 2879.590
# 43  - ps_ind_07_bin   1 8.766195e-01      9973   2824.466 2878.466
# 44  - ps_ind_08_bin   1 6.087734e-01      9974   2825.075 2877.075
# 45  + ps_ind_09_bin   1 2.986501e+00      9973   2822.089 2876.089

##############
# From the randomly chosen 10,000
# Final Model:
#   as.factor(target) ~ ps_ind_15 + ps_ind_17_bin + ps_reg_03 + ps_car_01_cat + 
#   ps_car_09_cat + ps_car_11 + ps_car_12 + ps_car_13 + ps_car_14 + 
#   ps_car_15 + ps_calc_02 + ps_calc_09 + ps_ind_09_bin
summary(glm1)

# want to try it with 1000 random sample
# perform the same prodecure as above
# random.train2 <- sample_n(data.train,1000)
# glm2 <- glm(as.factor(target)~.-id, data=random.train2, family = binomial(link = 'logit'))
# step <- stepAIC(glm2, direction="both")
# step$anova
# Final Model:
# as.factor(target) ~ ps_ind_01 + ps_ind_02_cat + ps_ind_03 + ps_ind_07_bin + 
#   ps_ind_08_bin + ps_ind_12_bin + ps_reg_02 + ps_reg_03 + ps_car_01_cat + 
#   ps_car_02_cat + ps_car_04_cat + ps_car_06_cat + ps_car_08_cat + 
#   ps_car_09_cat + ps_car_11_cat + ps_car_11 + ps_car_13 + ps_car_14 + 
#   ps_calc_02 + ps_calc_09 + ps_calc_11 + ps_calc_12 + ps_calc_15_bin + 
#   ps_calc_17_bin + ps_calc_18_bin + ps_calc_19_bin + ps_calc_20_bin
############
# We feel like 1000 is not a good sample for such a large dataset.
# So we drop this experiment.

# ------- Using the variables selected from stepwise regression to build logistic model ---------- #
reg.model<-glm(as.factor(target) ~ ps_ind_15 + ps_ind_17_bin + ps_reg_03 + ps_car_01_cat + 
                  ps_car_09_cat + ps_car_11 + ps_car_12 + ps_car_13 + ps_car_14 + 
                 ps_car_15 + ps_calc_02 + ps_calc_09 + ps_ind_09_bin, data=train, 
               family = binomial(link = 'logit'))
prob.glm <- predict(reg.model, newdata=train,type = 'response')

tpr<-normalized.gini.index(train[,2],prob.glm) # gini index
tpr
# 0.217639

prob.reg <- predict(reg.model, newdata=test,type = 'response')
reg.table <- data.frame(test$id, prob.reg)
# write files
write.table(reg.table, file = 'regression.csv',row.names=F, col.names=c("id", "target"), sep=",")

# ------ Cross Validation for logistic regression 2nd time ----- #
set.seed(1)
sub2 <- sample(1:nrow(train),size=nrow(train)*0.8)
data.train2 <- train[sub2,]     # Select subset for cross-validation
data.valid2 <- train[-sub2,]

reg.model.cv<-glm(as.factor(target) ~ ps_ind_15 + ps_ind_17_bin + ps_reg_03 + ps_car_01_cat + 
                 ps_car_09_cat + ps_car_11 + ps_car_12 + ps_car_13 + ps_car_14 + 
                 ps_car_15 + ps_calc_02 + ps_calc_09 + ps_ind_09_bin, data=data.train2, 
               family = binomial(link = 'logit'))
prob.glm.cv <- predict(reg.model, newdata=data.valid2,type = 'response')

tpr.cv<-normalized.gini.index(data.valid2[,2],prob.glm.cv) # gini index
tpr.cv
# 0.2090159



# -------------- using linear regression --------------- #
lm1 <- lm(target~.-id,data=random.train1)
summary(lm1)
# using stepwise to choose the variables
step.lm <- stepAIC(lm1, direction = 'both')
step.lm$anova

# Final Model:
#   target ~ ps_ind_05_cat + ps_ind_15 + ps_ind_17_bin + ps_reg_03 + 
#   ps_car_01_cat + ps_car_04_cat + ps_car_09_cat + ps_car_13 + 
#   ps_car_14 + ps_calc_02 + ps_calc_09 + ps_ind_09_bin

lm.model <- lm(target ~ ps_ind_05_cat + ps_ind_15 + ps_ind_17_bin + ps_reg_03 + 
                    ps_car_01_cat  + ps_car_04_cat+ ps_car_09_cat + ps_car_13 + 
                    ps_car_14 + ps_calc_02 + ps_calc_09 + ps_ind_09_bin, data=train)
prob.lm <- predict(lm.model, newdata=train)
# prob.lm[prob.lm <0] <- 0
# prob.lm[prob.lm >1] <- 1
tpr<-normalized.gini.index(train[,2],prob.lm)
tpr
# 0.2376464

prob.lm.test <- predict(lm.model, newdata=test)
lm.table <- data.frame(test$id, prob.lm.test)
# write files
write.table(reg.table, file = 'linearregression.csv',row.names=F, col.names=c("id", "target"), sep=",")


# ------ Cross Validation for linear regression 2nd time ----- #
set.seed(1)
sub2 <- sample(1:nrow(train),size=nrow(train)*0.8)
data.train2 <- train[sub2,]     # Select subset for cross-validation
data.valid2 <- train[-sub2,]

lm.model.cv <- lm(target ~ ps_ind_05_cat + ps_ind_15 + ps_ind_17_bin + ps_reg_03 + 
                 ps_car_01_cat  + ps_car_04_cat+ ps_car_09_cat + ps_car_13 + 
                 ps_car_14 + ps_calc_02 + ps_calc_09 + ps_ind_09_bin, data=data.train2)
prob.lm.cv <- predict(lm.model.cv, newdata=data.valid2,type = 'response')

tpr.cv.lm<-normalized.gini.index(data.valid2[,2],prob.lm.cv) # gini index
tpr.cv.lm
# 0.2213452



### Basic Model ### (Jo)(can be deleted/commented out) used for initial submission
# mypred<-predict(lmraw,newdata = test)
# mypred[mypred<0]<-0
# mypred[mypred>1]<-1
# final_table<-data.frame(test$id, mypred)
# # Write files
# write.table(final_table, file="first.csv", row.names=F, col.names=c("id", "target"), sep=",")

####################  K-NN  ##################### (Jo)
tpr_table<-NULL   
for (n in c(1:5)){
  #not perfect balance, but close.
  #since not able to cv the whole dataset, I random sample for each round.
  balancetrain<-rbind(all1,sample_n(train[train$target==0,],30000))
  mydata<-sample_n(balancetrain,20000)
  mytrain<-mydata[1:10000,]
  mytest<-mydata[1:10000,]
  tpr_lst<-c()
  print("new data generated")
    for (j in c(1:5)){
    myknn<-knn(train=mytrain[,-c(1:2)],test=mytest[,-c(1:2)],cl=mytrain[,2],k=j)
    #tpr<-sum(myknn==mytest[,2] & mytest[,2]==1)/sum(mytest[,2]==1) #True positive rate
    tpr<-sum(myknn==mytest[,2] & mytest[,2]==0)/sum(mytest[,2]==0) #True negative rate
    #tpr<-normalized.gini.index(mytest[,2],myknn) #gini index doesn't help
    tpr_lst<-c(tpr_lst,tpr)
    print("new k")}
  tpr_table<-rbind(tpr_table,tpr_lst)
}
colnames(tpr_table)<-c(1:5)
tpr_table

#result for k:3,5,10,20,30 (with train):  below 5 seem to be better
#   tpr\ K       *3*           5 10 20 30
#tpr_lst 0.014164306 0.002832861  0  0  0
#tpr_lst 0.005763689 0.002881844  0  0  0
#tpr_lst 0.005277045 0.000000000  0  0  0
#tpr_lst 0.005649718 0.000000000  0  0  0
#tpr_lst 0.002747253 0.000000000  0  0  0

#result for k:1-5 (with train)：   1 always outperform others
#    tpr\ K *[,1]*       [,2]        [,3]        [,4]        [,5]
#tpr_lst 0.04986877 0.03937008 0.002624672 0.002624672 0.002624672
#tpr_lst 0.06268657 0.04776119 0.011940299 0.005970149 0.002985075
#tpr_lst 0.06631300 0.06366048 0.002652520 0.002652520 0.000000000
#tpr_lst 0.06628242 0.04899135 0.002881844 0.000000000 0.000000000
#tpr_lst 0.08421053 0.05263158 0.007894737 0.002631579 0.000000000

#result for k:1-5 (with balancetrain)：  
# tpr\ K     1         2         3         4         5
#    tpr_lst 1 0.7019346 0.6548842 0.5932649 0.5502747
#    tpr_lst 1 0.6968968 0.6560019 0.5807072 0.5388501
#    tpr_lst 1 0.6989222 0.6452671 0.5871603 0.5485005
#    tpr_lst 1 0.7092080 0.6519561 0.5813454 0.5438931
#    tpr_lst 1 0.7139457 0.6446930 0.5935269 0.5530700

#true negative rate result for k:1-5 (with balancetrain)：
#
#        1         2         3         4         5
#tpr_lst 1 0.7960549 0.8373928 0.7879931 0.8137221
#tpr_lst 1 0.8144858 0.8423300 0.7975743 0.8228562
#tpr_lst 1 0.8008284 0.8410425 0.7947877 0.8182603
#tpr_lst 1 0.8122407 0.8561549 0.8084371 0.8354080
#tpr_lst 1 0.8129374 0.8474142 0.8028674 0.8294931

balancetrain<-sample_n(rbind(all1,sample_n(train[train$target==0,],30000)),10000)
knnpred<-knn(train=balancetrain[,-c(1:2)],test=test[,-c(1:2)],cl=balancetrain[,2],k=1)
final_table<-data.frame(test$id, knnpred)
write.table(final_table, file="knn.csv", row.names=F, col.names=c("id", "target"), sep=",")

####################  Trees  #################### (Sai)

################  Random Forest  ################ (Sai)
