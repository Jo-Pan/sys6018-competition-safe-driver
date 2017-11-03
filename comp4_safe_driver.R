#competition 4 safe driver
#https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/data
#https://github.com/Jo-Pan/sys6018-competition-safe-driver

library(caret) #data cleaning, knn
library(class) #knn
library(dplyr) #sample_n for sampling rows

train<-read.csv("train.csv") #892816,59
test<-read.csv("test 2.csv") #595212,59

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

# let me verify these categorical values

cat_vars

unique(train$ps_ind_02_cat) #4
unique(train$ps_ind_04_cat) #2
unique(train$ps_ind_05_cat) #7
unique(train$ps_car_01_cat) #12
unique(train$ps_car_02_cat) #2
unique(train$ps_car_04_cat) #10
unique(train$ps_car_06_cat) #18
unique(train$ps_car_07_cat) #2
unique(train$ps_car_08_cat) #2
unique(train$ps_car_09_cat) #5
unique(train$ps_car_10_cat) #3
unique(train$ps_car_11_cat) #104 - we have a problem. need to talk to prof.

#convert cat_vars to factor
comb[cat_vars]<-lapply(comb[cat_vars],factor)

# i don't have a data dictionary.
# I won't be using this column: train$ps_car_11_cat

#split train & test sets
train<-comb[is.na(comb$target)==FALSE,]
test<-comb[is.na(comb$target)==TRUE,]


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
#Balanced data linear regression (best model) ----------------------------------------------------

# Since the train data is highly unbalanced (much more observations with target 0 than 1), we want to reweight 
# the train data to build the model

all1<-train[train$target==1,]
nrow(all1) # 21694
all0 <- train[train$target==0,]
nrow(all0) # 573518
# randomly choose 30000 observations from target = 0 data
random.0 <- sample_n(all0,30000)
# combine 30000 of target = 0 data with all the target =1 data
comb.balance <- rbind(all1, random.0)
# Then randomly sample 10000 from the balanced data to do the analysis
random.sample.bal <- sample_n(comb.balance,10000)

bal.lm <- lm(target~.-id,data=random.sample.bal)
# using stepwise to choose the variables
step.bal <- stepAIC(bal.lm, direction = 'both')
step.bal$anova
# final model
# target ~ ps_ind_01 + ps_ind_02_cat + ps_ind_03 + ps_ind_04_cat + 
#  ps_ind_05_cat + ps_ind_07_bin + ps_ind_08_bin + ps_ind_13_bin + 
#  ps_ind_15 + ps_ind_17_bin + ps_reg_01 + ps_reg_02 + ps_car_01_cat + 
#  ps_car_04_cat + ps_car_07_cat + ps_car_08_cat + ps_car_13 + 
#  ps_car_14 + ps_calc_18_bin + ps_calc_19_bin

# cross validate 
lm.model.cv.bal <- lm(target ~ ps_ind_01 + ps_ind_02_cat + ps_ind_03 + ps_ind_04_cat + 
                        ps_ind_05_cat + ps_ind_07_bin + ps_ind_08_bin + ps_ind_13_bin + 
                        ps_ind_15 + ps_ind_17_bin + ps_reg_01 + ps_reg_02 + ps_car_01_cat + 
                        ps_car_04_cat + ps_car_07_cat + ps_car_08_cat + ps_car_13 + 
                        ps_car_14 + ps_calc_18_bin + ps_calc_19_bin, data=data.train2)
prob.lm.cv.bal <- predict(lm.model.cv.bal, newdata=data.valid2,type = 'response')

tpr.cv.lm.bal<-normalized.gini.index(data.valid2[,2],prob.lm.cv.bal) # gini index
tpr.cv.lm.bal
# [1] 0.2365163

prob.lm.test.bal <- predict(lm.model.cv.bal, newdata=test)
prob.lm.test.bal[prob.lm.test.bal<0] = 0
prob.lm.test.bal[prob.lm.test.bal>1] = 1
lm.table.bal <- data.frame(test$id, prob.lm.test.bal)
# write files
write.table(lm.table.bal, file = 'linearregressionbal.csv',row.names=F, col.names=c("id", "target"), sep=",")



# separate train for cross validation -------------------------------------
# originally intend to separate data into training set and validation set
# But the number of obervations is too huge

set.seed(1)
sub <- sample(1:nrow(train),size=nrow(train)*0.7)
data.train <- train[sub,]     # Select subset for cross-validation
data.valid <- train[-sub,]

# ---------- model building --------- #
# since observations are too big, random sample 10,000 from it to build the model
random.train1 <- sample_n(data.train,10000) # randomly choose 10,000 observations from training set

# using logistic regression (imbalanced data)------------------------------------------------
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
write.table(reg.table, file = 'logisticregression.csv',row.names=F, col.names=c("id", "target"), sep=",")

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


# using linear regression (imbalanced data)--------------------------------------------------------
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
write.table(lm.table, file = 'linearregression.csv',row.names=F, col.names=c("id", "target"), sep=",")


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

############  Step-wise Regression  ############# (Jo)




####################  Trees  #################### (Sai)

# what percentage of target column is 1?

nrow(subset(train, train$target == 1))*100/nrow(train)

# just 3.64%. We have a class imbalance problem in this data set. Need to think about that

# decision trees

library(tree)

train2 = train

train2[c("ps_car_11_cat")] <- list(NULL) # I am removing the columns that it is a categorical variable but has 104 unique values

train2$target = as.factor(train2$target)

# this is a classification problem

tree.porto =tree(target ~. ,train2)

summary(tree.porto)

tree.porto

# residual mean deviance = 0.0313

# importantly, misclassification = 0.03645

test = train2[sample(nrow(train2), 10000), ]

tree.pred =predict(tree.porto,test,type ="class")
table(tree.pred,test$target)

#tree.pred   0    1
#0         9638  362
#1           0    0

library(rpart)
prob <- predict(tree.porto, test, type = "prob")


# lets do cross validation to find optimal value

cv.porto <- cv.tree(tree.porto, FUN=prune.misclass)
plot(cv.porto$size, cv.porto$dev, type = "b")
tree.min <- which.min(cv.porto$dev)
points(tree.min, cv.porto$dev[tree.min], col = "red", cex = 2, pch = 20)

# what does it mean it cannot prune further. Let's see the tree

plot(tree.porto)
text(tree.porto, pretty = 0)

# oh. its a single node tree


# bagging approach

library(randomForest)

train3 <- train2[sample(nrow(train2), 10000), ]

test = train2[sample(nrow(train2), 10000), ]

summary(train3$target)

bag.porto <- randomForest(target ~ ., data = train3, mtry = 55, ntree = 5, importance = TRUE)
yhat.bag <- predict(bag.porto, newdata = test, type = "class")

summary(yhat.bag)
table(yhat.bag,test$target)

prob <- predict(bag.porto, test, type = "prob")

# there's a key learning here. Classification trees run much faster than regression trees. Tried that just to check

#yhat.bag   0       1
#   0       9608    353
#   1       39      8 



# random forest

library(caret)

objControl <- trainControl(method='cv', number=3, returnResamp='none', summaryFunction = twoClassSummary, classProbs = TRUE)

train5 <- train2[sample(nrow(train2), 1000), ]
train5$target<-as.factor(train5$target)
levels(train5$target) <- make.names(levels(factor(train5$target)))


tunegrid <- expand.grid(mtry=c(5:15))
myrf <- train(train5[,-c(1,2)], train5[,2], 
              method="rf", 
              metric="ROC", 
              tuneGrid=tunegrid, 
              trControl=objControl)
summary(myrf)
plot(myrf)

# The results are really inconclusive. My laptops computing power is not enough to use a larger training set

# For this project, we are going to stick to mtry = 7 (close to square root of 55)

train4 <- train2[sample(nrow(train2), 10000), ]
rf.porto <- randomForest(target ~ ., data = train4, mtry = 8, ntree = 500, importance = TRUE)
yhat.rf <- predict(rf.porto, newdata = train2, type = "class")

# mtry is square root of total predictors

# I have realized that to do better on gini especially on this data set, probability of 1 should  be close to zero for most of the rows (class imbalance). The probability is better estimated by random forests if I increase the number of trees rather than sample size of training.

summary(yhat.rf)
table(yhat.rf,train2$target)

#for sample = 10,000 and ntree = 5

#yhat.bag   0     1
#0         9607   360
#1          29    4

#for sample = 100,000 and ntree = 5

#yhat.bag     0     1
#0          96161  3200
#1           156   483

# for sample = 100,000 and ntree = 50 (the maximum my laptop can run probably)

#yhat.bag     0     1
#0          96333  3033
#1            0   634

# Overall, it seems like many 1's are being misclassified. Need to fix that

# sample size 10,000  number of trees = 500 (I bet this will have the highest gini)

#yhat.rf      0      1
#0 573516  21282
#1      2    412


prob <- as.data.frame(predict(rf.porto, train2, type = "prob"))

normalized.gini.index(as.numeric(train2$target),prob$`1`)

#0.1999 that's decent. 

# Need to tinker with balancing data proportion. Cost functions won't help because we can make the model classify more 1's but it won't fundamentally alter probabilities like the way balancing does

# what is the optimal data distribution? Lets find out

gini <- c()

for (p in 1:10) {

  for (k in 1:10) {
    
  
all1<-train2[train2$target==1,]
random.1 <- sample_n(all1,1000*p)

all0 <- train2[train2$target==0,]
random.0 <- sample_n(all0,1000*k)
comb.balance <- rbind(random.1, random.0)

sid<-as.numeric(rownames(comb.balance))
test<-train2[-sid,]

# We wish to maintain the class ratio in test set. Hence further adjustments to the test set.

test <- test[sample(nrow(test), ((21694 - 1000*p)*570000/21694)), ]

rf.comb.porto <- randomForest(target ~ ., data = comb.balance, mtry = 7, ntree = 500, importance = TRUE)
prob <- as.data.frame(predict(rf.comb.porto, test, type = "prob"))

gini <- c(gini, normalized.gini.index(as.numeric(test$target),prob$`1`), p, k)

}
}


# write.csv(gini, file = "full_clean_noNA.csv", row.names = FALSE)

# The following trends are observed in the results broadly

# Gini tends to go up with sample size of training (obvious)

# The best gini's are produced by close to balanced training set or slight bias towards 0's. Models that are extremely biased towards one class have bad gini's

# training the final model

all1<-train2[train2$target==1,]
random.1 <- sample_n(all1,10000)

all0 <- train2[train2$target==0,]
random.0 <- sample_n(all0,12000)
comb.balance <- rbind(random.1, random.0)

sid<-as.numeric(rownames(comb.balance))
test<-train2[-sid,]

# We wish to maintain the class ratio in test set. Hence further adjustments to the test set.

test <- test[sample(nrow(test), ((21694 - 1000*p)*570000/21694)), ]

rf.comb.porto <- randomForest(target ~ ., data = comb.balance, mtry = 7, ntree = 500, importance = TRUE)
prob <- as.data.frame(predict(rf.comb.porto, test, type = "prob"))

gini_1012 <- normalized.gini.index(as.numeric(test$target),prob$`1`)

#gini_1012  0.2557 on the training set



library(caret)

test.all = test

test.all[c("ps_car_11_cat")] <- list(NULL) 

comb.balance$target<-as.factor(comb.balance$target)
levels(comb.balance$target) <- make.names(levels(factor(comb.balance$target)))

objControl <- trainControl(method='cv', number=3, returnResamp='none', summaryFunction = twoClassSummary, classProbs = TRUE)

tunegrid <- expand.grid(mtry=c(7))
myrf3 <- train(comb.balance[,-c(1,2)], comb.balance[,2], 
               method="rf", 
               metric="ROC", 
               tuneGrid=tunegrid, 
               trControl=objControl)



rf3result <- predict(myrf3, newdata = test.all[,-c(1)],type="prob")
final_table<-data.frame(test.all$id, rf3result$X1)
write.table(final_table, file="rf_bal.csv", row.names=F, col.names=c("id", "target"), sep=",")

# Kaggle score gini: 0.243

# Let me try calculating probabilities in another way (also slightly altering ratio). Also training on literally all the 1's. Can't get better than that unless I fundamentally alter the setting.

all1<-train2[train2$target==1,]
random.1 <- sample_n(all1,21694)

all0 <- train2[train2$target==0,]
random.0 <- sample_n(all0,21694)
comb.balance <- rbind(random.1, random.0)

# probabilities on the actual testing set

rf.comb.porto <- randomForest(target ~ ., data = comb.balance, mtry = 7, ntree = 1000, importance = TRUE)
prob <- as.data.frame(predict(rf.comb.porto, test, type = "prob"))

final_table2 <- data.frame(test$id, prob$`1`)
write.table(final_table2, file="rf_bal_2nd.csv", row.names=F, col.names=c("id", "target"), sep=",")

#0.245 kaggle score





# Boosting
#Boosting 1------------------------------------------------------------------------------------
train5 <- train2[sample(nrow(train2), 10000), ]
train5$target<-as.factor(train5$target)
levels(train5$target) <- make.names(levels(factor(train5$target)))

library(caret)
objControl <- trainControl(method='cv', number=3, returnResamp='none', summaryFunction = twoClassSummary, classProbs = TRUE)
gbmGrid <-  expand.grid(interaction.depth = c(1, 5, 10), 
                        n.trees = c(50,100,500,1000,1500), 
                        shrinkage = c(0.1,0.01,0.001),
                        n.minobsinnode = c(10,20))
objModel <- train(train5[,-c(1,2)], train5[,2], 
                  method='gbm', 
                  trControl=objControl,  
                  metric = "ROC",
                  preProc = c("center", "scale"),
                  tuneGrid = gbmGrid)
#Boosting 1 train summary ------------------------------------------------------------------------
summary(objModel)
# var     rel.inf
# ps_car_06_cat   ps_car_06_cat 22.03165674
# ps_car_13           ps_car_13 14.75327775
# ps_car_01_cat   ps_car_01_cat 12.82256031
# ps_ind_05_cat   ps_ind_05_cat  7.99337009
# ps_reg_03           ps_reg_03  5.51148099
# ps_car_09_cat   ps_car_09_cat  3.56288546
# ps_car_14           ps_car_14  3.23284808
# ps_ind_15           ps_ind_15  2.77970094
# ps_ind_03           ps_ind_03  2.36881750
# ps_calc_13         ps_calc_13  2.25026944
# ps_car_15           ps_car_15  1.94172940
# ps_reg_02           ps_reg_02  1.89683400
# ps_ind_01           ps_ind_01  1.62794619
# ps_calc_11         ps_calc_11  1.57151831
# ps_calc_14         ps_calc_14  1.56407381
# ps_calc_12         ps_calc_12  1.49601807
# ps_calc_02         ps_calc_02  1.26637475
# ps_car_12           ps_car_12  1.23922587
# ps_calc_03         ps_calc_03  1.23512725
# ps_calc_07         ps_calc_07  1.18022965
# ps_calc_01         ps_calc_01  1.06972823
# ps_reg_01           ps_reg_01  0.90786879
# ps_ind_06_bin   ps_ind_06_bin  0.83844095
# ps_calc_10         ps_calc_10  0.70949013
# ps_calc_09         ps_calc_09  0.59376955
# ps_ind_04_cat   ps_ind_04_cat  0.53937635
# ps_calc_06         ps_calc_06  0.42583904
# ps_ind_16_bin   ps_ind_16_bin  0.35234179
# ps_calc_08         ps_calc_08  0.30768537
# ps_ind_02_cat   ps_ind_02_cat  0.25891251
# ps_car_04_cat   ps_car_04_cat  0.21760200
# ps_ind_07_bin   ps_ind_07_bin  0.21621282
# ps_calc_17_bin ps_calc_17_bin  0.21068628
# ps_calc_18_bin ps_calc_18_bin  0.19533569
# ps_ind_08_bin   ps_ind_08_bin  0.18714070
# ps_calc_15_bin ps_calc_15_bin  0.16587587
# ps_car_02_cat   ps_car_02_cat  0.15007212
# ps_calc_19_bin ps_calc_19_bin  0.13244138
# ps_calc_05         ps_calc_05  0.11305711
# ps_car_11           ps_car_11  0.08217873

print(objModel)
# shrinkage	interaction.depth	n.minobsinnode	n.trees	ROC	Sens
# 0.01	5	20	100	0.6298884	1
# 0.01	5	10	50	0.6297854	1
# 0.001	5	10	1000	0.6293613	1
# 0.001	10	10	100	0.6291573	1
# 0.001	5	20	500	0.6287911	1

# ROC was used to select the optimal model using  the largest value.
# The final values used for the model were n.trees = 100, interaction.depth =
#   5, shrinkage = 0.01 and n.minobsinnode = 20.

#Boosting 1 gini = 0.2027495------
model.tree1 <- predict(objModel, newdata = test[,-c(1,2)],type="prob")

normalized.gini.index(as.numeric(test$target),model.tree1$X2)
#0.2027495

#Boosting 2 gini = 0.2143287 ------------------------------------------------------------------------
gbmGrid2 <-  expand.grid(interaction.depth = 5, 
                        n.trees = c(1000,2000), 
                        shrinkage = c(0.001,0.0001),
                        n.minobsinnode = 10)
objModel2 <- train(train5[,-c(1,2)], train5[,2], 
                  method='gbm', 
                  trControl=objControl,  
                  metric = "ROC",
                  preProc = c("center", "scale"),
                  tuneGrid = gbmGrid2)
print(objModel2)
#he final values used for the model were n.trees = 2000, interaction.depth =
#5, shrinkage = 1e-04 and n.minobsinnode = 10.
model.tree2 <- predict(objModel2, newdata = test[,-c(1,2)],type="prob")

normalized.gini.index(as.numeric(test$target),model.tree2$X1)
#0.2143287

#Boosting 3 gini = 0.2182747 ------------------------------------------------------------------------
gbmGrid3 <-  expand.grid(interaction.depth = 5, 
                         n.trees = c(2000,5000), 
                         shrinkage = 0.0001,
                         n.minobsinnode = 10)
objModel3 <- train(train5[,-c(1,2)], train5[,2], 
                   method='gbm', 
                   trControl=objControl,  
                   metric = "ROC",
                   preProc = c("center", "scale"),
                   tuneGrid = gbmGrid3)

print(objModel3)
#The final values used for the model were n.trees = 5000, interaction.depth =
#5, shrinkage = 1e-04 and n.minobsinnode = 10.
model.tree3 <- predict(objModel3, newdata = test[,-c(1,2)],type="prob")
normalized.gini.index(as.numeric(test$target),model.tree3$X1)

#0.2182747



#Boosting 4 (Balanced data) gini = 0.2692118 ------------------------------------------------------------------------
all1<-train2[train2$target==1,]
nrow(all1) # 21694
all0 <- train2[train2$target==0,]
nrow(all0) # 573518
# randomly choose 30000 observations from target = 0 data
random.0 <- sample_n(all0,30000)
# combine 30000 of target = 0 data with all the target =1 data
comb.balance <- rbind(all1, random.0)
# Then randomly sample 10000 from the balanced data to do the analysis
random.sample.bal <- sample_n(comb.balance,10000)
random.sample.bal$target<-as.factor(random.sample.bal$target)
levels(random.sample.bal$target) <- make.names(levels(factor(random.sample.bal$target)))

objModel4 <- train(random.sample.bal[,-c(1,2)], random.sample.bal[,2], 
                   method='gbm', 
                   trControl=objControl,  
                   metric = "ROC",
                   preProc = c("center", "scale"),
                   tuneGrid = gbmGrid3)
print(objModel4)
#The final values used for the model were n.trees = 5000, interaction.depth =
#5, shrinkage = 1e-04 and n.minobsinnode = 10.

model.tree4 <- predict(objModel4, newdata = test[,-c(1,2)],type="prob")
normalized.gini.index(as.numeric(test$target),model.tree4$X1)

#Boosting 5 (All balanced data)   ------------------------------------------------------------------------
comb.balance$target<-as.factor(comb.balance$target)
levels(comb.balance$target) <- make.names(levels(factor(comb.balance$target)))
gbmGrid5 <-  expand.grid(interaction.depth = 5, 
                         n.trees = c(5000), 
                         shrinkage = 0.0001,
                         n.minobsinnode = 10)

objModel5 <- train(comb.balance[,-c(1,2)], comb.balance[,2], 
                   method='gbm', 
                   trControl=objControl,  
                   metric = "ROC",
                   preProc = c("center", "scale"),
                   tuneGrid = gbmGrid5)

#Boosting 5 Final Submission ------------------
test.all<-read.csv("test 2.csv") #595212,59
test.all[c("ps_car_11_cat")] <- list(NULL) 
boosting <- predict(objModel5, newdata = test.all[,-c(1)],type="prob")
final_table<-data.frame(test.all$id, boosting$X1)
write.table(final_table, file="boosting.csv", row.names=F, col.names=c("id", "target"), sep=",")
