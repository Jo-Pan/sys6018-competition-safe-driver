#competition 4 safe driver
#https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/data
#https://github.com/Jo-Pan/sys6018-competition-safe-driver
setwd("~/Desktop")
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

#convert cat_vars to factor
comb[cat_vars]<-lapply(comb[cat_vars],factor)

#split train & test sets
train<-comb[is.na(comb$target)==FALSE,]
test<-comb[is.na(comb$target)==TRUE,]

all1<-train[train$target==1,]
balancetrain<-rbind(all1,sample_n(train[train$target==0,],30000))
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
### Basic Model ### (Jo)(can be deleted/commented out)
mypred<-predict(lmraw,newdata = test)
mypred[mypred<0]<-0
mypred[mypred>1]<-1
final_table<-data.frame(test$id, mypred)
# Write files
write.table(final_table, file="first.csv", row.names=F, col.names=c("id", "target"), sep=",")

####################  K-NN  ##################### (Jo)
tpr_table<-NULL   
for (n in c(1:5)){
  mytrain<-sample_n(train,10000)
  mytest<-sample_n(train,10000)
  tpr_lst<-c()
  print(1)
    for (j in c(1:5)){
    myknn<-knn(train=mytrain[,-c(1:2)],test=mytest[,-c(1:2)],cl=mytrain[,2],k=j)
    tpr<-sum(myknn==mytest[,2] & mytest[,2]==1)/sum(mytest[,2]==1) #True positive rate
    tpr_lst<-c(tpr_lst,tpr)
    print(2)}
  tpr_table<-rbind(tpr_table,tpr_lst)
}
colnames(tpr_table)<-c(1:5)
tpr_table

#result for k:3,5,10,20,30:  below 5 seem to be better
# K                3           5 10 20 30
#tpr_lst 0.014164306 0.002832861  0  0  0
#tpr_lst 0.005763689 0.002881844  0  0  0
#tpr_lst 0.005277045 0.000000000  0  0  0
#tpr_lst 0.005649718 0.000000000  0  0  0
#tpr_lst 0.002747253 0.000000000  0  0  0

#result for k:1-5    1 always outperform others
#          [,1]       [,2]        [,3]        [,4]        [,5]
#tpr_lst 0.04986877 0.03937008 0.002624672 0.002624672 0.002624672
#tpr_lst 0.06268657 0.04776119 0.011940299 0.005970149 0.002985075
#tpr_lst 0.06631300 0.06366048 0.002652520 0.002652520 0.000000000
#tpr_lst 0.06628242 0.04899135 0.002881844 0.000000000 0.000000000
#tpr_lst 0.08421053 0.05263158 0.007894737 0.002631579 0.000000000

knnpred<-knn(train=balancetrain[,-c(1:2)],test=test[,-c(1:2)],cl=balancetrain[,2],k=1)
############  Step-wise Regression  ############# (Jo)

####################  Trees  #################### (Sai)

################  Random Forest  ################ (Sai)