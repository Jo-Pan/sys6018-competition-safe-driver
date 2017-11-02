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





# lets explore some more trees based approaches. we will see what turns up





# bagging approach

library(randomForest)

train3 <- train2[sample(nrow(train2), 10000), ]

test = train2[sample(nrow(train2), 10000), ]

summary(train3$target)

# 3.66% are 1's. This is pretty close to real value

bag.porto <- randomForest(target ~ ., data = train3, mtry = 55, ntree = 5, importance = TRUE)
yhat.bag <- predict(bag.porto, newdata = test, type = "class")

summary(yhat.bag)
table(yhat.bag,test$target)

prob <- predict(bag.porto, test, type = "prob")

# there's a key learning here. Classification trees run much faster than regression trees. Tried that

#yhat.bag   0       1
#   0       9608    353
#   1       39      8 


# random forest

train4 <- train2[sample(nrow(train2), 10000), ]
test = train2[sample(nrow(train2), 10000), ]

rf.porto <- randomForest(target ~ ., data = train4, mtry = 8, ntree = 5, importance = TRUE)
yhat.bag <- predict(rf.porto, newdata = test, type = "class")

# mtry is square root of total predictors

summary(yhat.bag)
table(yhat.bag,test$target)

#yhat.bag   0     1
#0         9607   360
#1          29    4

prob <- predict(rf.porto, test, type = "prob")

importance(rf.porto)


#                         0           1 MeanDecreaseAccuracy MeanDecreaseGini
# id             -0.04025452  1.19480792           0.22436613     38.435765253
# ps_ind_01       7.26756030  0.66030796           7.37068398     16.014471436
# ps_ind_02_cat   6.31924829 -2.76233898           5.73173411      7.723166354
# ps_ind_03       9.63014257 -0.80416356           9.39749901     22.728275983
# ps_ind_04_cat   3.02688606 -1.15456769           2.76923899      2.717580042
# ps_ind_05_cat   3.40735190  5.59960499           4.80798570     12.870403906
# ps_ind_06_bin   6.32610310 -4.30309718           5.68893905      4.070708102
# ps_ind_07_bin   9.10699732  0.43839186           9.01106816      4.616474510
# ps_ind_08_bin  -0.16343486 -0.47308637          -0.24338395      3.729193202
# ps_ind_09_bin   2.23581646  0.22723215           2.27478513      3.424232808
# ps_ind_10_bin   0.00000000  0.00000000           0.00000000      0.006555556
# ps_ind_11_bin  -0.63149883  0.50181888          -0.53881494      0.900562965
# ps_ind_12_bin  -1.29387359 -0.28742447          -1.35540318      0.675569327
# ps_ind_13_bin   0.00000000  0.00000000           0.00000000      0.011000000
# ps_ind_14      -0.46055574 -0.93651001          -0.64025888      1.151615790
# ps_ind_15       4.14353762 -2.98249073           3.56687577     20.458815383
# ps_ind_16_bin   3.32474858  1.04347526           3.37351544      4.683771938
# ps_ind_17_bin   3.32909503 -1.47988561           3.10815190      4.115044208
# ps_ind_18_bin   4.87978072 -1.94385470           4.55183159      3.765969895
# ps_reg_01       9.10131802 -2.13150926           8.68911012     14.916797865
# ps_reg_02      15.76111410 -4.58177899          15.28321769     21.291418343
# ps_reg_03      17.64536016 -4.24819971          17.21763925     35.080668913
# ps_car_01_cat  12.15826592 -1.09687902          11.82951055     21.893125697
# ps_car_02_cat   5.17487659 -0.84526378           5.05511422      1.823042205
# ps_car_04_cat  10.31848484 -4.44213631          10.37836365      7.594751139
# ps_car_06_cat  15.19963665 -3.16994795          14.75500687     34.450838803
# ps_car_07_cat   1.93152475  0.97869192           2.14254174      2.415850881
# ps_car_08_cat   5.36645517 -1.63867233           5.04314170      2.461433323
# ps_car_09_cat   9.55069867 -2.15774461           9.18570057      9.569265712
# ps_car_10_cat   0.86159299 -1.13184780           0.60131984      1.008233945
# ps_car_11       8.07042022 -1.54194628           7.98228545      7.825892097
# ps_car_12      15.69841377 -5.80065546          15.60270581     16.655136838
# ps_car_13      23.24586819 -6.19574103          23.08973305     36.670207999
# ps_car_14      17.23885918 -4.07272447          16.97557335     31.099228675
# ps_car_15      11.96761504 -2.70281046          11.70712394     17.871010417
# ps_calc_01      0.29694591  0.85880967           0.47588421     18.769930229
# ps_calc_02     -0.15760363  1.16557736           0.08235381     19.186725772
# ps_calc_03     -0.58138077 -0.74451936          -0.72507750     19.991917859
# ps_calc_04      1.48638950  0.65619288           1.58804445     15.620913856
# ps_calc_05     -0.68430568  0.31019346          -0.60348025     13.714182263
# ps_calc_06     -1.67403848  0.04657021          -1.61485861     16.520092623
# ps_calc_07      1.90385655  1.62223998           2.21127337     17.594644864
# ps_calc_08     -1.92376163  0.07354145          -1.87086006     19.185903123
# ps_calc_09      0.96613520 -0.32422003           0.88134313     16.891908493
# ps_calc_10      0.77287775  0.16943014           0.74451541     23.171962667
# ps_calc_11     -0.60163415  0.15572323          -0.54568160     22.328087578
# ps_calc_12      0.82113456  0.16485055           0.84045166     14.335544525
# ps_calc_13     -1.32710452  1.12377149          -1.06006525     20.847960793
# ps_calc_14     -0.97513747  0.10661703          -0.97331766     22.352432738
# ps_calc_15_bin -0.56437663  1.21231494          -0.35806428      4.597289599
# ps_calc_16_bin -1.30474076 -0.44964802          -1.33012676      5.060269362
# ps_calc_17_bin  0.18868541  1.93110915           0.65823878      5.496522677
# ps_calc_18_bin  2.29677307 -0.33518759           2.13992097      5.323885313
# ps_calc_19_bin -0.22784601 -0.05488170          -0.22386072      5.717105071
# ps_calc_20_bin  0.30370860  0.50430792           0.39440685      3.522452208


# Boosting

train5 <- train2[sample(nrow(train2), 10000), ]


library(gbm)
boost.porto =gbm(target~.,data=train5, distribution = "bernoulli", n.trees =500, interaction.depth = 10, shrinkage = 0.2, verbose = F)

# actually, in my boosting model, there are no predictors that had non zero influence!

#"A gradient boosted model with bernoulli loss function.
#500 iterations were performed.
#There were 55 predictors of which 0 had non-zero influence."

summary(boost.porto)

yhat.bag <- predict(boost.porto, newdata = test, n.trees = 50)

yhat.bag

prob <- predict(boost.porto, test, type = "prob")


summary(yhat.bag)
table(yhat.bag,test$target)



