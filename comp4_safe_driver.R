#competition 4 safe driver
#https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/data
#https://github.com/Jo-Pan/sys6018-competition-safe-driver
train<-read.csv("train.csv")
test<-read.csv("test 2.csv")
train_id<-train$id
#Combine train and test files
test$target <- NA
comb <- rbind(train, test)

################  Data Cleaning  ################## (Jo)
#indicate nas:
comb[comb==-1]<-NA

#what % in columns are na:
colMeans(is.na(comb)) #none is significantly large

#collect names of all categorical variables 
cat_vars <- names(comb)[grepl('_cat$', names(comb))]

#convert cat_vars to factor
comb[cat_vars]<-lapply(comb[cat_vars],factor)

#split train & test sets
train<-comb[train_id,]
test<-comb[-train_id,]

#drop id column
train<-train[,-1]
test<-test[,-1]

################  Linear Model  ################## (Teresa)
### Basic Model ### (Jo)(can be deleted/commented out)
mypred<-predict(lmraw,newdata = test)
mypred[mypred<0]<-0
mypred[mypred>1]<-1
final_table<-data.frame(test$id, mypred)
# Write files
write.table(final_table, file="first.csv", row.names=F, col.names=c("id", "target"), sep=",")

####################  K-NN  ##################### (Jo)


############  Step-wise Regression  ############# (Jo)

####################  Trees  #################### (Sai)

################  Random Forest  ################ (Sai)