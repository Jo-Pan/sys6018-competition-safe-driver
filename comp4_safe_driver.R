#competition 4 safe driver
#https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/data

train<-read.csv("train.csv")
summary(train)
test<-read.csv("test.csv")

# need data clean 
train2<-read.csv("train.csv")
lmraw<-lm(target~.,data=train)

mypred<-predict(lmraw,newdata = test)
mypred[mypred<0]<-0
mypred[mypred>1]<-1
final_table<-data.frame(test$id, mypred)
# Write files
write.table(final_table, file="first.csv", row.names=F, col.names=c("id", "target"), sep=",")