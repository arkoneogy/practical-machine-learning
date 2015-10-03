library(data.table)
library(caret)

# q1
library(ElemStatLearn)
data(vowel.train)
data(vowel.test)

vowel.train$y= as.factor(vowel.train$y)
vowel.test$y= as.factor(vowel.test$y)
set.seed(33833)

rf= train(y~., data= vowel.train, method= 'rf')
rf.preds= predict(rf, vowel.test)
confusionMatrix(vowel.test$y, rf.preds)

gb= train(y~., data= vowel.train, method= 'gbm')
gb.preds= predict(gb, vowel.test)
confusionMatrix(vowel.test$y, gb.preds)

agree= data.table(actual= vowel.test$y, rf= rf.preds, gb= gb.preds)
agree[, ifmatch:= ifelse(rf.preds==gb.preds, 1, 0)]

confusionMatrix(agree[ifmatch==1]$actual, agree[ifmatch==1]$rf)



# q2
library(AppliedPredictiveModeling)
set.seed(3433)
data(AlzheimerDisease)
adData= data.frame(diagnosis, predictors)
inTrain = createDataPartition(adData$diagnosis, p = 3/4)[[1]]
training = adData[ inTrain,]
testing = adData[-inTrain,]

set.seed(62433)
m1= train(diagnosis~., data= training, method= 'rf')
m2= train(diagnosis~., data= training, method= 'gbm')
m3= train(diagnosis~., data= training, method= 'lda')

training$p1= predict(m1, training)
training$p2= predict(m2, training)
training$p3= predict(m3, training)

testing$p1= predict(m1, testing)
testing$p2= predict(m2, testing)
testing$p3= predict(m3, testing)


mm= train(diagnosis ~ p1 + p2 + p3, data= training, method= 'rf')

preds0= predict(m1, testing)
preds1= predict(m2, testing)
preds2= predict(m3, testing)
preds3= predict(mm, testing)

confusionMatrix(preds0, testing$diagnosis)
confusionMatrix(preds1, testing$diagnosis)
confusionMatrix(preds2, testing$diagnosis)
confusionMatrix(preds3, testing$diagnosis)




# q3
set.seed(3523)
library(AppliedPredictiveModeling)
data(concrete)
inTrain = createDataPartition(concrete$CompressiveStrength, p = 3/4)[[1]]
training = concrete[ inTrain,]
testing = concrete[-inTrain,]

set.seed(233)
ls= train(CompressiveStrength ~ . , data= training, method= 'lasso')
plot.enet

# q5
set.seed(325)
library(e1071)
m= svm(CompressiveStrength ~ . , data= training)
testing$preds= predict(m, testing)
testing$err= testing$CompressiveStrength - testing$preds
testing$sq.err= testing$err ^ 2
mean(testing$sq.err) ^ 0.5




# q4
library(lubridate)  # For year() function below
dat = read.csv("~/Downloads/gaData.csv")
training = dat[year(dat$date) < 2012,]
testing = dat[(year(dat$date)) > 2011,]
tstrain = ts(training$visitsTumblr)

library(forecast)
m= bats(tstrain)
ff= forecast(m, level= 95, h= nrow(testing))
dx= data.table(up= ff$upper, low= ff$lower, actual= testing$visitsTumblr)
dx[, ifcont:= ifelse(actual <= up.V1 & actual >= low.V1, 1, 0)]
sum(dx$ifcont)/nrow(dx)
