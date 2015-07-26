library(data.table)
library(AppliedPredictiveModeling)
library(ElemStatLearn)
library(caret)


#Q1
data(segmentationOriginal)
segmentationOriginal= data.table(segmentationOriginal)
# intrain= createDataPartition(y= segmentationOriginal$Case, p= 0.7, list=F)
training= segmentationOriginal[Case=='Train']
testing= segmentationOriginal[Case=='Test']
set.seed(125)
m1= train(Class ~., method= 'rpart', data= training)
plot(m1$finalModel, uniform=T)
text(m1$finalModel, use.n=T, all=T, cex= 0.8)


# Q3
library(pgmm)
data(olive)
olive = olive[,-1]
m2= train(Area ~ . , data= olive, method= 'rpart')
newdt= as.data.frame(t(colMeans(olive)))
predict(m2, newdata = newdt)


# Q4
data(SAheart)
set.seed(8484)
train = sample(1:dim(SAheart)[1],size=dim(SAheart)[1]/2,replace=F)
trainSA = SAheart[train,]
testSA = SAheart[-train,]
missClass = function(values,prediction){
  sum(((prediction > 0.5)*1) != values)/length(values)
}
set.seed(13234)
m3= train(chd ~ age + alcohol + obesity + tobacco + typea + ldl, data= trainSA, 
          method= 'glm', family= 'binomial')
preds1= predict(m3, trainSA)
preds2= predict(m3, testSA)
missClass(trainSA$chd, preds1)
missClass(testSA$chd, preds2)


# Q5
data(vowel.train)
data(vowel.test) 
vowel.train$y= as.factor(vowel.train$y)
vowel.test$y= as.factor(vowel.test$y)
set.seed(33833)
m4= train(y ~ ., method= 'rf', data= vowel.train)
x= varImp(m4$finalModel)
data.table(var= rownames(x), imp= x$Overall)[order(-imp)]
