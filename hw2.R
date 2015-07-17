library(caret)
library(AppliedPredictiveModeling)


# q1,3,4
data(AlzheimerDisease)
set.seed(3433)
adData = data.frame(diagnosis,predictors)
trainIndex = createDataPartition(diagnosis, p = 0.750, list= F)
training = adData[trainIndex,]
testing = adData[-trainIndex,]
ss= training[,58:69]
sx= testing[,58:69]

s1= data.frame(ss, training$diagnosis)
model= train(training.diagnosis ~ . , data= s1, method= 'glm')
preds= predict(model, testing)
confusionMatrix(preds, testing$diagnosis)

x= prcomp(ss, center = T, scale. = T)
y= x$sdev ^2
yy= y/sum(y)
plot(cumsum(yy), type= 'l')
z= prcomp(sx, center = T, scale. = T)

s2= data.frame(x$x[,1:7], training$diagnosis)
s3= data.frame(z$x[,1:7], testing$diagnosis)
model= train(training.diagnosis ~ . , data= s2, method= 'glm')
preds= predict(model, s3)
confusionMatrix(preds, s3$testing.diagnosis)

# q2
data(concrete)
set.seed(1000)
inTrain = createDataPartition(mixtures$CompressiveStrength, p = 3/4)[[1]]
training = mixtures[ inTrain,]
testing = mixtures[-inTrain,]
hist(training$Superplasticizer)
qplot(Superplasticizer, data= training, geom= 'density')
summary(training$Superplasticizer)

