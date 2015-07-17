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
model1= train(training.diagnosis ~ . , data= s1, method= 'glm')
preds1= predict(model1, testing)
confusionMatrix(preds1, testing$diagnosis)

x= prcomp(ss, center = T, scale. = T)
y= x$sdev ^2
yy= y/sum(y)
plot(cumsum(yy), type= 'l')

pcaModel= preProcess(ss, method= 'pca', pcaComp = 7)
training.pca= data.frame(predict(pcaModel, ss), training$diagnosis)
testing.pca= data.frame(predict(pcaModel, sx), testing$diagnosis)
model2= train(training.diagnosis ~ . , data= training.pca, method= 'glm')
preds2= predict(model2, testing.pca)
confusionMatrix(preds2, testing.pca$testing.diagnosis)

ctrl= trainControl(preProcOptions = list(thresh= 0.8))
model3= train(training.diagnosis ~ . , data= s1, method= 'glm', preProcess= 'pca', trControl= ctrl)
preds3= predict(model3, sx)
confusionMatrix(preds3, testing$diagnosis)



# q2
data(concrete)
set.seed(1000)
inTrain = createDataPartition(mixtures$CompressiveStrength, p = 3/4)[[1]]
training = mixtures[ inTrain,]
testing = mixtures[-inTrain,]
hist(training$Superplasticizer)
qplot(Superplasticizer, data= training, geom= 'density')
summary(training$Superplasticizer)

