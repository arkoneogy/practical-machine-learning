library(data.table)
library(ggplot2)
library(caret)

training= fread('pml-training.csv')
final.testing= fread('pml-testing.csv')

dim(training)
names(training)
irrel.cols= 'V1'
rel.cols= setdiff(names(training), irrel.cols)
training= training[, rel.cols, with=F]

# basic preprocessing
na.cols= names(training)[training[, lapply(.SD, function(x){sum(is.na(x))}), .SDcols= names(training)]>19000]
non.na.cols= setdiff(names(training), na.cols)
training = training[,non.na.cols,with=F]
dim(training)

nzv= nearZeroVar(training, saveMetrics = T)
nzv.cols= rownames(nzv)[nzv$nzv==T]
non.nzv.cols= setdiff(names(training), nzv.cols)
training = training[, non.nzv.cols, with=F]


dim(training)
names(training)


# split train and test
set.seed(12936)
training[, classe:= as.factor(classe)]
intrain= as.vector(createDataPartition(training$classe, p= 0.7, list=F))
mytrain= training[intrain,]
mytest= training[-intrain,]

dim(mytrain)
dim(mytest)


# train  RF model with caret by 5 fold CV
library(doMC)
registerDoMC(2)

ctrl.rf= trainControl(method= 'cv', number= 5, classProbs = TRUE)
rf.model= train(classe ~ ., data= mytrain, method= 'rf', trControl= ctrl.rf, allowParallel= T)

print(rf.model)
print(rf.model$finalModel)

confusionMatrix(predict(rf.model, mytrain), training[intrain,]$classe)
confusionMatrix(predict(rf.model, mytest), mytest$classe)

saveRDS(rf.model, 'project_rfmodel.RDS')


# final predictions
myfinalpreds = predict(rf.model, final.testing)
myfinalpreds= as.character(myfinalpreds)

# prepare submission files
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}

pml_write_files(myfinalpreds)
