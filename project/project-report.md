---
title: "Physical Exercise Activity Prediction in R"
author: "Arkosnato Neogy"
date: "26 July 2015"
output: html_document
---


```
## Loading required package: methods
```

In this document we describe the steps to performing a classification modeling exercise. The goal of the model is to predict the nature of physical activity being performed by a subject. The training data in question contains a label variable `classe` which specifies a certain kind of activity, along with a set of accelerometer and gyroscope measurements from different body parts of the subject (along with variables derived from raw measurements). The training dataset comprises about 19k rows and 160 variables.


```r
training= fread('pml-training.csv')
final.testing= fread('pml-testing.csv')
dim(training)
```

```
## [1] 19622   160
```

A quick look at the data reveals that the column named V1 is simply an index so we remove this column right away. Following this we perform some basic preprocessing with the primary objective of reducing variables that have no utility in the data. As a starting step, we first take a look at the data, and observe that many variables have NA values in them. We first remove variables that have mostly NAs in them. 


```r
irrel.cols= 'V1'
rel.cols= setdiff(names(training), irrel.cols)
training= training[, rel.cols, with=F]

na.cols= names(training)[training[, lapply(.SD, function(x){sum(is.na(x))}), .SDcols= names(training)]>19000]
non.na.cols= setdiff(names(training), na.cols)
training = training[,non.na.cols,with=F]
dim(training)
```

```
## [1] 19622    92
```

Secondly, we check variables that have zero or near zero variance. Such variables are also removed from the modeling exercise. 


```r
library(caret)
```

```
## Loading required package: lattice
```

```r
nzv= nearZeroVar(training, saveMetrics = T)
nzv.cols= rownames(nzv)[nzv$nzv==T]
non.nzv.cols= setdiff(names(training), nzv.cols)
training = training[, non.nzv.cols, with=F]
dim(training)
```

```
## [1] 19622    58
```

This leaves us with about 60 columns. This is a fairly good reduction in the number of variables from where we started, so we proceed to the training exercise at this stage. We setup the modeling data using the data partitioning function in the 'caret' package as follows.


```r
set.seed(12936)
training[, classe:= as.factor(classe)]
intrain= as.vector(createDataPartition(training$classe, p= 0.7, list=F))
mytrain= training[intrain,]
mytest= training[-intrain,]
```

We are now ready to train our model. We seek to use the model tuning capabilities of caret on the fly, so to speed up the process we make use of multicore parallel processing. Having set up the number of cores, we train a RF on the training data partition using the default tuning grid used by caret. We use a 5-fold cross validation to pick the optimal model. Doing so gives us an optimized model which is displayed as an output.



```r
library(doMC)
```

```
## Loading required package: foreach
## Loading required package: iterators
## Loading required package: parallel
```

```r
registerDoMC(2)

dim(mytrain)
```

```
## [1] 13737    58
```

```r
ctrl.rf= trainControl(method= 'cv', number= 5, classProbs = TRUE)
rf.model= train(classe ~ ., data= mytrain, method= 'rf', trControl= ctrl.rf, allowParallel= T)
```

```
## Loading required package: randomForest
## randomForest 4.6-10
## Type rfNews() to see new features/changes/bug fixes.
```

```r
print(rf.model$finalModel)
```

```
## 
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry, allowParallel = ..1) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 40
## 
##         OOB estimate of  error rate: 0.08%
## Confusion matrix:
##      A    B    C    D    E  class.error
## A 3906    0    0    0    0 0.0000000000
## B    2 2655    1    0    0 0.0011286682
## C    0    3 2392    1    0 0.0016694491
## D    0    0    2 2249    1 0.0013321492
## E    0    0    0    1 2524 0.0003960396
```

We see that the optimal Random Forest model used 40 variables and 500 trees.

The performance of the model is checked on both the train and the test as follows:


```r
confusionMatrix(predict(rf.model, mytrain), training[intrain,]$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 3906    0    0    0    0
##          B    0 2658    0    0    0
##          C    0    0 2396    0    0
##          D    0    0    0 2252    0
##          E    0    0    0    0 2525
## 
## Overall Statistics
##                                      
##                Accuracy : 1          
##                  95% CI : (0.9997, 1)
##     No Information Rate : 0.2843     
##     P-Value [Acc > NIR] : < 2.2e-16  
##                                      
##                   Kappa : 1          
##  Mcnemar's Test P-Value : NA         
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   1.0000   1.0000   1.0000   1.0000
## Specificity            1.0000   1.0000   1.0000   1.0000   1.0000
## Pos Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Neg Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Prevalence             0.2843   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2843   0.1935   0.1744   0.1639   0.1838
## Detection Prevalence   0.2843   0.1935   0.1744   0.1639   0.1838
## Balanced Accuracy      1.0000   1.0000   1.0000   1.0000   1.0000
```

```r
confusionMatrix(predict(rf.model, mytest), mytest$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1674    0    0    0    0
##          B    0 1139    0    0    0
##          C    0    0 1026    1    0
##          D    0    0    0  962    0
##          E    0    0    0    1 1082
## 
## Overall Statistics
##                                      
##                Accuracy : 0.9997     
##                  95% CI : (0.9988, 1)
##     No Information Rate : 0.2845     
##     P-Value [Acc > NIR] : < 2.2e-16  
##                                      
##                   Kappa : 0.9996     
##  Mcnemar's Test P-Value : NA         
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   1.0000   1.0000   0.9979   1.0000
## Specificity            1.0000   1.0000   0.9998   1.0000   0.9998
## Pos Pred Value         1.0000   1.0000   0.9990   1.0000   0.9991
## Neg Pred Value         1.0000   1.0000   1.0000   0.9996   1.0000
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2845   0.1935   0.1743   0.1635   0.1839
## Detection Prevalence   0.2845   0.1935   0.1745   0.1635   0.1840
## Balanced Accuracy      1.0000   1.0000   0.9999   0.9990   0.9999
```

The model performance is quite impressive, giving an accuracy of almost 1 on the test dataset!

We use the above model to produce our final predictions needed for the submission:


```r
myfinalpreds = predict(rf.model, final.testing)
myfinalpreds
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```
