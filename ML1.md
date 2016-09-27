---
title: "Machine Learning WriteUp"
author: "Eric L. Eakin"
date: "September 20, 2016"
output: 
  html_document: 
    keep_md: yes
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
```

### Overview

#### This project is the final assignment for Coursera’s Practical Machine Learning unit. This assignment is meant to augment the course quiz. The Machine Learning algorithm will be applied to twenty quiz questions. 

### Dataset Overview

#### The data for this project is from http://groupware.les.inf.puc-rio.br/har.

##### Full source: Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. “Qualitative Activity Recognition of Weight Lifting Exercises. Proceedings of 4th International Conference in Cooperation with SIGCHI (Augmented Human ’13)”. Stuttgart, Germany: ACM SIGCHI, 2013.

#### Load Packages/Libraries

```{r, echo= FALSE}
library(knitr)
library(plyr)
library(caret)
library(rpart)
library(rpart.plot)
library(rattle)
library(randomForest)
library(corrplot)
library(gbm)
set.seed(12345)
```

#### Data Loading and Cleaning

```{r}
UrlTrain <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
```

```{r}
UrlTest  <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
```

```{r}
training <- read.csv(url(UrlTrain))
```

```{r}
testing  <- read.csv(url(UrlTest))
```

##### Create a partition with the training dataset 

```{r}
inTrain  <- createDataPartition(training$classe, p=0.7, list=FALSE)
TrainSet <- training[inTrain, ]
TestSet  <- training[-inTrain, ]
```

```{r, results='hide'}
dim(TrainSet)
dim(TestSet)
```

#### The two datasets have 160 variables. We need to remove the 'NA' placeholders and 'near zero variance'.

###### Remove variables with Nearly Zero Variance

```{r}
NZV <- nearZeroVar(TrainSet)
TrainSet <- TrainSet[, -NZV]
TestSet  <- TestSet[, -NZV]
```

```{r, results='hide'}
dim(TrainSet)
dim(TestSet)
```

##### Remove NA placeholders

```{r}
AllNA    <- sapply(TrainSet, function(x) mean(is.na(x))) > 0.95
TrainSet <- TrainSet[, AllNA==FALSE]
TestSet  <- TestSet[, AllNA==FALSE]
```

```{r, results= 'hide'}
dim(TrainSet)
dim(TestSet)
```

##### Remove identification only variables.

```{r}
TrainSet <- TrainSet[, -(1:5)]
TestSet  <- TestSet[, -(1:5)]
```

```{r }
dim(TrainSet)
dim(TestSet)
```

##### With the cleaning process above, the number of variables for the analysis has been reduced to 54.

#### Prediction Model Building

#### We will use three models for regression in the ‘train’ dataset. The three methods are:
1. Random Forests
2. Decision Tree
3. Generalized Boosted Model

##### A Confusion Matrix is plotted at the end of each analysis to visualize the accuracy of the models.

### Random Forest Method

```{r}
set.seed(12345)
controlRF <- trainControl(method="cv", number=3, verboseIter=FALSE)
modFitRandForest <- train(classe ~ ., data=TrainSet, method="rf",
        trControl=controlRF)
```

```{r}
modFitRandForest$finalModel
```

##### Prediction on Test dataset

```{r}
predictRandForest <- predict(modFitRandForest, newdata=TestSet)
confMatRandForest <- confusionMatrix(predictRandForest, TestSet$classe)
confMatRandForest
```

##### Plot matrix results

```{r, fig.width= 5, fig.height= 5}
plot(confMatRandForest$table, col = confMatRandForest$byClass, 
     main = paste("Random Forest - Accuracy =",
                  round(confMatRandForest$overall['Accuracy'], 4)))
```

### Method: Decision Tree Method

```{r}
set.seed(12345)
modFitDecTree <- rpart(classe ~ ., data=TrainSet, method="class")
fancyRpartPlot(modFitDecTree)
```

#### Prediction on Test dataset

```{r}
predictDecTree <- predict(modFitDecTree, newdata=TestSet, type="class")
confMatDecTree <- confusionMatrix(predictDecTree, TestSet$classe)
confMatDecTree
```

#### Plot matrix results

```{r, fig.width= 5, fig.height= 5}
plot(confMatDecTree$table, col = confMatDecTree$byClass, 
     main = paste("Decision Tree - Accuracy =",
                  round(confMatDecTree$overall['Accuracy'], 4)))
```

#### Method: Generalized Boosted Model

```{r}
set.seed(12345)
controlGBM <- trainControl(method = "repeatedcv", number = 5, repeats = 1)
modFitGBM  <- train(classe ~ ., data=TrainSet, method = "gbm",
                    trControl = controlGBM, verbose = FALSE)
```

```{r}
modFitGBM$finalModel
```


##### Prediction on Test dataset

```{r}
predictGBM <- predict(modFitGBM, newdata=TestSet)
confMatGBM <- confusionMatrix(predictGBM, TestSet$classe)
confMatGBM
```

#### Plot matrix results

```{r, fig.width= 5, fig.height= 5}
plot(confMatGBM$table, col = confMatGBM$byClass, 
     main = paste("GBM - Accuracy =", round(confMatGBM$overall['Accuracy'], 4)))
```

### The accuracy/results of the three modeling methods above are:
1. Random Forest : 0.9964
2. Decision Tree : 0.7368
3. GBM : 0.9859

#### In this exercise the Random Forest model will be applied to predict the 20 quiz results (testing dataset) as shown below.

```{r}
predictTEST <- predict(modFitRandForest, newdata=testing)
predictTEST
```

(c) Eric L. Eakin, 2016
