#Breast Cancer Diagnosis Model using Breast Cancer Wisconsin (Diagnosis) dataset


#libraries
library(tidyverse)
library(ggcorrplot)
library(caret)
library(factoextra)
#load data into vector
wbrc <- read.csv("./data.csv", header = T, stringsAsFactors = F)

#remove nulls

wbrc$X <- NULL

#remove id col
wbrc <- wbrc[,-1]
wbrc$diagnosis <- as.factor(wbrc$diagnosis)

# near zero variance check on predictors
nzv <- nearZeroVar(wbrc[,-1])
nzv

#review data summaries

str(wbrc)
summary(wbrc)


#analyze correlations for all 

wbrc_corall <- cor(wbrc[,-1])
p.matall <- cor_pmat(wbrc_corall)
ggcorrplot(wbrc_corall, p.mat = p.matall, hc.order = TRUE, type = "lower", insig = "blank")

# analyze correlations separately for all, mean, sd, and worst columns to improve visual

#mean
wbrc_cormean <- cor(wbrc[,c(2:11)])
p.matmean <- cor_pmat(wbrc_cormean)
ggcorrplot(wbrc_cormean, p.mat = p.matmean, hc.order = TRUE, type = "lower", insig = "blank")
#sd
wbrc_corsd <- cor(wbrc[,c(12:21)])
p.matsd <- cor_pmat(wbrc_corsd)
ggcorrplot(wbrc_corsd, p.mat= p.matsd, hc.order = TRUE, type = "lower", insig = "blank")
#worst
wbrc_corworst <- cor(wbrc[,c(22:31)])
p.matworst <- cor_pmat(wbrc_corworst)
ggcorrplot(wbrc_corworst, p.mat = p.matworst, hc.order = TRUE, type = "lower", insig = "blank")


#Principal Component Analysis
# numerous highly correlated variables found.  Let's reduce the number of variables with PCA to increase efficiency of model

#wbrc_pca <- transform(wbrc)
wbrc_all_pca <- prcomp(wbrc[,-1], cor=TRUE, scale = TRUE)
summary(wbrc_all_pca)

# test set will be 40% of Wisconsin Breast Cancer (wbrc) data
set.seed(1)
test_index <- createDataPartition(y = wbrc$diagnosis, times = 1, p = 0.4, list = FALSE)
train <- wbrc[-test_index,]
#train$diagnosis <- as.factor(train$diagnosis)
test <- wbrc[test_index,]
#test$diagnosis <- as.factor(test$diagnosis)

#check proportion of benign/malignant diagnosis in train/test sets

prop.table(table(train$diagnosis))
prop.table(table(test$diagnosis))

# start modeling

# using rpart for decision trees

library(rpart)
fit_rp <- rpart(diagnosis~.,data=train,control=rpart.control(minsplit=2))
predict_rp <- predict(fit_rp, test[,-1], type="class")
cm_rp  <- confusionMatrix(predict_rp, test$diagnosis)   
cm_rp


#  using randomforest

library(randomForest)
fit_rf <- randomForest(diagnosis~., data=train, proximity=TRUE, importance=TRUE)
predict_rf   <- predict(fit_rf, test[,-1])
cm_rf    <- confusionMatrix(predict_rf, test$diagnosis)
cm_rf


#fit_RFmtry <- train(diagnosis~., method = "rf", data=train,
#                  nodesize = 1,
#                  tuneGrid = data.frame(mtry = seq(6, 30, 2)))
#predict_RFmtry <- predict(fit_RFmtry, test[,-1])
#cm_RFmtry <-confusionMatrix(predict_RFmtry, test$diagnosis)

# using SVM,  method="svmLinear"

fit_svm <- train(method="svmLinear", diagnosis~., data=train)
predict_svm <- predict(fit_svm, test[,-1])
cm_svm <- confusionMatrix(predict_svm, test$diagnosis)
cm_svm

# using SVM, method="svmRadial"


fit_svm_Rad <- train(method="svmRadial", diagnosis~., data=train)
predict_svm_Rad <- predict(fit_svm_Rad, test[,-1])
cm_svm_Rad <- confusionMatrix(predict_svm_Rad, test$diagnosis)
cm_svm_Rad
# other svm radial methods tried but no improvement above svm Radial
fit_svm_RadC <- train(method="svmRadialCost", diagnosis~., data=train)
predict_svm_RadC <- predict(fit_svm_RadC, test[,-1])
cm_svm_RadC <- confusionMatrix(predict_svm_RadC, test$diagnosis)
cm_svm_RadC

#http://topepo.github.io/caret/train-models-by-tag.html#Support_Vector_Machines
#Notes: This SVM model tunes over the cost parameter and the RBF kernel parameter sigma. 
#In the latter case, using tuneLength will, at most, evaluate six values of the kernel parameter. 
#This enables a broad search over the cost parameter and a relatively narrow search over sigma


fit_svm_RadSig <- train(method="svmRadialSigma", diagnosis~., data=train)
predict_svm_RadSig <- predict(fit_svm_RadSig, test[,-1])
cm_svm_RadSig <- confusionMatrix(predict_svm_RadSig, test$diagnosis)
cm_svm_RadSig

#  full dataset prediciton with svmRadial as the best model based on Balanced Accuracy.

predict_svm_Radial_wbrc <- predict(fit_svm_Rad, wbrc[,-1])
cm_svm_Radial_wbrc <- confusionMatrix(predict_svm_Radial_wbrc, wbrc$diagnosis)
cm_svm_Radial_wbrc

# add prediction column with wbrc data frame and create wbrc_predicts dataset to compare predicition and diagnosis values. 
# Look for any predictor patterns in the wrong predictions.
wbrc_predicts <- cbind(prediction = predict_svm_Radial_wbrc, wbrc)

ind <- which(wbrc_predicts$prediction != wbrc_predicts$diagnosis)
wbrc_predicts[ind,]
