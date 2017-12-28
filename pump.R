library(caret)
library(data.table)
library(dplyr)
library(mice)
library(xgboost)

# Import test and train set
train <- fread("train_features.csv", stringsAsFactors = TRUE)
train_labels <- fread("train_labels.csv", stringsAsFactors = TRUE)
test <- fread("test.csv", stringsAsFactors = TRUE )


#merge everything
train_1 <- bind_cols(train, train_labels)
comb <- bind_rows(train_1, test)
#transform to data frame only
comb <- as.data.frame(comb)
#cleaning part 1, drop features with too many levels or just 1

comb$funder <- NULL
comb$wpt_name <- NULL
comb$subvillage <- NULL
comb$recorded_by <- NULL
comb$date_recorded <- NULL  #it doesn't carry any information
comb$longitude <- NULL
comb$latitude <- NULL
comb$num_private <- NULL

#continue to convert chr to factors
comb$installer <- as.factor(comb$installer)
comb$ward      <- as.factor(comb$ward)
comb$scheme_management <- as.factor(comb$scheme_management)
comb$scheme_name      <- as.factor(comb$scheme_name)
comb$extraction_type  <- as.factor(comb$extraction_type)
#convert boolean to dummy 
comb$public_meeting <- as.integer(comb$public_meeting)
comb$permit         <- as.integer(comb$permit)
#drop Id1 
comb$id1 <- NULL
#take a look at the correlations 
library(GGally)
ggcorr(comb)



#convert factor to numbers
comb$installer <- as.numeric(comb$installer)-1
comb$basin     <- as.numeric(comb$basin)-1
comb$region    <- as.numeric(comb$region)-1
comb$lga       <- as.numeric(comb$lga)-1
comb$ward      <- as.numeric(comb$ward)-1
comb$scheme_management <- as.numeric(comb$scheme_management)-1
comb$scheme_name   <- as.numeric(comb$scheme_name)-1
comb$extraction_type <- as.numeric(comb$extraction_type)-1
comb$extraction_type_group <- as.numeric(comb$extraction_type_group)-1
comb$extraction_type_class <- as.numeric(comb$extraction_type_class)-1
comb$management         <- as.numeric(comb$management)
comb$management_group <- as.numeric(comb$management_group)
comb$payment      <-as.numeric(comb$payment)-1
comb$payment_type  <- as.numeric(comb$payment_type)-1
comb$water_quality <- as.numeric(comb$water_quality)-1
comb$quality_group <- as.numeric(comb$quality_group)-1
comb$quantity     <- as.numeric(comb$quantity)-1
comb$quantity_group <- as.numeric(comb$quantity_group)-1
comb$source        <- as.numeric(comb$source)-1
comb$source_type   <- as.numeric(comb$source_type)-1
comb$source_class  <- as.numeric(comb$source_class)-1
comb$waterpoint_type <- as.numeric(comb$waterpoint_type)-1
comb$waterpoint_type_group <- as.numeric(comb$waterpoint_type_group)-1


#check NA
sapply(comb, function(x) sum(is.na(x)))
#permit and public_meeting have fair NA, perform imputation 
comb_train <- comb[1:59400,]
comb_test  <- comb[59401:74250,]
#check for NA , 
sapply(comb_train, function(x) sum(is.na(x)))
mice_imp <- mice(comb_train, method = "cart", m=1)
densityplot(mice_imp, ~public_meeting)
densityplot(mice_imp, ~permit)

imputed <- complete(mice_imp)
############impute missing values in the dataset###########
comb_train$public_meeting <- imputed$public_meeting
comb_train$permit         <- imputed$permit
#########merge again##############
comb <- bind_rows(comb_train, comb_test)

#take a look at the correlations 
library(GGally)
ggcorr(comb)

#create the most obvious feature: how old is the well? 
comb$logOldWell <- log1p(2017 - comb$construction_year)

comb2 <- comb
#check for Near zero predictors
# delete near zero predictors IT IMPROVED LB
nzv.data <- nearZeroVar(comb2, saveMetrics = TRUE)
# take any of the near-zero-variance perdictors
drop.cols <- rownames(nzv.data)[nzv.data$nzv == TRUE]
combine2 <- as.data.frame(combine2)
combine2 <- combine2[,!names(combine2) %in% drop.cols]
combine2 <- setDT(combine2)
##############
status<- comb$status_group
comb$status_group <- NULL
comb$status_group <- status


#split again
train_2 <- comb[1:59400,]  #1460
#train_2 <- train_temp2
test_2  <- comb[59401:74250,] #1461:2919
#generate train label 

train.label <- train_2$status_group
#train.label <- train_temp2$status_group
#train.label <- train_temp2$status_group
test.label <- test_2$status_group
#convert dataset to matrix
train_2<- as.matrix(train_2[,2:32])
#train_2<- as.matrix(train_temp2[,2:32])
test_2<- as.matrix(test_2[,2:32])


#dtrain <- xgb.DMatrix(data = train_2, label = train.label)
dtrain <- xgb.DMatrix(data = train_2, label = train.label)
dtest  <- xgb.DMatrix(data = test_2, label = test.label)


set.seed(1234)
# Set our hyperparameters
param <- list(objective   = "multi:softmax",
              num_class = 4,
              eval_metric = "merror",
              max_depth   = 8, #10
              subsample   = 0.399,
              eta         = 0.0107,
              gammma      = 0,  #1
              colsample_bytree = 0.47,
              min_child_weight = 4)



cvFolds <- createFolds(comb$status_group[!is.na(comb$status_group)], k=5, list=TRUE, returnTrain=FALSE)

xgb_cv <- xgb.cv(data = dtrain,
                 params = param,
                 nrounds = 3000,
                 maximize = FALSE,
                 prediction = TRUE,
                 stratified = TRUE,
                 folds = cvFolds,
                 print_every_n = 10,
                 early_stopping_round = 50)

best_iter<- xgb_cv$best_iteration

# Pass in our hyperparameteres and train the model 
system.time(xgb <- xgboost(params  = param,
                           data    = dtrain,
                           label   = train.label, 
                           nrounds = best_iter,
                           print_every_n = 100,
                           verbose = 1))

system.time(xgb2 <- xgboost(params  = param,
                           data    = dtrain,
                           label   = train.label, 
                           nrounds = best_iter,
                           print_every_n = 100,
                           seed = 12345,
                           verbose = 1))

system.time(xgb3 <- xgboost(params  = param,
                           data    = dtrain,
                           label   = train.label, 
                           nrounds = best_iter,
                           print_every_n = 100,
                           seed= 12346,
                           verbose = 1))

system.time(xgb4 <- xgboost(params  = param,
                           data    = dtrain,
                           label   = train.label, 
                           nrounds = best_iter,
                           print_every_n = 100,
                           seed= 12347,
                           verbose = 1))

names <- dimnames(dtrain)[[2]]

# Compute feature importance matrix
importance_matrix <- xgb.importance(names, model=xgb)[0:20] # View top 20 most important features

# Plot
xgb.plot.importance(importance_matrix)


# Prediction on test and train sets
pred_xgboost_test <- predict(xgb, dtest)
pred_xgboost_train <- predict(xgb, dtrain)
pred_xgboost_test2 <- predict(xgb2, dtest)
pred_xgboost_test3 <- predict(xgb3, dtest)
pred_xgboost_test4 <- predict(xgb4, dtest)

pred_ensamble <- (pred_xgboost_test + pred_xgboost_test2 + pred_xgboost_test3 + pred_xgboost_test4)/4
pred_ensamble <- round(pred_ensamble)
submit <- data.frame(id = comb[59401:74250,c("id")], status_group = pred_ensamble)
library(plyr)
submit$status_group <- as.factor(submit$status_group)
submit$status_group<-submit$status_group<-revalue(submit$status_group, c("1"="functional", "2"="needs repair", "3"="non functional"))

write.csv(submit, file = "pump1.csv", row.names = FALSE)






