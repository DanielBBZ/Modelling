rm(list=ls(all=TRUE))
gc(reset=T)

## put the required packages here
#install.packages("data.table","bit74","caret","gbm")
require(data.table)
require(bit64)
require(caret)
require(gbm)
require(ggplot2)

library(xgboost) # for xgboost
library(tidyverse) # general utility functions

## set the working directory to the path that contains all the data files:
# - train_v2.csv
# - test_v2.csv
# - sampleSubmission.csv
# - Defaulter_features.RData
# - LGD_features.RData
setwd('C:/Users/daniel/Documents/Kaggle/cardbalance')

# read in our data & put it in a data frame
credit_df <- read_csv("./Credit.csv")
head(credit_df)
str(credit_df)

hist(credit_df$Balance, breaks=c(0, 100, 300, 500, 700, 1000, 1500, 2000))


# the user active statas
credit_df$active <- ifelse(credit_df$Balance>0, 1, 0)

credit_df$active


#another way to convert to numeric
cols = c(3,4,5,6,7,12) 
credit_df[,cols] = apply(credit_df[,cols], 2, function(x) as.numeric(as.character(x)))

# one-hot matrix for just the first few rows of the "country" column
model.matrix(~Ethnicity-1,head(credit_df))

# convert categorical factor into one-hot encoding
region <- model.matrix(~Ethnicity-1,credit_df)

ifmale <- ifelse(credit_df$Gender=="Male", 1, 0)
ifstudent <- ifelse(credit_df$Student == "Yes", 1, 0)
ifmarried <- ifelse(credit_df$Married == "Yes", 1, 0)


# select just the numeric columns
credit_df_numeric <- credit_df %>%
  select(-X1) %>% # the id shouldn't contain useful information
  select_if(is.numeric) # select remaining numeric columns


# add our one-hot encoded variable and convert the dataframe into a matrix
credit_df <- cbind(credit_df_numeric, region, ifmale, ifstudent, ifmarried)
credit_df <- subset(credit_df,select=-c(Balance,active))
credit_matrix <- data.matrix(credit_df)

#head(credit_df)

# get a boolean vector of training labels
ActiveLabels <- ifelse(credit_df$active== "1", TRUE, FALSE)

BalanceValue <- credit_df$Balance



#Split data


# get the numb 60/40 training test split
numberOfTrainingSamples <- round(length(ActiveLabels) * .6)

# training data
train_data <- credit_matrix[1:numberOfTrainingSamples,]
train_acive_labels <- ActiveLabels[1:numberOfTrainingSamples]
train_BalanceValue <- BalanceValue[1:numberOfTrainingSamples]

# testing data
test_data <- credit_matrix[-(1:numberOfTrainingSamples),]
test_active_labels <- ActiveLabels[-(1:numberOfTrainingSamples)]
test_BalanceValue <- BalanceValue[-(1:numberOfTrainingSamples)]



#Convert the cleaned dataframe to a dmatrix
# put our testing & training data into two seperates Dmatrixs objects
dtrain <- xgb.DMatrix(data = train_data, label= train_acive_labels)
dtest <- xgb.DMatrix(data = test_data, label= test_active_labels)


# train a model using our training data
model <- xgboost(data = dtrain, # the data   
                 nround = 7, # max number of boosting iterations
                 objective = "binary:logistic")  # the objective function


# generate predictions for our held-out testing data
pred <- predict(model, dtest)

# get & print the classification error
err <- mean(as.numeric(pred > 0.45) != test_active_labels)
print(paste("test-error=", err))




# get the number of negative & positive cases in our data
negative_cases <- sum(train_acive_labels == FALSE)
postive_cases <- sum(train_acive_labels == TRUE)



#Tuning our model
# train an xgboost model
model_tuned <- xgboost(data = dtrain, # the data           
                       max.depth = 5, # the maximum depth of each decision tree, default is 6
                       nround = 3, # max number of boosting iterations
                       objective = "binary:logistic") # the objective function 

# generate predictions for our held-out testing data
pred <- predict(model_tuned, dtest)

# get & print the classification error
err <- mean(as.numeric(pred > 0.5) != test_active_labels)
print(paste("test-error=", err))


# scale the imbalance data
# get the number of negative & positive cases in our data
negative_cases <- sum(train_labels == FALSE)
postive_cases <- sum(train_labels == TRUE)

# train a model using our training data
model_tuned <- xgboost(data = dtrain, # the data           
                       max.depth = 3, # the maximum depth of each decision tree
                       nround = 10, # number of boosting rounds
                       early_stopping_rounds = 3, # if we dont see an improvement in this many rounds, stop
                       objective = "binary:logistic", # the objective function
                       scale_pos_weight = negative_cases/postive_cases) # control for imbalanced classes

# generate predictions for our held-out testing data
pred <- predict(model_tuned, dtest)

# get & print the classification error
err <- mean(as.numeric(pred > 0.5) != test_labels)
print(paste("test-error=", err))

# can see overfitting  based on the result


#Gamma is a measure of how much an additional split will need to reduce loss in 
#order to be added to the ensemble. If a proposed model does not reduce loss by 
#at least whatever-you-set-gamma-to, it won't be included. Here, I'll set it to 
#one, which is fairly high. (By default gamma is zero.)

# train a model using our training data
model_tuned <- xgboost(data = dtrain, # the data           
                       max.depth = 3, # the maximum depth of each decision tree
                       nround = 10, # number of boosting rounds
                       early_stopping_rounds = 3, # if we dont see an improvement in this many rounds, stop
                       objective = "binary:logistic", # the objective function
                       scale_pos_weight = negative_cases/postive_cases, # control for imbalanced classes
                       gamma = 1) # add a regularization term

# generate predictions for our held-out testing data
pred <- predict(model_tuned, dtest)

# get & print the classification error
err <- mean(as.numeric(pred > 0.5) != test_labels)
print(paste("test-error=", err))


#Examining our model
# plot them features! what's contributing most to our model?
install.packages('DiagrammeR')
xgb.plot.multi.trees(feature_names = names(credit_df_numeric), 
                     model = model)

# convert log odds to probability
odds_to_probs <- function(odds){
  return(exp(odds)/ (1 + exp(odds)))
}


# probability of leaf above countryPortugul
odds_to_probs(-0.599)


# get information on how important each feature is
importance_matrix <- xgb.importance(names(credit_df_numeric), model = model)

# and plot it!
xgb.plot.importance(importance_matrix)
