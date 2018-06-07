# libraries we'll need
library(tidyverse) # utility functions
library(caret) # hyperparameter tuning
library(randomForest) # for our model
library(Metrics) # handy evaluation functions

setwd("C:/Users/daniel/Documents/Kaggle/PUBG")

# read in our data
player_statistics <- read_csv("./PUBG_Player_Statistics.csv")

# only information on solo plays
player_statistics <- player_statistics %>% select(starts_with("solo"))

# check out the first few rows
head(player_statistics)


# remove empty columns

# make a data frame with the max & min value per column
max_min <- data_frame(max = apply(player_statistics, 2, max),
                      min = apply(player_statistics, 2, min),
                      columns = names(player_statistics))

# vector of useless column names
useless_columns <- max_min$columns[max_min$min == max_min$max]
# add  minus signs so select() will remove them
useless_columns <- paste("-", useless_columns, sep = "")

# remove useless columns
player_statistics <- player_statistics %>% select_(.dots = useless_columns )



# remove leaky variables
player_statistics <- player_statistics %>%
  group_by(solo_WinRatio) %>% #use group_by to protect our target variable
  select(-contains("win")) %>% # remove any columns with "win" in the name
  select(-contains("loss")) %>% # remove any columns with "loss" in the name
  ungroup() # remove grouping


#We also want to only use numeric variables, so let's get rid of any rows with na values or columns that aren't numeric.

# numeric data only
player_statistics <- player_statistics %>%
  na.omit() %>% 
  select_if(is.numeric)


# check out our final dataset
str(player_statistics)




# split data into testing & training
set.seed(1234)

# train/test split
training_indexs <- createDataPartition(player_statistics$solo_WinRatio, p = .2, list = F)
training <- player_statistics[training_indexs, ]
testing  <- player_statistics[-training_indexs, ]

# get predictors
predictors <- training %>% select(-solo_WinRatio) %>% as.matrix()
output <- training$solo_WinRatio


#Train an untuned model

#Because we're trying to predict a numeric variable (how often a given player will win) rather than categorize 
#players (e.g. high win-rate vs. low win-rate players), we're going to treat this as a regression problem 
#rather than a classification problem. 
#We're going to model it using random forests. Random forests have two main parameters that you can change, 
#ntree and mtry.

#ntree: This is the total number of trees in your final ensemble model.
#mtry: The number of features to use to build each tree.

#We're only going to be using caret to tune mtry, not ntrees. Why? It has to do with how the overall error of 
#the model changes as we change these parameters. Caret works by finding the parameter value where we have the
#lowest overall error. Some features (like ntrees) will usually continue to reduce overall error as you increase them.
#You can see an example of this in this 2015 paper by Buskirk & Kolenikov:

#you want to pick a value near the "elbow", where you have high accuracy but aren't training more trees than 
#you need to. In practice, most random forest models will have good performance with a number of trees 
#somewhere between 50 and 500.

# train a random forest model
model <- randomForest(x = predictors, y = output,
                      ntree = 50) # number of trees

# check out the details
model


# check out our model's root mean squared error on the held out test data
rmse(predict(model, testing), testing$solo_WinRatio)


#Tuning with caret
# use caret to pick a value for mtry
tuned_model <- train(x = predictors, y = output,
                     ntree = 5, # number of trees (passed ot random forest)
                     method = "rf") # random forests

print(tuned_model)


# plot the rmse for various possible training values
ggplot(tuned_model)


# compare the model
print("base model rmse:")
print(rmse(predict(model, testing), testing$solo_WinRatio))

print("tuned model rmse:")
print(rmse(predict(tuned_model$finalModel, testing), testing$solo_WinRatio))

# plot both plots at once
par(mfrow = c(1,2))

varImpPlot(model, n.var = 6)
varImpPlot(tuned_model$finalModel, n.var = 6)
