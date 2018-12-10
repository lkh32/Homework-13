library(dplyr)
library(caret)
library(nnet)
library(magrittr)

show_number <- function(m, i, oriented=T)
{
  im <- matrix(mtrain[i,], byrow=T, nrow=28)
  
  if (oriented) {
    im_orient <- matrix(0, nrow=28, ncol=28)
    for (i in 1:28)
      im_orient[i,] <- rev(im[,i])
    
    im <- im_orient
  }
  image(im)
}

# get the training datasets
if (!exists("mtrain")) {
  mtrain <- read.csv("mnist_train.csv", header=F) %>% as.matrix
  for (i in 1:nrow(mtrain)){
    cn <- mtrain[i,1]
    if (cn == "3"){
      mtrain[i,1] <- 1
    } 
    else {mtrain[i,1] <- 0}
  }
  train_classification <- mtrain[,1]
  mtrain <- mtrain[,-1]/256
  colnames(mtrain) <- NULL
  rownames(mtrain) <- NULL
}


# look at a sample
show_number(mtrain, 8)

#filter to use the first 1000 mnist samples
mtrain <- mtrain[1:1000,]

y <- factor(train_classification[1:1000])
x <- mtrain

prediction_errors <- function(train_classification, file, nnet)
{
  y <- factor(train_classification[1:1000])
  x <- file
  
  true_y <- y
  pred_y <- predict(nnet, x)
  
  n_samples <- nrow(x)
  error <- sum(true_y != pred_y)/n_samples
  return (error)
}


# fit the data to a neural net, nnet model in caret
# the nnet model has the following parameters: size, decay
# part i)
tuning_df <- data.frame(size=8:12, decay=0)

fitControl <- trainControl(method="none")

fitControl <- trainControl(## 2-fold CV
  method = "repeatedcv",
  number = 2,
  ## repeated ten times
  repeats = 3)

t_out <- caret::train(x=x, y=y, method="nnet",
                      trControl = fitControl,
                      tuneGrid=tuning_df, maxit=100000)


# part ii)
tuning_df <- data.frame(size=8:12, decay=c(0, .1, .5, 1, 2))

fitControl <- trainControl(method="none")

fitControl <- trainControl(## 2-fold CV
  method = "repeatedcv",
  number = 2,
  ## repeated ten times
  repeats = 2)

t_out2 <- caret::train(x=x, y=y, method="nnet",
                      trControl = fitControl,
                      tuneGrid=tuning_df, maxit=100000)


#Testing the neural network on the rest of the fie 

if (!exists("mtest")) {
  mtest <- read.csv("mnist_test.csv", header=F) %>% as.matrix
  for (i in 1:6000){
    cn <- mtest[i,1]
    if (cn == "3"){
      mtest[i,1] <- 1
    } else {
      mtest[i,1] <- 0
    }
  }
  test_classification <- mtest[,1]
  mtest <- mtest[,-1]/256 
  colnames(mtest) <- NULL
  rownames(mtest) <- NULL
}
mtest <- mtest[1:1000,]

prediction_errors(train_classification,mtest,t_out)
prediction_errors(train_classification,mtest,t_out2)

