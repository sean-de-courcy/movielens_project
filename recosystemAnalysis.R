options(timeout = 10000) # So that the program can deal with the huge dataset without timing out

##########################################################
# Create edx set, validation set (final hold-out test set)
##########################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# if using R 3.6 or earlier:
# movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
#                                           title = as.character(title),
#                                           genres = as.character(genres))
# if using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))


movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)



#######################################################################
# The code above was provided by the edX Data Science: Capstone course.
# The rest of the code is my own original work.
# -- Sean Coursey
#######################################################################

# Loading libraries #
#library(tidyverse) -- already loaded
#library(caret)     -- already loaded
if(!require(recosystem)) install.packages("recosystem", repos = "http://cran.us.r-project.org")
if(!require(Metrics)) install.packages("Metrics", repos = "http://cran.us.r-project.org")
library(recosystem)
library(Metrics)

# Saving metadata for each movieId for future use #
movieIdMeta <- edx %>% select(movieId, title, genres)
movieIdMeta <- unique(movieIdMeta, by = "movieId")

# Cleaning edx and validation sets #
edx <- edx %>% select(userId, movieId, rating)
validation <- validation %>% select(userId, movieId, rating)

# Creating train and test sets #
set.seed(2, sample.kind = "Rounding") # to maximize reproducibility
testindex = createDataPartition(y = edx$rating, times = 1, p = 0.1, list = FALSE)
train <- edx[-testindex]
temp <- edx[testindex]
test <- temp %>% semi_join(train, by = "userId") %>% semi_join(train, by = "movieId")
train <- rbind(train, anti_join(temp, test))
rm(edx, temp, testindex)

# Centering Data
average = mean(train$rating)
train <- train %>% mutate(delta = rating - average)
variance <- var(train$delta)

# Creating table of userIds and the average rating given by that user #
userIds <- unique(train$userId)
len <- length(userIds)
avg_rating <- vector(mode = "numeric", length = len)
i <- 1
j <- 0
for (id in userIds) { # WARNING: This took half an hour on my computer.
  avg_rating[i] <- mean(train %>% filter(userId == id) %>% .$delta)
  # Prints out percent done every percent #
  if ((i/len*100) %/% 1 > j) {
    j <- j + 1
    print(j)
  }
  i <- i + 1
}
userAvgRat <- data.frame(user = userIds, mu = avg_rating)

rm(userIds, avg_rating, id)

# Couldn't find a vectorizable way of treating userAvgRat as dictionary, so I made one #
userAvgRatLookup <- function(U) {
  len <- length(U)
  avg_ratings <- vector(mode = "numeric", length = len)
  i <- 1
  j <- 0
  for (u in U) {
    avg_ratings[i] <- userAvgRat[userAvgRat$user == u, ]$mu
    # Prints out percent done every half percent (it's a slow function) #
    if ((i/len*100) %/% 0.5 > j*2) {
      j <- j + 0.5
      print(j)
    }
    
    i <- i + 1
  }
  return(avg_ratings)
}

# Centering train set using average user rating #
# WARNING: This took my computer around 1 hour.
userCentered <- train %>% mutate(delta = delta - userAvgRatLookup(userId))
variance_user <- var(userCentered$delta)

# Doing the same thing but for movieIds #
movieIds <- unique(userCentered$movieId)
len <- length(movieIds)
avg_rating <- vector(mode = "numeric", length = len)
i <- 1
j <- 0
for (id in movieIds) {
  avg_rating[i] <- mean(userCentered %>% filter(movieId == id) %>% .$delta)
  if ((i/len*100) %/% 1 > j) {
    j <- j + 1
    print(j)
  }
  i <- i + 1
}
movieAvgRat <- data.frame(movie = movieIds, mu = avg_rating)
rm(movieIds, avg_rating, id)

movieAvgRatLookup <- function(M) {
  len <- length(M)
  avg_ratings <- vector(mode = "numeric", length = len)
  i <- 1
  j <- 0
  for (m in M) {
    avg_ratings[i] <- movieAvgRat[movieAvgRat$movie == m, ]$mu
    # Prints out percent done every half percent (it's a slow function) #
    if ((i/len*100) %/% 0.5 > j*2) {
      j <- j + 0.5
      print(j)
    }
    
    i <- i + 1
  }
  return(avg_ratings)
}

# Centering train set using movie average ratings #
movieCentered <- train %>% mutate(delta = delta - movieAvgRatLookup(movieId))
variance_movie <- var(movieCentered$delta)

# Finding percent variance explained by user and movie averages independently #
percent_variance_user <- (variance - variance_user)/variance*100
percent_variance_movie <- (variance - variance_movie)/variance*100

# Centering the train set using movie and user average ratings #
train <- train %>% mutate(delta = movieCentered$delta + userCentered$delta - delta)
variance_both <- var(train$delta)

# Finding precent variance explained by user and movie averages together #
percent_variance_usermovie <- (variance - variance_both)/variance*100

rm(userCentered, movieCentered, variance, variance_user, variance_movie)
rm(i, j, len)

# Creating a recosystem object and a version of the training set compatible with it #
r <- Reco()
trainReco <- data_memory(user_index = train$userId, item_index = train$movieId, rating = train$delta, index1 = TRUE)

# Tuning the recosystem model # 
# WARNING: The next two sections of code use multi-threading, change "nthread"
# if you do not have 8 CPU cores.
opts_tune <- r$tune(trainReco,
                    opts = list(dim = c(10, 20, 30),
                                costp_l2 = c(0.1, 0.01),
                                costq_l2 = c(0.1, 0.01),
                                costp_l1 = 0,
                                costq_l1 = 0,
                                lrate = c(0.1, 1),
                                nthread = 8,
                                niter = 15,
                                verbose = TRUE))

# Training recosystem model with optimal options #
r$train(trainReco, opts = c(opts_tune$min, niter = 50, nthread = 8))

r_mat <- r$output(out_P = out_memory(), out_Q = out_memory()) # user (P) and movie (Q) matrices

# Using recosystem model to predict test set #
testReco <- data_memory(test$userId, test$movieId, index1 = TRUE)
testDeltPred <- r$predict(testReco, out_memory())
testRatPred <- testDeltPred + userAvgRatLookup(test$userId) + movieAvgRatLookup(test$movieId) + average

# Getting initial rmse from test set: 0.795 #
rmse(test$rating, testRatPred)

# Lookup function for the delta contribution of the nth factor $
nthFactorLookup <- function(U, M, n, upTo = FALSE) {
  len <- length(U)
  products <- vector(mode = "numeric", length = len)
  I <- 1:len
  if (upTo) {
    for (i in I) {
      products[i] <- sum(r_mat$P[U[i], 1:n]*r_mat$Q[M[i], 1:n])
    }
  } else {
    for (i in I) {
      products[i] <- r_mat$P[U[i],n]*r_mat$Q[M[i],n]
    }
  }
  return(products)
}

# Calculating the variance explained by each factor independently #
len <- length(r_mat$P[1,])
percent_var_ind <- vector(mode = "numeric", length = len)
I <- 1:len
for (i in I) {
  temp <- train %>% mutate(delta = delta - nthFactorLookup(userId, movieId, i))
  percent_var_ind[i] <- (var(train$delta) - var(temp$delta))/var(train$delta)*100
}

# Calculating the variance explained by the factors cumulatively #
percent_var_cum <- vector(mode = "numeric", length = len)
for (i in I) {
  temp <- train %>% mutate(delta = delta - nthFactorLookup(userId, movieId, i, TRUE))
  percent_var_cum[i] <- (var(train$delta) - var(temp$delta))/var(train$delta)*100
}

# Calculating the variance uniquely explained by each factor #
percent_var_dep <- vector(mode = "numeric", length = len)
percent_var_dep[1] = percent_var_cum[1]
for (i in 2:len) {
  percent_var_dep[i] = percent_var_cum[i] - percent_var_cum[i-1]
}

rm(i, I, len)

# Plotting the movies by the two most important factors #
movie_data_2factors <- movieIdMeta %>% mutate(X1 = r_mat$Q[movieId, which.max(percent_var_ind)], X2 = r_mat$Q[movieId, order(percent_var_ind)[29]],
                                              genres = str_extract(genres, "^[:alpha:]+(?=\\|)"))
plot_2factor <- movie_data_2factors %>% ggplot(aes(x = X1, y = X2, color = genres)) + geom_point(alpha = 0.2)
plot_2factor

# Initial rmse was satisfactory, moving on to validation set #
validationReco <- data_memory(validation$userId, validation$movieId, index1 = TRUE)
validationDeltPred <- r$predict(validationReco, out_memory())
validationRatPred <- validationDeltPred + userAvgRatLookup(validation$userId) + movieAvgRatLookup(validation$movieId) + average

# Getting validation rmse: 0.794 #
# NOTE: Some randomness is involved in the training process due to
# multi-threading, so the exact value varies slightly.
rmse(validation$rating, validationRatPred)

options(timeout = 60) # Just setting things back to normal
