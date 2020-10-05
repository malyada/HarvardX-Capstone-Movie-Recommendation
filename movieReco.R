# install packages if not present.
install.packages('tidyverse', dependencies = T)
# ~brew install pkg-config~ (mac) if the below fails due to data.table not found
install.packages('data.table', dependencies = T)
install.packages('caret', dependencies = T)
install.packages('Metrics', dependencies = T)
install.packages('rafalib')
install.packages('recosystem')
install.packages('furrr')

#load the needed libraries
library(tidyverse)
library(caret)
library(data.table)
library(Metrics)
library(rafalib)
library(recosystem)
library(parallel)
library(furrr)

#getting datasets
dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# if using R 3.6 or earlier:
#movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
#                                           title = as.character(title),
#                                           genres = as.character(genres))
# if using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Create edx set, validation set (final hold-out test set)
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

#remove unnecessary objects to save memory
rm(movielens)
rm(movies)
rm(removed)
rm(temp)
rm(ratings)
rm(test_index)

#to avoid rerunning it every time. 
write_csv(edx, 'edxtrain.csv')
write_csv(validation, 'edxtest.csv')

edx <- as.tibble(read_csv("edxtrain.csv"))
validation <- as.tibble(read_csv("edxtest.csv"))

#goal: to predict the ratings for the missing/test movie user combination.
#we will be using the rmse error metric to measure our model improvement.

#data exploration
print(head(edx))
 

edx %>% summarise(
  n_users=n_distinct(userId),# unique users from train
  n_movies=n_distinct(movieId),# unique movies from train
  min_rating=min(rating),  # the lowest rating 
  max_rating=max(rating) # the highest rating
)

#if we multiply n_users x n_movies, we get a number of almost 750 million. To visualize this is
#not possible because of the size constraints and it might crash r
#lets visualize some movies users.
# matrix for a random sample of 100 movies and 100 users with yellow 
# indicating a user/movie combination for which we have a rating.
users <- as.vector(sample(unique(edx$userId), 100))
rafalib::mypar()
edx %>% filter(userId %in% users) %>% 
  select(userId, movieId, rating) %>%
  mutate(rating = 1) %>%
  spread(movieId, rating) %>% select(sample(ncol(.), 100)) %>% 
  as.matrix() %>% t(.) %>%
  image(1:100, 1:100,. , xlab="Movies", ylab="Users") + 
abline(h=0:100+0.5, v=0:100+0.5, col = "grey")

#remove objects no longer needed
rm(users)

#we see that the matrix is too sparse
#distribution of the avg rating of the movies
edx %>%
  group_by(movieId) %>%
  summarise(movierating = mean(rating)) %>%
  ggplot(aes(movierating)) +
  geom_histogram()
#distribution of the avg rating given by the users
edx %>%
  group_by(userId) %>%
  summarise(userrating = mean(rating)) %>%
  ggplot(aes(userrating)) +
  geom_histogram()

#Methods and analysis
#simplest model of giving mean of all the ratings as prediction
mu <- mean(edx$rating)
print('trainig rmse base model:')
print(rmse(edx$rating, mu))
print('test rmse base model:')
base_rmse <- rmse(validation$rating, mu)
print(base_rmse)
rmse_results <- tibble(Model = "Average of ratings", RMSE = base_rmse)

#We noticed in the data exploratin that some movies are rated differently than others.
#there seems to be a movie bias associated to each movie.
#since there are soo many rows.. fitting a linear model takes a lot of time
# we know that the least square estimate bi is just the average of yi - mu  for each movie i. So we can compute them 

mbias <- edx %>%
  group_by(movieId) %>%
  summarise(bm = mean(rating - mu)) %>%
  ungroup()

#distribution of bm
mbias %>%
  ggplot(aes(bm)) + geom_histogram()
#train rmse
pred <- mu + edx %>%
  left_join(mbias, by = 'movieId') %>%
  .$bm
pred = ifelse(pred >= 0, pred, 0)
print('train rmse movie bias model:')
print(rmse(edx$rating, pred))
#test rmse
pred <- mu + validation %>%
  left_join(mbias, by = 'movieId') %>%
  .$bm
pred = ifelse(pred >= 0, pred, 0)
print('test rmse movie bias model:')
rmse_mbias <- rmse(validation$rating, pred)
print(rmse_mbias)
rmse_results <- bind_rows(rmse_results,
                          tibble(Model="movie_bias_effect",  
                                     RMSE = rmse_mbias))

#We also notices that some users rate differently
#considering user bias
#it is not practical to do a linear regression for userbias and movie bias
#bu is the average of yiu - mu − bi (i->movie, u->user)
ubias <- edx %>%
  left_join(mbias, by = 'movieId') %>%
  group_by(userId) %>%
  summarise(bu = mean(rating - mu - bm)) %>%
  ungroup()
#distribution of bu
ubias %>%
  ggplot(aes(bu)) + geom_histogram()
#train rmse
pred <- mu + edx %>%
  left_join(mbias, by = 'movieId') %>%
  left_join(ubias, by = 'userId') %>%
  mutate(req = bm + bu) %>%
  .$req
pred = ifelse(pred >= 0, pred, 0)
print('train rmse movie-user bias model:')
print(rmse(edx$rating, pred))
#test rmse
pred <- mu + validation %>%
  left_join(mbias, by = 'movieId') %>%
  left_join(ubias, by = 'userId') %>%
  mutate(req = bm + bu) %>%
  .$req
pred = ifelse(pred >= 0, pred, 0)
print('test rmse movie-user bias model:')
mb_bias <- rmse(validation$rating, pred)
print(mb_bias)
rmse_results <- bind_rows(rmse_results,
                          tibble(Model="movie_user_bias_effect",  
                                     RMSE = mb_bias))

#lets look at where we went wrong
analysis <- edx %>%
  left_join(mbias, by = 'movieId') %>%
  left_join(ubias, by = 'userId') %>%
  mutate(predr = (bm + bu + mu), diff = (predr - rating))  %>%
  group_by(title) %>%
  summarise(n = n(), rating =  mean(rating), predr = mean(predr), diff = mean(diff), diffsq = diff^2) %>%
  arrange(-diffsq) 
print(head(analysis, 20))
print('avg number of ratings per movie')
print(mean(analysis$n))

#we have an avg of 843 ratings per movie
#lets see how many odd ones are present. say the odd ones are the ones having less than 50 ratings
analysis %>%
  filter(n < 50)
#there are roughly 3500 odd movies with less than 50 ratings.. this will drastically effect our analysis

#cases where we underestimated
analysis %>%
  filter(diff < 0, n < 50) %>%
  arrange(-diffsq)

#cases where we overestimated 
analysis %>%
  filter(diff > 0, n < 50) %>%
  arrange(-diffsq)

#since we have extremes in the movies which are rated less.. in most cases by one or 2 users its is good to 
# regularise the moviebias bm and user bias bu in order to penalise the estimates that come from small sample sizes.

#to choose the appropriate regularization parameter using 4 fold cv, lets divide the edx set to 4 sets.
set.seed(1, sample.kind="Rounding")
tmp <- createDataPartition(y = edx$rating, times = 1, p = 0.5, list = F)
t <- edx[tmp, ]
p <- createDataPartition(y = t$rating, times = 1, p = 0.5, list = F)
p1 <- t[p, ]
p2 <- t[-p, ]
t <- edx[-tmp, ]
p <- createDataPartition(y = t$rating, times = 1, p = 0.5, list = F)
p3 <- t[p, ]
p4 <- t[-p, ]

v <- list(p1, p2, p3, p4)

#lambda is the regularization parameter which we will optimize over the regltn set
possible_lambdas <-  seq(4, 5.5, by = 0.25)

plan(multicore) #we will be using multiprocessing to run in parallel

options(future.globals.maxSize= 891289600)

getrmses <- function(l) {
  r <- future_map_dbl(v, getmeancvrmse, l = l) 
  mean(r)
}

getmeancvrmse <- function(temp, l = 0) {
  #we make train and regltn sets appropriately for all the 4 cross validation folds
  train <- anti_join(edx, temp)
  # Make sure userId and movieId in regltn set are also in train set
  regltn <- temp %>% 
    semi_join(train, by = "movieId") %>%
    semi_join(train, by = "userId")
  
  # Add rows removed from regltn set back into train set
  removed <- anti_join(temp, regltn)
  train <- rbind(train, removed)
  
  mbias <- train %>%
    group_by(movieId) %>%
    summarise(bm = sum(rating - mu)/(n() + l)) %>%
    ungroup()
  ubias <- train %>%
    left_join(mbias, by = 'movieId') %>%
    group_by(userId) %>%
    summarise(bu = sum(rating - mu - bm)/(n() + l)) %>%
    ungroup()
  pred <- mu + regltn %>%
    left_join(mbias, by = 'movieId') %>%
    left_join(ubias, by = 'userId') %>%
    mutate(req = bm + bu) %>%
    .$req
  pred = ifelse(pred >= 0, pred, 0)
  rmse(regltn$rating, pred)
}

rmses <- future_map_dbl(possible_lambdas, getrmses)
plot(possible_lambdas, rmses)
rmses

l <- possible_lambdas[min(rmses) == rmses]

#we got the optimal value of lambda(l)
#training the whole edx at l

mbias <- edx %>%
  group_by(movieId) %>%
  summarise(bm = sum(rating - mu)/(n() + l)) %>%
  ungroup()

ubias <- edx %>%
  left_join(mbias, by = 'movieId') %>%
  group_by(userId) %>%
  summarise(bu = sum(rating - mu - bm)/(n() + l)) %>%
  ungroup()
#predictions for train and test sets
predtrain <- mu + edx %>%
  left_join(mbias, by = 'movieId') %>%
  left_join(ubias, by = 'userId') %>%
  mutate(req = bm + bu) %>%
  .$req
predtrain = ifelse(predtrain >= 0, predtrain, 0)

predtest <- mu + validation %>%
  left_join(mbias, by = 'movieId') %>%
  left_join(ubias, by = 'userId') %>%
  mutate(req = bm + bu) %>%
  .$req
predtest = ifelse(predtest >= 0, predtest, 0)
#rmse computations
print('train rmse regularised movie-user bias model:')
print(rmse(edx$rating, predtrain))
print('test rmse regularised movie-user bias model:')
mbreg_bias <- rmse(validation$rating, predtest)
print(mbreg_bias)
rmse_results <- bind_rows(rmse_results,
                          data_frame(Model="regularised_movie_user_bias_effect",  
                                     RMSE = mbreg_bias))

#To further improve upon the rmse, predicting the residuals.
edx_res <- edx %>% 
  left_join(mbias, by = "movieId") %>%
  left_join(ubias, by = "userId") %>%
  mutate(res = rating - mu - bm - bu) %>%
  select(userId, movieId, res)
head(edx_res)

#we will be using recosystem library to perform matrix factorization on the residuals
#organizing both training and validation sets to 3 columns userId, movieId, value.

# as matrix 
train_fact <- as.matrix(edx_res)
test_fact <- validation %>% 
  select(userId, movieId, rating)
test_fact <- as.matrix(test_fact)

# build a recommender object
r <-Reco()

# write train_fact and test_fact tables on disk
write.table(train_fact , file = "trainset.txt" , sep = " " , row.names = FALSE, col.names = FALSE)
write.table(test_fact, file = "validset.txt" , sep = " " , row.names = FALSE, col.names = FALSE)

# use data_file() to specify a data set from a file in the hard disk.
# r$tune needs training data to be of type DataSource.
set.seed(2000) 
train_fact <- data_file("trainset.txt")
test_fact <- data_file("validset.txt")

# tuning training set using the default 5 fold cross validation
cores <- detectCores()
opts <- r$tune(train_fact, opts = list(dim = c(40, 45), lrate = c(0.05, 0.1, 0.2),
                                      costp_l1 = 0, costq_l1 = 0,
                                      nthread = (cores - 1), niter = 10))
opts
# training the recommender model
r$train(train_fact, opts = c(opts$min, nthread = 1, niter = 20))

# Making prediction on validation set and calculating RMSE:
pred_file <- tempfile()
y_pred_resid <- scan(pred_file)
y_pred <- predtest + y_pred_resid
rmse_mf <- RMSE(y_pred,validation$rating) 
cat('final Rmse:', rmse_mf)
rmse_results <- bind_rows(rmse_results,
                          data_frame(Model="factorization + regularised_movie_user_bias",  
                                     RMSE = rmse_mf))
#final result
#we achieved a final rmse on test set (validation given by edx) of < 0.79

#out model improvement in steps
print(rmse_results)