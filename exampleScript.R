.libPaths("/mnt/vol/dev/privateRepo/")

# example with submission: https://www.kaggle.com/wrrosa/nlp-getting-started-tutorial-in-r

## Importing packages
library(tidyverse)
library(stringi)
library(tm)
library(glmnet)
library(caret)
require(e1071)
library(text2vec) # text vectorization
library(glmnet) # building model cv.gmlnet()
library(MLmetrics) # F1_Score()

train_df<-read_csv("data/train.csv")
test_df<-read_csv("data/test.csv")

# Removing invalid targets
train_df <- train_df %>% distinct(keyword,location,text,.keep_all = TRUE)
print(nrow(train_df))

it <- itoken(train_df[1:5,'text'] %>% dplyr::pull(), preprocess_function = tolower, 
             tokenizer = word_tokenizer, chunks_number = 10, progessbar = F)

# using unigrams here
vocab <- create_vocabulary(it, ngram = c(1L, 1L))
dtm = create_dtm(it, vocab_vectorizer(vocab))
str(dtm)

# training data:
it <- itoken(train_df[['text']], preprocess_function = tolower, 
             tokenizer = word_tokenizer, chunks_number = 10, progessbar = F)
# using unigrams here
vocab <- create_vocabulary(it, ngram = c(1L, 1L))
# and pruning our vocabulary
pruned_vocab <- prune_vocabulary(vocab, term_count_min = 10,
                                 doc_proportion_max = 0.5, doc_proportion_min = 0.001)
dtm = create_dtm(it, vocab_vectorizer(pruned_vocab))
# test data:
it_test <- itoken(test_df[['text']], preprocess_function = tolower, 
                  tokenizer = word_tokenizer, chunks_number = 10, progessbar = F)
dtm_test = create_dtm(it_test, vocab_vectorizer(pruned_vocab ))

set.seed(23011990) # just making model reproduceable

fit <- cv.glmnet(x = dtm, y = train_df[['target']], 
                 family = 'binomial', 
                 # lasso penalty
                 alpha = 1,
                 # interested mean absolute error
                 type.measure = "mae",
                 # 5-fold cross-validation
                 nfolds = 5,
                 # high value, less accurate, but faster training
                 thresh = 1e-3,
                 # again lower number iterations for faster training
                 # in this vignette
                 maxit = 1e3,
                 # mixing fits (see cv.gmlnet() documentation for more details)
                 relax = TRUE
)

plot(fit)

print (paste("min MAE = ", round(min(fit$cvm), 4)))

pred = predict(fit_tf,dtm_tf,type = "class")
print(paste("F1 (train_df, binomial + TfIdf) = ",round(F1_Score(actual,pred,positive = 1),4)))

sample_submission <- read_csv("./data/sample_submission.csv")
sample_submission["target"] = predict(fit_tf,dtm_tf_test,type = "class")
write.csv(sample_submission,"submission.csv", row.names = FALSE,quote = FALSE)

