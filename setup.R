.libPaths("/mnt/vol/dev/privateRepo/")

## Importing packages
library(tidyverse)
library(magrittr)
library(stringi)
library(tm)
library(glmnet)
library(caret)
require(e1071)
library(text2vec) # text vectorization
library(glmnet) # building model cv.gmlnet()
library(MLmetrics) # F1_Score()