source("setup.R")
source("loadData.R")

summary(m1 <- glm(target ~ keyword + location, family = "poisson", data = train_df))

pred <- predict(m1, test_df)
# this does not work because there are new "levels" (words) in test data