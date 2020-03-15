colsSpec <- cols(
  id = col_integer(),
  keyword = col_character(),
  location = col_character(),
  text = col_character(),
  target = col_integer()
)
train_df<-read_csv("data/train.csv", col_types = colsSpec)
test_df<-read_csv("data/test.csv", col_types = colsSpec)