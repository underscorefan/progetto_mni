gen_random_dataframe<-function(rows = 10, cols = 10, mean = 0, sd = 1){
  df<-data.frame()
  for (col in 1:cols){
    vals<-abs(rnorm(rows, mean = mean, sd = sd))
    if (col == 1){
      df<-data.frame(col1 = vals)
    }else{
      df[[gen_col(col)]] = vals
    }
  } 
  df
}

gen_col<- function(i = 0){
  paste0("col", as.character(i))
}