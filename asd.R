final_models_compare
final_model_choice



row = 200
m = 3
final_model_choice[row, 2:4]

a <- c(final_model_choice[row, 2:4]$m1c, final_model_choice[row, 2:4]$m2c,final_model_choice[row, 2:4]$m3c)
a

final_models_compare[a[1],m]
final_models_compare[a[2],m+1]
final_models_compare[a[3],m+2]


which.max(c(
  final_models_compare[a[1],m],
  final_models_compare[a[2],m+1],
  final_models_compare[a[3],m+2]
))

eps <- 0.00000000001

a <- c(22, 0, 10)
test_df

b<-which.max(c(
  ifelse(length(final_models_compare[a[1],m])==0, eps ,final_models_compare[a[1],m]),
  ifelse(length(final_models_compare[a[2],m+1])==0, eps, final_models_compare[a[2],m+1]),
  ifelse(length(final_models_compare[a[3],m+2])==0, eps, final_models_compare[a[3], m+2])))


final_model_choice[row,5] <- final_model_choice[row,b+1]

max_compare = function(row){
  a <- row
  
  b<-which.max(c(
    ifelse(length(final_models_compare[a[1],m])==0, eps ,final_models_compare[a[1],m]),
    ifelse(length(final_models_compare[a[2],m+1])==0, eps, final_models_compare[a[2],m+1]),
    ifelse(length(final_models_compare[a[3],m+2])==0, eps, final_models_compare[a[3], m+2])))
  row[4] = row[b]
}


c <- apply(final_model_choice[,c("m1c","m2c","m3c","mfc")], MARGIN = 1, FUN=max_compare)
final_model_choice$mfc = c

final_model_choice$mfc


table(y_val, final_model_choice$mfc)
F_score(table(y_val, final_model_choice$mfc))
