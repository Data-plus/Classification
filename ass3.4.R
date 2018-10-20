## Install packages
#install.packages("keras")
#install_keras()
#install.packages('curl')
#install.packages("tm")
#install.packages("qdap")

set.seed(12345)

library(keras)
install_keras(tensorflow = "gpu")
library(tm)

setwd("~/Uni/2018-2/FIT5149 Applied Data Analysis/Assignment")


## Load Data
text <- readLines("training_docs.txt")
text <- as.data.frame(text)
text <- as.data.frame(text[!(text$text=="EOD"),], header = FALSE)
colnames(text) <- c("text")
#text <- text[!grepl("ID te_doc_",text$text),] # For test dataset
text <- text[!grepl("ID tr_doc_",text$text),] # For train dataset
text <- as.data.frame(text)
text <- as.data.frame(text[!(text$text==""),],header= FALSE)
colnames(text) <- c("text")
text <- lapply(text,function(x) gsub("TEXT ","",x))
text <- as.data.frame(text)

label <- as.data.frame(readLines("training_labels_final.txt"))
label$labels <- sub(".* ", "",label$`readLines("training_labels_final.txt")`)
label$`readLines("training_labels_final.txt")` <- sub(" .*", "",label$`readLines("training_labels_final.txt")`)
colnames(label) <- c("doc","label")


## Initialise
# Word vector
maxlen <- 200 # Input documents have 100 words each
max_words <- 50000  # Size of the featues in the text data. original = 50000
embedding_dim <- 200 # Dim size of embedding matrix


# Custom Functions
F_score <- function(cm_table){
  # Recall
  my_recall <- vector(mode="numeric", length=23)
  for (i in 1:23){
    my_recall[i] <- cm_table[i,i]/sum(cm_table[,i])
  }
  
  # Precision
  my_precision <- vector(mode="numeric", length=23)
  for (i in 1:23){
    my_precision[i] <- cm_table[i,i]/sum(cm_table[i,])
  }
  
  my_F1 <- vector(mode="numeric", length=23)
  for (i in 1:23){
    my_F1[i] <- 2*((my_recall[i]*my_precision[i]) / (my_recall[i]+my_precision[i]))
  }
  Final_F1 <- mean(my_F1)
  
  return (Final_F1)
  
} 

class_accuracy <- function(cm_table){
  for (i in 1:ncol(cm_table)){
    class_acc[i,2] <- (sum(cm_table[i,i]) / sum(cm_table[,i]))
  }
  return (class_acc)
}

# Early stopping
es_callback <- callback_early_stopping(monitor='val_acc', min_delta=0, patience=1, verbose=0)

# Compile + Fitting model
compile_fit <- function(model, epochs=10, batch=32, split_ratio=0.2, optimizer="rmsprop"){
  model %>% compile(
    optimizer = optimizer,
    loss = "categorical_crossentropy",
    metrics = c("acc")
  )
  history <- model %>% fit(
    x_train, y_train,
    epochs = epochs,
    batch_size = batch,
    callbacks = list(es_callback),
    validation_split = split_ratio
  )
  return (history)
}


# Ensemble
ensemble <- function(val_data = 'x_val', w = c(0.6, 0.20, 0.20)){
  
  N <- nrow(val_data)
  w = w # weighting
  # reset existing
  model1_pred <- 0
  model2_pred <- 0
  model3_pred <- 0
  
  
  # Probability storing
  model1_pred <- predict_proba(object = cbi_lstm, x = val_data)
  model2_pred <- predict_proba(object = conv_pool_cnn, x = val_data)
  model3_pred <- predict_proba(object = model_test, x=val_data)
  
  # Weighted average
  cs <-(w[1]*model1_pred[1:N,]+w[2]*model2_pred[1:N,]+w[3]*model3_pred[1:N,])
  
  # Return prediction
  return(apply(cs, MARGIN = 1,FUN = which.max))
}




# Pre-processing text
corpus <- Corpus(VectorSource(text$text))
corpus <- tm_map(corpus, removeNumbers)
corpus <- tm_map(corpus, content_transformer(tolower))
corpus <- tm_map(corpus, removePunctuation)
corpus <- tm_map(corpus, removeWords, stopwords('english'))
#corpus <- tm_map(corpus, content_transformer(stemDocument)) # Stemming didn't help
corpus <- tm_map(corpus, content_transformer(stripWhitespace))
cleaned.docs = data.frame(text=sapply(corpus,identity), stringsAsFactors = F)

head(cleaned.docs,1)

# Only used for overviewing the features
# DTM_train <- DocumentTermMatrix(corpus, control=list(wordLengths=c(4,Inf)))
# DTM_train # Number of features = terms: 159389
# DTM_train <- removeSparseTerms(DTM_train,0.97)
# DTM_train # Number of features = terms: 493
# tfm <- weightTfIdf(TermDocumentMatrix(corpus[training_indices]))



texts = cleaned.docs$text
tokenizer <- text_tokenizer(num_words = max_words) %>% # max_word = most common words
    fit_text_tokenizer(texts)

# Gether the sequences
sequences <- texts_to_sequences(tokenizer, texts)

# Gether word index
word_index = tokenizer$word_index

# Pad sequences to datafarme
data <- pad_sequences(sequences, maxlen = maxlen)

labels = as.array(label$label)
labels = as.numeric(gsub("C", "", labels))



## Data Split
indices <- sample(1:nrow(data))
training_samples = 100000
training_indices <- indices[1:training_samples]
validation_indices <- indices[(training_samples + 1):nrow(data)]

x_train <- data[training_indices,]
y_train <- labels[training_indices]
y_train <- y_train-1
y_train = to_categorical(y_train,num_classes = length(unique(y_train)))
x_val <- data[validation_indices,]
y_val <- labels[validation_indices]



########### Model 1 [ Convolutional Bi-directional LSTM ] ##############
# 0.7579 epoch 3, 10k data
cbi_lstm <- keras_model_sequential() %>%
  layer_embedding(input_dim = max_words, output_dim = embedding_dim,
                  input_length = maxlen) %>%
  layer_conv_1d(filters = 1000, kernel_size = 4, activation = 'relu') %>%
  layer_max_pooling_1d(pool_size=2) %>%
  layer_spatial_dropout_1d(0.5) %>%
  bidirectional(layer_cudnn_lstm(units = 500, return_sequences = TRUE)) %>%
  layer_dropout(0.25) %>%
  bidirectional(layer_cudnn_lstm(units = 250, return_sequences = TRUE)) %>%
  layer_dropout(0.25) %>%
  layer_flatten() %>%
  layer_dense(units = 23, activation = "softmax")
summary(cbi_lstm)

compile_fit(cbi_lstm, epochs = 3)
F_score(table(y_val, predict_classes(cbi_lstm, x_val)))






########### Model 2 [ ConvPool-CNN-C ] 0.7188 ##############
conv_pool_cnn <- keras_model_sequential() %>%
  layer_embedding(input_dim = max_words, output_dim = embedding_dim,
                  input_length = maxlen) %>%
  layer_conv_1d(filters = 96, kernel_size=3, activation='relu') %>%
  layer_conv_1d(filters = 96, kernel_size = 3,  activation='relu') %>%
  layer_max_pooling_1d(pool_size=3) %>%
  layer_conv_1d(filters = 192, kernel_size = 3, activation='relu', padding = 'same') %>%
  layer_conv_1d(filters = 192, kernel_size = 3, activation='relu', padding = 'same') %>%
  layer_max_pooling_1d(pool_size=(3)) %>%
  layer_conv_1d(filters = 192, kernel_size = 3, activation='relu', padding = 'same') %>%
  layer_conv_1d(filters = 192, kernel_size = 1, activation='relu') %>%
  layer_conv_1d(filters = 23, kernel_size = 1) %>%
  layer_global_max_pooling_1d() %>%
  layer_activation(23, activation='softmax')
summary(conv_pool_cnn)

compile_fit(conv_pool_cnn, epochs = 3)

F_score(table(y_val, predict_classes(conv_pool_cnn, x_val)))





########### Model 3 [ LSTM Conv Net ] 0.7334 ##############
model_test <- keras_model_sequential() %>%
  layer_embedding(input_dim = max_words, output_dim = embedding_dim,
                  input_length = maxlen) %>%
  layer_spatial_dropout_1d(0.2) %>%
  layer_cudnn_lstm(units = 128, return_sequences = TRUE) %>%
  layer_spatial_dropout_1d(0.5) %>%
  layer_conv_1d(filters = 64, kernel_size = 2, activation = 'relu') %>%
  layer_average_pooling_1d(pool_size = 2) %>%
  layer_max_pooling_1d(pool_size = 2) %>%
  layer_spatial_dropout_1d(0.2) %>%
  layer_flatten() %>%
  layer_dense(units = 23, activation = "softmax")
summary(model_test)

compile_fit(model_test, 3)

F_score(table(y_val, predict_classes(model_test, x_val)))



## Ensemble ##
en_prediction <- ensemble(val_data=x_val, w=c(0.5,0.3,0.2))

F_score(table(y_val, en_prediction))
# 0.7586 - 80k data - seed 1234 Actual Test run was 0.75
# 0.7652 - 100k data, seed 12345 c(0.5,0.3,0.2) 
# w=c(0.5,0.2,0.3) 0.7531831
# w=c(0.5,0.3,0.2) 0.7652686
table(y_val, en_prediction)












# # Saving Model
# save_model_hdf5(cbi_lstm, "./Model/blcnn_model", overwrite = TRUE,
#                 include_optimizer = TRUE)
# save_model_hdf5(conv_pool_cnn, "./Model/conv_pool_cnn_model", overwrite = TRUE,
#                 include_optimizer = TRUE)
# # Loading Model
# cbi_lstm <- load_model_hdf5('./Model/blcnn_model')
# conv_pool_cnn <- load_model_hdf5('./Model/conv_pool_cnn_model')



##############  Evaluation  #################
# cbi_lstm %>% evaluate(x_val, y_val)
# conv_pool_cnn %>% evaluate(x_val, y_val)


# Evaluation M1
#history
class_acc <-data.frame('Class'=(1:23), 'Accuracy'=rep(0,23))

# With Validation data
Y_test_predicted1 <- predict_classes(cbi_lstm, x_val)
cm_table_cbi_lstm <- table(y_val, Y_test_predicted1)
cm_table_cbi_lstm
class_accuracy(cm_table_cbi_lstm)
F_score(cm_table_cbi_lstm)
cbi_lstm_model_result <- class_accuracy(cm_table_cbi_lstm)


# Evaluation M2
class_acc <-data.frame('Class'=(1:23), 'Accuracy'=rep(0,23))

# With Validation data
Y_test_predicted2 <- predict_classes(conv_pool_cnn, x_val)
cm_table_conv_cnn <- table(y_val, Y_test_predicted2)
cm_table_conv_cnn
class_accuracy(cm_table_conv_cnn)
F_score(cm_table_conv_cnn)
conv_model_result <- class_accuracy(cm_table_conv_cnn)


# Evaluation M3
class_acc <-data.frame('Class'=(1:23), 'Accuracy'=rep(0,23))

# With Validation data
Y_test_predicted3 <- predict_classes(model_test, x_val)
cm_table_model_test <- table(y_val, Y_test_predicted3)
cm_table_model_test
class_accuracy(cm_table_model_test)
F_score(cm_table_model_test)



# Combined
run5 <- cbind(cbi_lstm_model_result, conv_model_result)
class_comb <- cbind(class_accuracy(cm_table_cbi_lstm), class_accuracy(cm_table_conv_cnn), class_accuracy(cm_table_model_test))
class_comb


cb_m <- as.numeric(gsub('%','',run1[,2])) + as.numeric(gsub('%','',run2[,2])) + as.numeric(gsub('%','',run3[,2]))+ as.numeric(gsub('%','',run4[,2]))+ as.numeric(gsub('%','',run5[,2]))
conv_m <- as.numeric(gsub('%','',run1[,4])) + as.numeric(gsub('%','',run2[,4])) + as.numeric(gsub('%','',run3[,4])) + as.numeric(gsub('%','',run4[,4]))+ as.numeric(gsub('%','',run5[,2]))
cbind(bl_cnn=cb_m, conv=conv_m, outperform=(conv_m-bc_m)/5)

#####



## Finalising Output
text <- readLines("testing_docs.txt")
text <- as.data.frame(text)
text <- as.data.frame(text[!(text$text=="EOD"),], header = FALSE)
colnames(text) <- c("text")
text <- text[!grepl("ID te_doc_",text$text),] # For test dataset
text <- as.data.frame(text)
text <- as.data.frame(text[!(text$text==""),],header= FALSE)
colnames(text) <- c("text")
text <- lapply(text,function(x) gsub("TEXT ","",x))
text <- as.data.frame(text)

# Pre-processing text
corpus <- Corpus(VectorSource(text$text))
corpus <- tm_map(corpus, removeNumbers)
corpus <- tm_map(corpus, content_transformer(tolower))
corpus <- tm_map(corpus, removePunctuation)
corpus <- tm_map(corpus, removeWords, stopwords('english'))
#corpus <- tm_map(corpus, content_transformer(stemDocument)) # Stemming didn't help
corpus <- tm_map(corpus, content_transformer(stripWhitespace))
cleaned.docs = data.frame(text=sapply(corpus,identity), stringsAsFactors = F)

head(cleaned.docs,1)



### Build Input Data
texts = cleaned.docs$text
tokenizer <- text_tokenizer(num_words = max_words) %>% # max_word = most common words
  fit_text_tokenizer(texts)

# Gether the sequences
sequences <- texts_to_sequences(tokenizer, texts)

# Gether word index
word_index = tokenizer$word_index

# Pad sequences to datafarme
test_data <- pad_sequences(sequences, maxlen = maxlen)

labels = as.array(label$label)
labels = as.numeric(gsub("C", "", labels))


# Prediction to output
final_prediction <- ensemble(val_data=test_data, w=c(0.6,0.2,0.2))

# Read Labels
testing_label <- readLines("testing_docs.txt")
Y_test_predicted<-data.frame(final_prediction)

doc <- data.frame(testing_label)
doc <- data.frame(label=testing_label[seq(from=1,to=nrow(doc),by=4)])
doc <- lapply(doc, function(x) gsub("ID ","",x))
doc <- data.frame(doc, stringsAsFactors = F)

for (line in 1:nrow(doc)){
  doc[line,] <- paste0(doc[line,], " C", Y_test_predicted[line,])
}
head(doc,2)


testing_labels_final1 <- write.table(doc, "C:/Users/abcd0/Documents/Uni/2018-2/FIT5149 Applied Data Analysis/Assignment/testing_labels_final1.txt", sep="\t", row.names = FALSE, quote = FALSE, col.names = FALSE)







######################################### BELOW IS FOR TESTING #################################################################

testing_labels_final <- readLines("training_labels_final.txt")
testing_labels_final1 <- readLines("testing_labels_final1.txt")

sum(cbind(testing_labels_final==testing_labels_final1))/length(testing_labels_final)

head(testing_labels_final)
head(testing_labels_final1)




head(doc)

Y_test_predicted <- data.frame(Y_test_predicted)
head(Y_test_predicted)
tail(Y_test_predicted)

cbind(doc,Y_test_predicted)


head(doc)




head(Y_test_predicted)
head(training_labels_final)


doc <- lapply(training_labels_final,function(x) gsub("C1","",x))

head(doc,1)



head(doc)



# F1
cm_table
history
F_score(cm_table)

class_accuracy(cm_table)







### Below needs to be completed

#data.correct <- data.val[which(data.val$y==data.val$predicted),]


# With train data for training model 2 use
Y_test_model1 <- predict_classes(model, x_train)
y_train <- labels[training_indices]
cm_table_m1 <- table(y_train, Y_test_model1)
cm_table_m1


data.val <- data.frame(y=y_train,x=x_train, predicted=Y_test_model1)
#data.correct <- data.val[which(data.val$y==data.val$predicted),]

head(data.val,2)




########### Model 2 ##############
# Create missclassified set
data.miss <- data.val[which(data.val$y!=data.val$predicted),]
data.miss <- data.miss[,1:101]

x_train_miss <- as.matrix(data.miss[,2:101])
y_train_miss <- as.matrix(data.miss[,1])

# Revert back to text
text_2 <- data.frame(y=y_train_miss, tokenizer$sequences_to_texts(x_train_miss))

# For each category ...
c2 <- text_2[text_2$y==2,]




#define model2
model2 <- keras_model_sequential() %>%
  layer_embedding(input_dim = max_words, output_dim = embedding_dim,
                  input_length = maxlen) %>%
  layer_flatten() %>%
  layer_dense(units = 1000, activation = "relu") %>%
  layer_dense(units = 500, activation = "relu") %>%
  layer_dense(units = 50, activation = "relu") %>%
  layer_dense(units = 50, activation = "relu") %>%
  layer_dense(units = 24, activation = "sigmoid")
summary(model)

# Compile model
model2 %>% compile(
  optimizer = "adam",
  loss = "categorical_crossentropy",
  metrics = c("acc")
)

# Early stopping
callback <- callback_early_stopping(monitor='val_loss', min_delta=0, patience=2, verbose=0, mode='min')

# Fitting model
history2 <- model2 %>% fit(
  x_train_miss, y_train_miss,
  epochs = 20,
  batch_size = 32,
  validation_split = (0.1)
  # callbacks = callback
)


model %>% evaluate(x_train_miss, y_train_miss)

Y_test_hat2 <- predict_classes(model2, x_train_miss)
cm_table2 <- table(data.miss[,1], Y_test_hat2)
cm_table2
mean(data.miss[,1] == Y_test_hat2)

cm_table3 <- cm_table + cm_table2
cm_table3


########## Evaluation #############


# Confusion matrix
# Combined table
cm_table_n <- cm_table3




## Save

#keras_save(mod, "full_model.h5")
#keras_save_weights(mod, "weights_model.h5")
#keras_model_to_json(mod, "model_architecture.json")


# install_keras(method = c("auto", "virtualenv", "conda"),
#               conda = "auto", version = "default", tensorflow = "default",
#               extra_packages = c("tensorflow-hub"))
#install_keras(tensorflow = "gpu")
# 
# https://keras.rstudio.com/reference/install_keras.html
