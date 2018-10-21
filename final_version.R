## Install packages
#install.packages("keras")
#install_keras()
#install.packages('curl')
#install.packages("tm")
#install.packages("qdap")
#install.packages("tm)

library(keras)
install_keras(tensorflow = "gpu") # gpu version must be used
library(tm)
library(ggplot2)

set.seed(12345) # to remove randomness

# Setting current working directory
setwd("~/Uni/2018-2/FIT5149 Applied Data Analysis/Assignment")


## Load Data
# Load text
text <- readLines("training_docs.txt")
text <- as.data.frame(text)
text <- as.data.frame(text[!(text$text=="EOD"),], header = FALSE)
colnames(text) <- c("text")
text <- text[!grepl("ID tr_doc_",text$text),] # For train dataset
text <- as.data.frame(text)
text <- as.data.frame(text[!(text$text==""),],header= FALSE)
colnames(text) <- c("text")
text <- lapply(text,function(x) gsub("TEXT ","",x))
text <- as.data.frame(text)

# Load labels
label <- as.data.frame(readLines("training_labels_final.txt"))
label$labels <- sub(".* ", "",label$`readLines("training_labels_final.txt")`)
label$`readLines("training_labels_final.txt")` <- sub(" .*", "",label$`readLines("training_labels_final.txt")`)
colnames(label) <- c("doc","label")


## Initialise
# Word vector
maxlen <- 200 # Input documents have 200 words each
max_words <- 50000  # Size of the featues in the text data. original = 50000
embedding_dim <- 200 # Dim size of embedding matrix


## Custom Functions
# Calculate F score
F_score <- function(cm_table){
  
  # Recall
  my_recall <- vector(mode="numeric", length=23)
  for (i in 1:23){
    my_recall[i] <- cm_table[i,i]/sum(cm_table[,i]) # Uses Confusion matrix
  }
  
  # Precision
  my_precision <- vector(mode="numeric", length=23)
  for (i in 1:23){
    my_precision[i] <- cm_table[i,i]/sum(cm_table[i,])
  }
  
  my_F1 <- vector(mode="numeric", length=23)
  for (i in 1:23){
    # Calculates based on formula in assignment details
    my_F1[i] <- 2*((my_recall[i]*my_precision[i]) / (my_recall[i]+my_precision[i]))
  }
  Final_F1 <- mean(my_F1)
  
  return (Final_F1)
  
} 

# Class Accuracy
class_accuracy <- function(cm_table){
  for (i in 1:ncol(cm_table)){
    # Takes confusion matrix as input, find accuracy of the model for each class
    class_acc[i,2] <- (sum(cm_table[i,i]) / sum(cm_table[,i]))
  }
  return (class_acc)
}

# Early stopping
# If validation accuracy does not improve, traning terminates 
es_callback <- callback_early_stopping(monitor='val_acc', min_delta=0, patience=2, verbose=0)


# Compile + Fitting model
# Compiles the model and fits, returns history(results of the training)
compile_fit <- function(model, epochs=10, batch=32, split_ratio=0.2, optimizer="rmsprop"){
  model %>% compile(
    optimizer = optimizer,
    loss = "categorical_crossentropy", # Multi class classification
    metrics = c("acc")
  )
  history <- model %>% fit(
    x_train, y_train,
    epochs = epochs,
    batch_size = batch,
    callbacks = list(es_callback), # Early stopping used to prevent overfitting
    # This option is to produce fixed outcome, however in reality this should not be used as it will cause
    # Over fitting problem for the model
    # Pretrained weights are also provided to produce reproducible output.
    # Also Using shuffle=TRUE may improve or worsen the prediction. 
    shuffle = FALSE, 
    validation_split = split_ratio # 8:2 ratio is set as default
  )
  return (history)
}



# Ensemble
# Emsemble function takes validation data, weights as input
# Apply input weights to each model, generates weighted average prediction
ensemble <- function(val_data = 'x_val', w = c(0.5, 0.30, 0.20)){
  
  # Initialise
  N <- nrow(val_data)
  w = w # weighting

  # Reset existing
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
  # Maximum probability will be chosen for each class
  return(apply(cs, MARGIN = 1,FUN = which.max))
}




# Pre-processing text
corpus <- Corpus(VectorSource(text$text))
corpus <- tm_map(corpus, removeNumbers)
corpus <- tm_map(corpus, content_transformer(tolower))
corpus <- tm_map(corpus, removePunctuation)
corpus <- tm_map(corpus, removeWords, stopwords('english'))
corpus <- tm_map(corpus, content_transformer(stripWhitespace))
cleaned.docs = data.frame(text=sapply(corpus,identity), stringsAsFactors = F)

head(cleaned.docs,1)

# Only used for overviewing the features
# DTM_train <- DocumentTermMatrix(corpus, control=list(wordLengths=c(4,Inf)))
# DTM_train # Number of features = terms: 159389
# DTM_train <- removeSparseTerms(DTM_train,0.95)
# DTM_train # Number of features = terms: 493
# tfm <- weightTfIdf(TermDocumentMatrix(corpus[training_indices]))

# Tokenise texts
texts = cleaned.docs$text
tokenizer <- text_tokenizer(num_words = max_words) %>% # max_word = most common words
    fit_text_tokenizer(texts)

# Gether the sequences
sequences <- texts_to_sequences(tokenizer, texts)

# Gether word index
word_index = tokenizer$word_index

# Pad sequences to datafarme
data <- pad_sequences(sequences, maxlen = maxlen)

# Clean labels
labels = as.array(label$label)
labels = as.numeric(gsub("C", "", labels))


## Data Split
# Random selection
set.seed(12) # set seed to  fix the random indices 12345
indices <- sample(1:nrow(data))
training_samples = 100000 # Using 100k samples to train model
training_indices <- indices[1:training_samples]
validation_indices <- indices[(training_samples + 1):nrow(data)]

# Split into train / validation
x_train <- data[training_indices,]
y_train <- labels[training_indices]
y_train <- y_train-1 # To make class category 1:23
y_train = to_categorical(y_train,num_classes = length(unique(y_train)))
x_val <- data[validation_indices,]
y_val <- labels[validation_indices]


##
## Note that training takes some time, so I have included pre-trained weights
## which can be loaded and used.
##


## If needed, one can just run below codes to load pre trained models
# cbi_lstm <- load_model_hdf5('cbi_lstm_model_grp90')
# conv_pool_cnn <- load_model_hdf5('conv_pool_cnn_grp90')
# model_test <- load_model_hdf5('model_test_grp90')



########### Model 1 [ Convolutional Bi-directional LSTM ] ##############
# 0.7487 epoch 3, 100k data
# 0.7527 epoch 3, 100k data - saved model
# 0.7607 epoch 3, 100k data
# 0.7552 epoch 3, 100k data
# 0.7476 epoch 3, 100k data
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

# Compiling and fitting model
compile_fit(cbi_lstm, epochs = 3)

# Generates F-score
F_score(table(y_val, predict_classes(cbi_lstm, x_val)))


# # Saving Model
# Model Saved / Loaded in current working directory
# save_model_hdf5(cbi_lstm, "cbi_lstm_model_grp90", overwrite = TRUE,
#                 include_optimizer = TRUE)



########### Model 2 [ ConvPool-CNN-C ] ##############
# 0.7213 epoch 3, 100k data
# 0.7265 epoch 3, 100k data
# 0.7279 epoch 3, 100k data
# 0.7124 epoch 3, 100k data
# 0.6962 epoch 3, 70k data
conv_pool_cnn <- keras_model_sequential() %>%
  layer_embedding(input_dim = max_words, output_dim = embedding_dim,
                  input_length = maxlen) %>%
  layer_conv_1d(filters = 96, kernel_size=3, activation='relu') %>%
  layer_conv_1d(filters = 96, kernel_size = 3,  activation='relu') %>%
  layer_max_pooling_1d(pool_size=3) %>%
  layer_conv_1d(filters = 192, kernel_size = 3, activation='relu', padding = 'same') %>%
  layer_conv_1d(filters = 192, kernel_size = 3, activation='relu', padding = 'same') %>%
  layer_max_pooling_1d(pool_size=3) %>%
  layer_conv_1d(filters = 192, kernel_size = 3, activation='relu', padding = 'same') %>%
  layer_conv_1d(filters = 192, kernel_size = 1, activation='relu') %>%
  layer_conv_1d(filters = 23, kernel_size = 1) %>%
  layer_global_max_pooling_1d() %>%
  layer_activation(23, activation='softmax')
summary(conv_pool_cnn)

compile_fit(conv_pool_cnn, epochs = 3)

F_score(table(y_val, predict_classes(conv_pool_cnn, x_val)))

# # Saving Model
# Model Saved / Loaded in current working directory
# save_model_hdf5(cbi_lstm, "conv_pool_cnn_grp90", overwrite = TRUE,
#                 include_optimizer = TRUE)







########### Model 3 [ LSTM Conv Net ] ##############
# 0.7324 epoch 3, 100k data
# 0.7456 epoch 3, 100k data
# 0.7327 epoch 4, 100k data
# 0.7382 epoch 4, 100k data
# 0.7234 epoch 3, 70k data
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

compile_fit(model_test, 4)

F_score(table(y_val, predict_classes(model_test, x_val)))


# # Saving Model
# Model Saved / Loaded in current working directory
# save_model_hdf5(model_test, "model_test_grp90", overwrite = TRUE,
#                 include_optimizer = TRUE)




## Ensemble ##
# Ensembling 3 models is done by using custom ensemble function

# 0.7586 - 80k data - seed 1234 'Actual Test run' was 0.757
# 0.7652 - 100k data, seed 12345 c(0.5,0.3,0.2)
# 0.7623 - 100k data, seed 12345 c(0.5,0.3,0.2)
# 0.7552 - 100k data, seed 12345 c(0.5,0.3,0.2) no shuffle
# 0.7616 - 100k data, seed 12 c(0.5,0.3,0.2) no shuffle <- used


en_prediction <- ensemble(val_data=x_val, w=c(0.5,0.3,0.2))
f_ensemble <- F_score(table(y_val, en_prediction))
f_ensemble

table(y_val, en_prediction)




## Generating Output
text <- readLines("testing_docs.txt")
## Load Data
text <- as.data.frame(text)
text <- as.data.frame(text[!(text$text=="EOD"),], header = FALSE)
colnames(text) <- c("text")
text <- text[!grepl("ID te_doc_",text$text),] # For train dataset
text <- as.data.frame(text)
text <- as.data.frame(text[!(text$text==""),],header= FALSE)
colnames(text) <- c("text")
text <- lapply(text,function(x) gsub("TEXT ","",x))
text <- as.data.frame(text)
head(text,3)

# Pre-processing text
corpus <- Corpus(VectorSource(text$text))
corpus <- tm_map(corpus, removeNumbers)
corpus <- tm_map(corpus, content_transformer(tolower))
corpus <- tm_map(corpus, removePunctuation)
corpus <- tm_map(corpus, removeWords, stopwords('english'))
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


# Prediction to output
final_prediction <- ensemble(val_data=test_data, w=c(0.5,0.3,0.2))
head(final_prediction)


# Read Labels
testing_label <- readLines("testing_docs.txt")
Y_test_predicted <- data.frame(final_prediction)

doc <- data.frame(testing_label)
doc <- data.frame(label=testing_label[seq(from=1,to=nrow(doc),by=4)])
doc <- lapply(doc, function(x) gsub("ID ","",x))
doc <- data.frame(doc, stringsAsFactors = F)

for (line in 1:nrow(doc)){
  doc[line,] <- paste0(doc[line,], " C", Y_test_predicted[line,])
}
head(doc,2)


testing_labels_pred <- write.table(doc, "testing_labels_pred.txt", sep="\t", row.names = FALSE, quote = FALSE, col.names = FALSE)
testing_labels_pred2 <- write.table(doc, "testing_labels_pred.csv", sep="\t", row.names = FALSE, quote = FALSE, col.names = FALSE)

