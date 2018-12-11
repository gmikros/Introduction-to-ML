library(stylo)
library(caret)
library(tibble)


# setting a working directory that contains the corpus. All text files should contain first
# the Author's name followed by underscore and the title of the text. Alternatively, the 
# first name shoule indicate the classification category of the text e.g. author's gender,
# author's age category, text's genre etc.
setwd("/Users/gmikr/Documents/corpus")

# we want all functions that use random values to use the exact same values each time they run
set.seed(1)

# loading the files from a specified directory and coumting the stylometric features.We also split the files in 1000 words text chunks
bigramch.texts = load.corpus.and.parse(files = "all", sample.size = 1000, sampling = "normal.sampling", features = "c", ngram.size = 2, encoding = "UTF-8")
trigramch.texts = load.corpus.and.parse(files = "all", sample.size = 1000, sampling = "normal.sampling", features = "c", ngram.size = 3, encoding = "UTF-8")
words.texts = load.corpus.and.parse(files = "all", sample.size = 1000, sampling = "normal.sampling", encoding = "UTF-8")
bigramw.texts = load.corpus.and.parse(files = "all", sample.size = 1000, sampling = "normal.sampling", features = "w", ngram.size = 2, encoding = "UTF-8")

# computing a list of most frequent features (trimmed to top n items):
feature.count = 50
features.bigramch = make.frequency.list(bigramch.texts, head = feature.count)
features.trigramch = make.frequency.list(trigramch.texts, head = feature.count)
features.words = make.frequency.list(words.texts, head = feature.count)
features.bigramw = make.frequency.list(bigramw.texts, head = feature.count)

# producing a table of relative frequencies:
bigramch = make.table.of.frequencies(bigramch.texts, features.bigramch, relative = TRUE)
trigramch = make.table.of.frequencies(trigramch.texts, features.trigramch, relative = TRUE)
words = make.table.of.frequencies(words.texts, features.words, relative = TRUE)
bigramw = make.table.of.frequencies(bigramw.texts, features.bigramw, relative = TRUE)

# renaming column names with Cbg (Character BiGrams) and a sequential number. Cbg_1 is the most
# frequent character bigram, Cbg_2 is the second most frequent etc. Ctg stands for Character TriGrams
# Wug for Word UniGrams and Wbg for Word BiGrams.
colnames(bigramch) <- c(paste0("Cbg_", 1:feature.count))
colnames(trigramch) <- c(paste0("Ctg_", 1:feature.count))
colnames(words) <- c(paste0("Wug_", 1:feature.count))
colnames(bigramw) <- c(paste0("Wbg_", 1:feature.count))

# concatening the feature matrices and converting them to dataframe
data <- cbind(bigramch, trigramch, words, bigramw)
data <- as.data.frame(data)

# convert rownames to a columng with the name "Author". You can change it to the name of the class
data <-rownames_to_column(data, var = "Author")

# removes the last part of the row name and keeps the author information or whatever exists
# before the underscore
data$Author <- gsub(pattern = "_.+", replacement = "", x = data$Author)

# splitting the training and the testing data. You can write an author's name or other value
# the "split.value" and use it as a filter for creating the testing dataframe.
split.value = "Unknown"
data_test <- data[data$Author == split.value, ]
data_train<-data[!(data$Author == split.value), ]
data_train$Author <- as.factor(data_train$Author)

# reducing the data size by eliminating the low variance features
nzv <- nearZeroVar(data_train)
data_train <- data_train[, -nzv]

# controlling the cross-validation procedure
fitControl<-trainControl(method="cv", number= 5, classProbs =  TRUE,  returnResamp="all")

# training and tuning the ML algorithm (SVM)
SVM <- train(Author ~., data=data_train, method="svmPoly", preProcess=c("center", "scale"), trControl=fitControl, tuneLength= 3,importance=TRUE)
SVM
confusionMatrix(SVM, "average")
plot(SVM)

Pred_SVM <- predict(SVM, newdata= data_test)
Pred_SVM
tb <- table(Pred_SVM)
tb


# training and tuning the ML algorithm (Random Forest)
RF <- train(Author ~., data=data_train, method="rf", preProcess=c("center", "scale"), trControl=fitControl, tuneLength= 3,importance=TRUE)
RF
confusionMatrix(RF, "average")
plot(RF)

Pred_RF <- predict(RF, newdata= data_test)
Pred_RF
tb <- table(Pred_RF)
tb


# we can compare the performance of the developed models using statistical significance testing
cvValues <- resamples(list(RF = RF, SVM = SVM))
summary(cvValues)
dotplot(cvValues)


