library(kknn)

dataset = dataset_sift[,2:26]

dataset[,1:24] = scale(dataset[,1:24])

#Random Cross Validation
set.seed(2)
split = sample.split(dataset$y, SplitRatio = 0.7)

training_set = subset(dataset, split==TRUE)
test_set = subset(dataset, split==FALSE)

modelo <- train.kknn(as.factor(y) ~ ., data = training_set, kmax = 9)

predict = predict(modelo, test_set[,-25])

### Matriz Confusion
library(caret)

m = confusionMatrix(predict,as.factor(test_set$y))
accuracy = m$overall[1]
precision = m$byClass[5]
recall = m$byClass[6]
f1 = m$byClass[7]

library(cvms)

cfm <- as_tibble(m$table)

cvms::plot_confusion_matrix(cfm, 
                            target_col = "Reference", 
                            prediction_col = "Prediction",
                            counts_col = "n")