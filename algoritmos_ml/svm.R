#Clasificacion Multiclase con SVM
install.packages("e1071")
install.packages("caTools")
install.packages("caret")
library(e1071)
library(caTools)

dataset = dataset_hog[,2:26]

#Random Cross Validation
set.seed(2)
split = sample.split(dataset$y, SplitRatio = 0.7)

training_set = subset(dataset, split==TRUE)
test_set = subset(dataset, split==FALSE)

#Modelo
modelo = svm(y~., data = training_set, kernel = "linear", type = 'C', scale = TRUE)

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

ggplot(data = training_set, aes(x = as.factor(y), y = x0, color = y)) +
  geom_boxplot(outlier.shape = NA) +
  geom_jitter(width = 0.1) +
  theme_bw() +
  theme(legend.position = "null") + 
  xlab("Y")