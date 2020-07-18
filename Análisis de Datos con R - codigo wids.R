library(caret)
library(corrplot)
library(devtools)

getwd()
setwd("C:/Users/ferna/Documents/wids - machine learning")

# Leer el archivo csv
data_banco <- read.csv("bank-additional-full.csv",sep=";",header=TRUE)
str(data_banco)

#Explorar
summary(data_banco)

#Estadística descriptiva
g <- ggplot(data_banco, aes(age))
g + geom_histogram(aes(fill = factor(y)), binwidth = 10, position = 'fill') + 
  labs( x= 'Edad', y= 'Porcentaje', title= 'ProporciÃ³n aceptar depÃ³sito vs edad', fill= 'Suscribe \n deposito')

g <- ggplot(data_banco, aes(marital ))
g + geom_bar(aes(fill = factor(y)), position = 'fill') + 
  labs( x= 'Estado Civil', y= 'Porcentaje', title= 'ProporciÃ³n aceptar deposito vs Estado Civil', fill= 'Suscribe \n depÃ³sito')


#Eliminar variables que no aportan
table(data_banco$pdays)
data_banco$pdays<-NULL

data_banco$duration<-NULL

#Identificar valores faltantes
sum(is.na(data_banco))

#En caso de haber datos faltantes, se puede eliminar todo el registro o se puede predecirlo por medio del algoritmo KNN
data_preproc <- preProcess(data_banco, method = c("knnImpute","center","scale"))
data_banco <- predict(data_preproc, data_banco)
str(data_banco)

#Correlación en variables numéricas
corrplot(cor(data_banco[,c(1,11,12,14:18)]), method = "number")

data_banco$euribor3m<-NULL
data_banco$nr.employed<-NULL
data_banco$cons.price.idx<-NULL

corrplot(cor(data_banco[,c(1,11,12,14,15)]), method = "number")

#Convertir las variables 
data_banco[,c(2:10,13,16)]<-lapply(data_banco[,c(2:10,13,16)], as.factor)
str(data_banco)


#Ojo: En la vida real la limpieza de datos es la parte que mÃ¡s toma tiempo (valores mal escritos, valores aberrantes, valores incoherentes, etc)


data_banco$y<-ifelse(data_banco$y=="yes",1,0)

dmy <- dummyVars(" ~ .", data = data_banco,fullRank = T) 
data_banco <- data.frame(predict(dmy, newdata = data_banco))
str(data_banco)

data_banco$y<-as.factor(ifelse(data_banco$y==1,"Si","No"))


## Crear train - test
## createDataPartition funcion de caret, genera Ã­ndices para muestra aleatoria
set.seed(123)
indice <- createDataPartition(data_banco$y, p= 0.80, list = FALSE)

## train y test
df_train <- data_banco[indice,]
df_test <- data_banco[-indice,]

str(df_train)

# Hacer un modelo lineal usando cross-validation
# de 5-fold,
# Crear trainControl
myControl <- trainControl(
  summaryFunction = twoClassSummary,  
  classProbs = TRUE, 
  verboseIter = TRUE,
  savePredictions = TRUE, 
  method = "cv", number = 5
)

#Modelo de regresiÃ³n logÃ­stica
set.seed(123)
modelo_logistico <- train(
  y ~ ., 
  df_train,
  method = "glm",
  family= "binomial",
  trControl = myControl
)

#Modelo de redes neuronales
modelo_nnet <- train(
  y ~ ., 
  df_train,
  method = "nnet",
  trControl = myControl
)


#PresentaciÃ³n del modelo
modelo_logistico
modelo_nnet

#Variables importantes para el modelo
plot(varImp(modelo_logistico), main="RegresiÃ³n logÃ­stica")
plot(varImp(modelo_nnet), main="Redes neuronales")


#PredicciÃ³n
predic_log<-predict.train(object=modelo_logistico,df_test[,-49],type="raw")
table(predic_log)

predic_nnet<-predict.train(object=modelo_nnet,df_test[,-49],type="raw")
table(predic_nnet)

confusionMatrix(predic_log,df_test[,49])
confusionMatrix(predic_nnet,df_test[,49])
