
library(tidyverse)
library(caret)
library(randomForest)
library(xgboost)
library(e1071)
library(ranger)
library(lubridate)
library(data.table)
library(ggplot2)
library(plotly)

## IMPORT DES DONNÉES 
data <- read.table(file = "/Users/achillegausseres/OneDrive/PRO/ACO/3A/Projet_Big_Data/dfplusIQA.csv", 
                   header = T, sep= ",", stringsAsFactors = T)


# Fonction qui renvoie 1 ou 0 en fonction des règles pour chaque profil
get_response <- function(IQA, profil) {
  if(profil == "Fumeur") {
    return(ifelse(IQA %in% c("dangereux"), 1, 0))
  } else if(profil == "Sans maladie") {
    return(ifelse(IQA %in% c("dangereux", "tres non-sain"), 1, 0))
  } else if(profil == "Asmatique") {
    return(ifelse(IQA == c("dangereux", "tres non-sain","non-sain"), 1, 0))
  } else if(profil == "Graves problemes respiratoires") {
    return(ifelse(IQA %in% c("dangereux", "tres non-sain","non-sain","non-sain pour sensibles"), 1, 0))
  } else if(profil == "Avec enfant") {
    return(ifelse(IQA == c("dangereux", "tres non-sain","non-sain","non-sain pour sensibles", "modere"), 1, 0))
  } else {
    return(NA)  # Si le profil n'est pas dans la liste
  }
}

data$IQA_binaire <- mapply(get_response, data$IQA, "Fumeur")

## REORGANISATION DU JEU DE DONNÉES   
data <- data[,-(6:11)]
data <- data[,-(13:20)]
data$IQA_binaire <- as.factor(data$IQA_binaire)



# Vérifier les valeurs manquantes
sapply(data, function(x) sum(is.na(x)))
data <- na.omit(data)
sapply(data, function(x) sum(is.na(x)))




# Utiliser data.table pour créer des lags par station
data <- as.data.table(data)
setkey(data, station, year, month, day, hour)

# Création des lags 
lag_days <- c(1,24, 168)

# Créer des lags pour chaque délai dans lag_days
for(i in lag_days){
  data[, paste0("TEMP_lag_", i, "h_") := shift(TEMP, n = i, type = "lag"), by = station]
  data[, paste0("PRES_lag_", i, "h_") := shift(PRES, n = i, type = "lag"), by = station]
  data[, paste0("DEWP_lag_", i, "h_") := shift(DEWP, n = i, type = "lag"), by = station]
  data[, paste0("RAIN_lag_", i, "h_") := shift(RAIN, n = i, type = "lag"), by = station]
  data[, paste0("wd_lag_", i, "h_") := shift(wd, n = i, type = "lag"), by = station]
  data[, paste0("WSPM_lag_", i, "h_") := shift(WSPM, n = i, type = "lag"), by = station]
}

# Convertir de nouveau en data.frame
data <- as.data.frame(data)

# Supprimer les premières lignes avec NA dues aux lags
data <- na.omit(data)


data <- data[,-(1:2)]
data <- data[,-2]
data <-data[,-9]

set.seed(123) # Pour la reproductibilité
trainIndex <- createDataPartition(data$IQA_binaire, p = 0.8, list = FALSE)
trainData <- data[trainIndex, ]
testData  <- data[-trainIndex, ]

#Méthode 1: Regression Multiple 

sapply(trainData, function(x) sum(is.na(x)))
any(is.na(data$y))       # Vérifie s'il y a des NA dans y
any(is.nan(data$y))      # Vérifie s'il y a des NaN dans y
any(is.infinite(data$y)) # Vérifie s'il y a des Inf dans y
trainData<- trainData[!is.na(trainData$IQA_binaire) & !is.nan(trainData$IQA_binaire) & !is.infinite(trainData$IQA_binaire), ]


mod.glm <- glm(formula = IQA_binaire ~., data = trainData, family = "binomial")

summary(mod.glm)

library(car)
library(plotROC)
Anova(mod.glm, type = 3, test.statistic = "LR", singular.ok = T)
pred.glm <- predict(mod.glm, newdata = testData[,-9], type = "response")
head(pred.glm)
head(testData[,9])
predclass.glm <- apply( data.frame(pred.glm) >=0.5 , 2, factor, labels=c("0", "1"))
head(c(predclass.glm))
head(testData[,9])




confusionMatrix(factor(c(predclass.glm)), 
                testData$IQA_binaire,
                positive = "1")

pred.glm <- predict(mod.glm, newdata = testData[,-9], type = "response")
seuils <- seq(from= 0.01,to = 0.99, length =1000)
TPR_test <- rep(0,times = length(seuils))
FPR_test <- TPR_test
for (k in 1:length(seuils)){
  
  class_pred <- ifelse(pred.glm > seuils[k], "Positif", "Négatif")
  table <- table(testData[,9], class_pred)
  tab <- round(prop.table(table, margin=1), 5)*100
  TPR_test[k] <- tab[2,2]
  FPR_test[k] <- tab[1,2]
  
}
plot(FPR_test, TPR_test, type = "l", main = "Courbe ROC Generalized Linear Model")





library(rpart)
library(rpart.plot)

arbre1<- rpart(IQA_binaire~., data= trainData, cp=0.000001)
print(arbre1)
library(rpart.plot)
plotcp(arbre1)
cp.opt <- arbre1$cptable %>% as.data.frame() %>% 
  filter(xerror == min(xerror)) %>% select(CP) %>% max() %>% as.numeric()

arbre.final <- prune(arbre1, cp = cp.opt)
predclass.arbre <- predict(arbre.final, newdata = testData, type ="class")

confusionMatrix(factor(c(predclass.arbre)), 
                testData$IQA_binaire,
                positive = "1")


predclass.arbreROC <- predict(arbre.final, newdata = testData, type ="prob")
seuils <- seq(from= 0.01,to = 0.99, length =1000)
TPR_test <- rep(0,times = length(seuils))
FPR_test <- TPR_test
for (k in 1:length(seuils)){
  
  class_pred <- ifelse(predclass.arbreROC[,2] > seuils[k], "1", "0")
  table <- table(testData[,9], class_pred)
  tab <- round(prop.table(table, margin=1), 5)*100
  TPR_test[k] <- tab[2,2]
  FPR_test[k] <- tab[1,2]
  
}
plot(FPR_test, TPR_test, type = "l", main = "Courbe ROC Arbre de Regression")

##COMPARAISON GLM ET ARBRE DE REGRESSION

score <- data.frame(arbre = predict(arbre.final, newdata = testData[,-9])[,2], 
                    glm = predict(mod.glm, newdata = testData[,-9], type ="response"), 
                    obs = testData[,9])



predprob.arbre <- predict(arbre.final, newdata = testData, type ="prob")
pred.glm <- predict(mod.glm, newdata = testData[,-9], type = "response")
seuils <- seq(from= 0.01,to = 0.99, length =1000)
TPR_arbre <- rep(0,times = length(seuils))
FPR_arbre <- TPR_test
TPR_glm <- TPR_test
FPR_glm <- TPR_test

for (k in 1:length(seuils)){
  
  class_pred_arbre <- ifelse(predclass.arbreROC[,2] > seuils[k], "1", "0")
  table <- table(testData[,9], class_pred_arbre)
  tab <- round(prop.table(table, margin=1), 5)*100
  TPR_arbre[k] <- tab[2,2]
  FPR_arbre[k] <- tab[1,2]
  class_pred_glm <- ifelse(pred.glm > seuils[k], "1", "0")
  table <- table(testData[,9], class_pred_glm)
  tab <- round(prop.table(table, margin=1), 5)*100
  TPR_glm[k] <- tab[2,2]
  FPR_glm[k] <- tab[1,2]
  
}

ggplot() +
  geom_line(data = ROC_comp, aes(x = FPR_glm, y = TPR_glm, color = "GLM"), size = 1) + 
  geom_line(data = ROC_comp, aes(x = FPR_arbre, y = TPR_arbre, color = "CART"), size = 1) + 
  scale_color_manual(values = c("GLM" = "blue", "CART" = "red")) +  
  labs(color = "Légende des courbes") +  
  theme_minimal() 

