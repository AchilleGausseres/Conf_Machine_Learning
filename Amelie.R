library(data.table)
library(rsample)
library(ranger)
library(rpart)

## IMPORT DES DONNÉES 
data <- read.table(file = "dfplusIQA.csv", 
                   header = T, sep= ",", stringsAsFactors = T)


## REORGANISATION DU JEU DE DONNÉES   
data <- data[,-(6:11)]
data <- data[,-(13:19)]
data <- data[,-14]

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

data <- data[,-(1:5)]
data <-data[,-7]


######### Forêt aléatoire ########

### Découpage des données ###

set.seed(123)

data.split <- initial_split(data, prop = 3/4)
data.train <- training(data.split)
data.test <- testing(data.split)

foret <- ranger(IQA~., data = data.train, probability = T)

prediction <- predict(foret, data.test)

# Extraire la classe avec la probabilité la plus élevée pour chaque observation
pred_class <- apply(prediction$predictions, 1, which.max)

# Facultatif : convertir les numéros de classes en niveaux de facteurs (si IQA est un facteur)
levels_IQA <- levels(data$IQA)  # Récupérer les niveaux de la variable IQA
pred_class <- factor(pred_class, levels = seq_along(levels_IQA), labels = levels_IQA)

# matrice de confusion pour evaluer le modèle
conf_matrix <- table(Predicted = pred_class, Actual = data.test$IQA)
print("Confusion Matrix:")
print(conf_matrix)

library(dplyr)

# Analyse des classes avec dplyr
class_distribution <- data %>%
  group_by(IQA) %>%
  summarise(Count = n(), Percentage = n() / nrow(data) * 100)

print(class_distribution)

##### équilibrer les données #####

class_weights <- 1 / table(data$IQA)  # Inverser la fréquence
foret2 <- ranger(IQA ~ ., data = data.train, probability = TRUE, class.weights = class_weights)

prediction2 <- predict(foret2, data.test)

pred_class2 <- apply(prediction2$predictions, 1, which.max)

levels_IQA <- levels(data$IQA)  # Récupérer les niveaux de la variable IQA
pred_class2 <- factor(pred_class2, levels = seq_along(levels_IQA), labels = levels_IQA)

# matrice de confusion pour evaluer le modèle
conf_matrix2 <- table(Predicted = pred_class2, Actual = data.test$IQA)
print("Confusion Matrix:")
print(conf_matrix2)
