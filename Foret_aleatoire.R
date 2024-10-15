library(data.table)
library(rsample)
library(ranger)
library(rpart)
library(caret)
library(dplyr)

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
lag_days <- c(1,2,3,4,5,6,12,18,24)

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

<<<<<<< HEAD:Amelie.R
library(dplyr)

=======
>>>>>>> f1d6cb227ba84b1f190207702313f84982312e11:Foret_aleatoire.R
# Analyse des classes avec dplyr
class_distribution <- data %>%
  group_by(IQA) %>%
  summarise(Count = n(), Percentage = n() / nrow(data) * 100)

print(class_distribution)

##### équilibrer les données #####

class_weights <- 1 / table(data$IQA)  # Inverser la fréquence
class_weights <- class_weights/sum(class_weights)

foret <- ranger(IQA ~ ., data = data.train, probability = TRUE, class.weights = class_weights)

prediction <- predict(foret, data.test)

pred_class <- apply(prediction$predictions, 1, which.max)

levels_IQA <- levels(data$IQA)  # Récupérer les niveaux de la variable IQA
pred_class <- factor(pred_class, levels = seq_along(levels_IQA), labels = levels_IQA)
pred_class <- factor(pred_class, levels = levels(data.test$IQA))

# matrice de confusion pour évaluer le modèle

confusionMatrix(pred_class2, data.test$IQA, mode = "prec_recall")
confusionMatrix(pred_class2, data.test$IQA)

##############

<<<<<<< HEAD:Amelie.R
#  Marche pas -> accuracy 47%
=======
# N'aide pas, tourne en boucle avec plus de répétition -> accuracy 47%
>>>>>>> f1d6cb227ba84b1f190207702313f84982312e11:Foret_aleatoire.R
# Définir le contrôle de la validation croisée avec sur-échantillonnage
# train_control <- trainControl(method = "cv", number = 5, sampling = "up")  # "up" pour sur-échantillonnage

# A faire
<<<<<<< HEAD:Amelie.R

# tester hyperparamètres :
# - taille arbres
# - nombres d'arbres
# - taille noeuds
=======
# testes hyperparametre :
       # - taille arbres
       # - nombre d'arbre
       # - taille noeud
# importance des variables
>>>>>>> f1d6cb227ba84b1f190207702313f84982312e11:Foret_aleatoire.R



