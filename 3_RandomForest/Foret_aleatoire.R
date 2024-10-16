library(data.table)
library(rsample)
library(ranger)
library(rpart)
library(caret)
library(dplyr)

## IMPORT DES DONNÉES 
data <- read.table(file = "dfplusIQA_NAcomplet.csv", 
                   header = T, sep= ",", stringsAsFactors = T, encoding = "utf-8")


## REORGANISATION DU JEU DE DONNÉES   
data <- data[,-(6:11)]    # enlever 6 facteurs de pollution
data <- data[,-(13:18)]   # enelever IQA par facteur


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

data <- data[,-(1:5)]   # enleve données temporelles
data <-data[,-7]        # enleve station


######### Forêt aléatoire ########

### Découpage des données 80/20 ###

set.seed(123) # pour la reproductibilité

data.train <- data[1:336615,]
data.test <- data[-(1:336615),]


# Analyse des classes avec dplyr
class_distribution <- data %>%
  group_by(IQA) %>%
  summarise(Count = n(), Percentage = n() / nrow(data) * 100)

print(class_distribution)


##### équilibrer les données #####

class_weights <- 1 / (table(data$IQA))
class_weights <- class_weights/sum(class_weights)

foret <- ranger(IQA ~ ., data = data.train, probability = TRUE, class.weights = class_weights, importance = "impurity")

prediction <- predict(foret, data.test)

pred_class <- apply(prediction$predictions, 1, which.max)

levels_IQA <- levels(data$IQA)  # Récupérer les niveaux de la variable IQA
pred_class <- factor(pred_class, levels = seq_along(levels_IQA), labels = levels_IQA)

# matrice de confusion pour évaluer le modèle

confusionMatrix(pred_class, data.test$IQA, mode = "prec_recall")
table_confu <- confusionMatrix(pred_class, data.test$IQA)


#### enregistrement RData ####

save(data, file = "data_randomforest.RData")
save(foret, file = "mod_foret.RData")
save(table_confu, file = "table_confu.RData")


# Test hyperparamètres

weights_vector <- class_weights[as.character(data.train$IQA)]
control <- trainControl(method = "cv", number = 5, search = "random")

# Random search sur Ranger
# model <- train(IQA ~ ., 
#                data = data.train, 
#                method = "ranger",
#                weights = weights_vector,
#                trControl = control, 
#                tuneLength = 3) # Essayez 3 configurations différentes

# => trop long à tourner, on garde les valeurs prise par ranger pour faire le modèle


##############

# N'aide pas, tourne en boucle avec plus de répétition -> accuracy 47%
# Définir le contrôle de la validation croisée avec sur-échantillonnage
# train_control <- trainControl(method = "cv", number = 5, sampling = "up")  # "up" pour sur-échantillonnage

# importance des variables


