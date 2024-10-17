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

##### ordonner modalités #####

levels_order <- c("bon", "modéré", "non-sain pour sensibles", "non-sain", "très non-sain", "dangereux")

data$IQA <- factor(data$IQA, levels = levels_order) # mettre les modalités dans l'ordre


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

pred_class <- factor(pred_class, levels = seq_along(levels_order), labels = levels_order)

# matrice de confusion pour évaluer le modèle

confusionMatrix(pred_class, data.test$IQA, mode = "prec_recall")
table_confu <- confusionMatrix(pred_class, data.test$IQA)

conf_table <- as.data.frame(table_confu$table)

# calcul le pourcentage de repartition par classe de référence
conf_table <- conf_table %>%
  group_by(Reference) %>%
  mutate(Pourcentage = Freq / sum(Freq) * 100)

# Plot de la matrice de confusion
plot_confu <- ggplot(data = conf_table, aes(x = Prediction, y = Reference, fill = Pourcentage)) +
  geom_tile(color = "white") +  # Utilise geom_tile pour un effet de grille
  scale_fill_gradient(low = "white", high = "blue", limits = c(0, 100)) +  # Dégradé de couleur (blanc -> bleu)
  geom_text(aes(label = sprintf("%.1f%%", Pourcentage)), vjust = 1) +  # Ajouter les pourcentages
  labs(title = "Matrice de confusion", x = "Classe Prédite", y = "Classe Réelle") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) # Rotation des labels
  

plot_confu

#### enregistrement RData ####

save(data, file = "data_randomforest.RData")
save(foret, file = "mod_foret.RData")
save(table_confu, file = "table_confu.RData")


####### Test hyperparamètres #######
 
# Random search sur Ranger
# => trop long à tourner, on garde les valeurs par défaut utilisé par ranger

##############

# N'aide pas, tourne en boucle avec plus de répétition -> accuracy 47%
# Définir le contrôle de la validation croisée avec sur-échantillonnage
# train_control <- trainControl(method = "cv", number = 5, sampling = "up")  # "up" pour sur-échantillonnage

