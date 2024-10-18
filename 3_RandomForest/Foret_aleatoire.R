library(data.table)
library(rsample)
library(ranger)
library(rpart)
library(caret)
library(dplyr)

######### IMPORT DES DONNÉES #########

data <- read.table(file = "dfplusIQA_NAcomplet.csv", 
                   header = T, sep= ",", stringsAsFactors = T, encoding = "utf-8")


######### REORGANISATION DU JEU DE DONNÉES #########

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


######### Ordonner les modalités #########

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

# La modalité "dangereux" est sur-représentée (41%), ce qui pourrais amener le modèle 
# à prédire davantage cette modalité au détriment des autres.
# On peut ajouter des poids pour essayer d'équilibrer les classes lors de la prédiction.


######### Equilibrer les données #########

class_weights <- 1 / (table(data$IQA))
class_weights <- class_weights/sum(class_weights)

foret <- ranger(IQA ~ ., data = data.train, probability = TRUE, 
                class.weights = class_weights, importance = "impurity")

prediction <- predict(foret, data.test)

pred_class <- apply(prediction$predictions, 1, which.max)

pred_class <- factor(pred_class, levels = seq_along(levels_order), 
                     labels = levels_order)

# Matrice de confusion pour évaluer le modèle

confusionMatrix(pred_class, data.test$IQA, mode = "prec_recall")
table_confu <- confusionMatrix(pred_class, data.test$IQA)

conf_table <- as.data.frame(table_confu$table)

# Calcul le pourcentage de repartition par classe de référence
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


######## Importance des variables ########

# Extraire l'importance des variables
importance_values <- foret$variable.importance

# Convertir en dataframe pour faciliter l'affichage
importance_data <- data.frame(Variable = names(importance_values),
                            Importance = importance_values)

# Trier par ordre décroissant d'importance
importance_data <- importance_data[order(importance_data$Importance, decreasing = TRUE),]

# Afficher l'importance des variables dans un graphique
plot_imp <- ggplot(importance_data, aes(x = reorder(Variable, Importance), y = Importance)) +
  geom_bar(stat = "identity", fill = "skyblue") +
  coord_flip() +  # Inverser les axes pour une meilleure lisibilité
  xlab("Variables") + 
  ylab("Importance (Impureté)") + 
  ggtitle("Importance des variables basée sur l'impureté") +
  theme_minimal()

plot_imp

# Affiche l'importance des varaibles, les variables avec une grande valeur joue un role 
# important dans la division des noeuds (diminution de l'impureté)
# On peut voir que la variable RAIN (mesure de précipitation) n'a que très peu de role
# dans la prédiction. On peut donc faire le modèle sans RAIN sans impacter la taux de prédiction.


#### Enregistrement RData ####

save(data, file = "data_randomforest.RData")
save(foret, file = "mod_foret.RData")
save(table_confu, file = "table_confu.RData")


####### Test hyperparamètres #######
 
# Random search sur Ranger
# => trop long à tourner, on garde les valeurs par défaut utilisé par ranger

##############

# Rééchantillonnage
# N'aide pas, tourne en boucle avec plus de répétition -> accuracy 47%
# Définir le contrôle de la validation croisée avec sur-échantillonnage
# train_control <- trainControl(method = "cv", number = 5, sampling = "up")  # "up" pour sur-échantillonnage

