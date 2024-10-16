#Import des données 
df_og <- read.table("/Users/achillegausseres/OneDrive/PRO/ACO/3A/Conf_Machine_Learning/Conf_Machine_Learning_repo/df.csv",
                    sep=",",
                    header=T,
                    stringsAsFactors = T)#jeu de données original
str(df_og)

#Visualisation des données manquantes
#install.packages("VIM")
library(VIM)
aggr(df_og)
matrixplot(df_og, sortby = 2)


compter_na_par_colonne <- function(tableau) {
  colSums(is.na(tableau))
}
compter_na_par_colonne(df_og)

#Imputation des données manquantes avec des forets aléatoires. 
#install.packages("missRanger")
library(missRanger)
impl <- missRanger(df_og, pmm.k = 5, num.trees = 100, seed = 1)

compter_na_par_colonne(impl)

save(impl, file = "dta_final.Rdata")


###



data_incomplete <- read.table("/Users/achillegausseres/OneDrive/PRO/ACO/3A/Conf_Machine_Learning/Conf_Machine_Learning_repo/df.csv",
                              sep=",",
                              header=T,
                              stringsAsFactors = T)#jeu de données original

introduce_pseudo_missing <- function(data, prop = 0.1) {
  data_missing <- data
  # Identifier les indices des valeurs observées
  observed_indices <- which(!is.na(data_missing), arr.ind = TRUE)
  
  # Nombre de valeurs à masquer
  n_missing <- floor(prop * nrow(observed_indices))
  
  # Sélectionner aléatoirement les indices à masquer
  selected_indices <- observed_indices[sample(1:nrow(observed_indices), n_missing), ]
  
  # Extraire les valeurs réelles pour l'évaluation
  true_values <- data_missing[selected_indices]
  print("Ok")
  print(nrow(selected_indices))
  
  # Introduire les valeurs manquantes
  for (i in 1:nrow(selected_indices)) {
    data_missing[selected_indices[i, "row"], selected_indices[i, "col"]] <- NA
  }
  
  list(data_missing = data_missing, 
       selected_indices = selected_indices, 
       true_values = true_values)
}

# Définir la grille des hyperparamètres
mtry_values <- c(1,  4,  6)
num_trees_fixed <- 50

# Créer une grille de paramètres
param_grid <- data.frame(mtry = mtry_values)


# Identifier les variables numériques et catégorielles
numeric_vars <- names(data_incomplete)[sapply(data_incomplete, is.numeric)]
categorical_vars <- names(data_incomplete)[sapply(data_incomplete, function(x) is.factor(x) | is.character(x))]

# Initialiser un data frame pour stocker les résultats
results <- data.frame(mtry = integer(),
                      RMSE = numeric(),
                      Accuracy = numeric(),
                      stringsAsFactors = FALSE)

# Introduire des valeurs manquantes pour la validation
validation <- introduce_pseudo_missing(data_incomplete, prop = 0.1)
data_train <- validation$data_missing
selected_indices <- validation$selected_indices
true_values <- validation$true_values

data_complete <- read.table("/Users/achillegausseres/OneDrive/PRO/ACO/3A/Conf_Machine_Learning/Conf_Machine_Learning_repo/df.csv",
                              sep=",",
                              header=T,
                              stringsAsFactors = T)#jeu de données original

# Convertir les variables catégorielles en facteurs si nécessaire
for (var in categorical_vars) {
  data_train[[var]] <- as.factor(data_train[[var]])
  data_complete[[var]] <- as.factor(data_complete[[var]])
}

# Boucle sur chaque valeur de mtry
for (m in mtry_values) {
  cat("Imputing with mtry =", m, "\n")
  
  # Imputer les données avec missRanger
  imputed_data <- missRanger(data_train, 
                             num.trees = num_trees_fixed,
                             pmm.k = 5,
                             mtry = m)
  
  # Initialiser les erreurs
  rmse_total <- 0
  rmse_count <- 0
  correct_preds <- 0
  total_categorical <- 0
  
  # Parcourir chaque valeur masquée
  for (i in 1:nrow(selected_indices)) {
    row <- selected_indices[i, "row"]
    col <- selected_indices[i, "col"]
    var_name <- names(data_incomplete)[col]
    
    if (var_name %in% numeric_vars) {
      # Calculer le RMSE pour les variables numériques
      error <- (data_complete[row, var_name] - imputed_data[row, var_name])^2
      rmse_total <- rmse_total + error
      rmse_count <- rmse_count + 1
    } else if (var_name %in% categorical_vars) {
      # Calculer l'Accuracy pour les variables catégorielles
      correct <- data_complete[row, var_name] == imputed_data[row, var_name]
      if (!is.na(correct)) { # Assurer que la comparaison n'est pas NA
        correct_preds <- correct_preds + as.integer(correct)
        total_categorical <- total_categorical + 1
      }
    }
  }
  
  # Calculer le RMSE moyen
  if (rmse_count > 0) {
    rmse_mean <- sqrt(rmse_total / rmse_count)
  } else {
    rmse_mean <- NA
  }
  
  # Calculer l'Accuracy
  if (total_categorical > 0) {
    accuracy <- correct_preds / total_categorical
  } else {
    accuracy <- NA
  }
  
  cat("  RMSE (numérique) =", rmse_mean, "\n")
  cat("  Accuracy (catégoriel) =", accuracy, "\n\n")
  
  # Stocker les résultats
  results <- rbind(results, data.frame(mtry = m, RMSE = rmse_mean, Accuracy = accuracy))
}

# Afficher les résultats
print(results)

# Résumer les résultats
summary_results <- results %>%
  arrange(RMSE) # Vous pouvez aussi considérer un compromis entre RMSE et Accuracy

print(summary_results)

# Sélectionner le meilleur mtry basé sur le RMSE et/ou l'Accuracy
# Par exemple, choisir le mtry avec le RMSE le plus bas
best_mtry_rmse <- summary_results$mtry[which.min(summary_results$RMSE)]
best_rmse <- min(summary_results$RMSE)

# Ou choisir basé sur l'Accuracy si pertinent
best_mtry_accuracy <- summary_results$mtry[which.max(summary_results$Accuracy)]
best_accuracy <- max(summary_results$Accuracy)

cat("Meilleur paramètre mtry basé sur RMSE:", best_mtry_rmse, "avec RMSE =", best_rmse, "\n")
cat("Meilleur paramètre mtry basé sur Accuracy:", best_mtry_accuracy, "avec Accuracy =", best_accuracy, "\n")

# Supposons que vous avez choisi best_mtry_rmse
final_imputed_data <- missRanger(data_incomplete, 
                                 num.trees = num_trees_fixed, 
                                 pmm.k = 5,
                                 mtry = best_mtry_rmse, 
                                 verbose = TRUE)

# Vérifier les résultats
summary(final_imputed_data)




