# Installation et chargement des packages
install.packages("keras")
library(keras)
install_keras()
library(tensorflow)
install_tensorflow()
library(lubridate)
library(dplyr)
library(caret)
library(ggplot2)

#Import du jeu de données et réorganisation des données temporelles 
data <- read.table(file = "/Users/achillegausseres/OneDrive/PRO/ACO/3A/Projet_Big_Data/dfplusIQA.csv", 
                   header = T, sep= ",", stringsAsFactors = T)
data <- data[,-(6:11)]
data <- data[,-(13:19)]
data <- data[,-14]
data <- data %>%
  arrange(year, month, day, hour) %>%
  mutate(Datetime = make_datetime(year = year, month = month, day = day, hour = hour))
data <- data %>%
  mutate(
    Sin_Heure = sin(2 * pi * hour / 24),
    Cos_Heure = cos(2 * pi * hour / 24),
    Sin_Mois = sin(2 * pi * month / 12),
    Cos_Mois = cos(2 * pi * month / 12),
    JourSemaine = wday(Datetime, label = FALSE),
    Sin_JourSemaine = sin(2 * pi * JourSemaine / 7),
    Cos_JourSemaine = cos(2 * pi * JourSemaine / 7))

#Encodage des stations en One Hot 
station_one_hot <- to_categorical(as.integer(data$Station) - 1, num_classes = 12)
features <- data %>%
  select(Feature1, Feature2, Feature3, Feature4, Feature5, Feature6, Sin_Heure, Cos_Heure, Sin_Mois, Cos_Mois)
features <- cbind(features, station_one_hot)

# Encodage de la cible
data$Target <- as.factor(data$Target)
y <- to_categorical(as.integer(data$Target) - 1, num_classes = 6)

# Normalisation des features
feature_means <- apply(features, 2, mean)
feature_sds <- apply(features, 2, sd)
features_scaled <- scale(features, center = feature_means, scale = feature_sds)

# Création des séquences
create_sequences <- function(features, targets, seq_length) {
  X <- list()
  y <- list()
  
  for (i in seq_len(nrow(features) - seq_length)) {
    X[[i]] <- features[i:(i + seq_length - 1), ]
    y[[i]] <- targets[i + seq_length, ]
  }
  
  array_X <- array(unlist(X), dim = c(length(X), seq_length, ncol(features)))
  array_y <- do.call(rbind, y)
  
  list(X = array_X, y = array_y)
}

## Ici on choisit la longeur des laps de temps étudiés
sequence_length <- 24
sequences <- create_sequences(features_scaled, y, sequence_length)
X <- sequences$X
y <- sequences$y

# Division des données
set.seed(123)
sample_size <- nrow(X)
train_size <- floor(0.7 * sample_size)
val_size <- floor(0.15 * sample_size)

X_train <- X[1:train_size,,]
y_train <- y[1:train_size, ]

X_val <- X[(train_size + 1):(train_size + val_size), , ]
y_val <- y[(train_size + 1):(train_size + val_size), ]

X_test <- X[(train_size + val_size + 1):sample_size, , ]
y_test <- y[(train_size + val_size + 1):sample_size, ]

# Construction du modèle
model <- keras_model_sequential() %>%
  layer_lstm(units = 100, input_shape = c(sequence_length, ncol(features_scaled)), return_sequences = FALSE) %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 100, activation = 'relu') %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 6, activation = 'softmax') # 6 classes

summary(model)

# Compilation du modèle
model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_adam(),
  metrics = c('accuracy')
)

# Définir les callbacks
callbacks_list <- list(
  callback_early_stopping(monitor = "val_loss", patience = 10),
  callback_reduce_lr_on_plateau(monitor = "val_loss", factor = 0.2, patience = 5)
)

# Entraînement du modèle
history <- model %>% fit(
  X_train, y_train,
  epochs = 100,
  batch_size = 64,
  validation_data = list(X_val, y_val),
  callbacks = callbacks_list
)

# Évaluation sur le test
score <- model %>% evaluate(X_test, y_test, verbose = 0)
cat('Test loss:', score$loss, '\n')
cat('Test accuracy:', score$accuracy, '\n')

# Prédictions
predictions <- model %>% predict(X_test)
predicted_classes <- apply(predictions, 1, which.max) - 1
true_classes <- apply(y_test, 1, which.max) - 1

predicted_classes <- factor(predicted_classes + 1, levels = 1:6)
true_classes <- factor(true_classes + 1, levels = 1:6)

# Matrice de confusion
confusionMatrix(predicted_classes, true_classes)

# Visualisation des performances
# Perte
plot_df <- data.frame(
  epoch = 1:length(history$metrics$loss),
  loss = history$metrics$loss,
  val_loss = history$metrics$val_loss
)

ggplot(plot_df, aes(x = epoch)) +
  geom_line(aes(y = loss, color = "Entraînement")) +
  geom_line(aes(y = val_loss, color = "Validation")) +
  labs(title = "Courbe de Perte", y = "Perte") +
  scale_color_manual("", values = c("Entraînement" = "blue", "Validation" = "red")) +
  theme_minimal()

# Précision
plot_df_acc <- data.frame(
  epoch = 1:length(history$metrics$accuracy),
  accuracy = history$metrics$accuracy,
  val_accuracy = history$metrics$val_accuracy
)

ggplot(plot_df_acc, aes(x = epoch)) +
  geom_line(aes(y = accuracy, color = "Entraînement")) +
  geom_line(aes(y = val_accuracy, color = "Validation")) +
  labs(title = "Courbe de Précision", y = "Précision") +
  scale_color_manual("", values = c("Entraînement" = "blue", "Validation" = "red")) +
  theme_minimal()
