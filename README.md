# Conf_Machine_Learning

Préparation d'une conférence sur le Machine Learning pour le Master de l'Institut Agro de Rennes spécialisation mathématiques appliquées et statistiques 

## Comparaison entre méthodes de Machine Learning et Deep Learning pour prédire l’indice de qualité de l’air à partir de données météo


La qualité de l'air est devenue un enjeu majeur pour les grandes villes, avec des impacts significatifs sur la santé publique et l'environnement. Face à l'augmentation des sources de pollution et à l'évolution rapide des conditions météorologiques, prédire l'indice de qualité de l'air est un défi essentiel pour anticiper les pics de pollution et informer les populations. Ce projet s'appuie sur des données météorologiques et environnementales pour améliorer la prévision des indices de qualité de l'air.

Le jeu de données utilisé pour cette étude contient l’évolution de six facteurs météorologiques et six facteurs de pollution mesurés toutes les heures dans 12 stations à Pékin entre 2013 et 2017. Les facteurs météorologiques mesurés sont : la température de l’air, la pression atmosphérique, la température de rosée, les précipitations, la direction du vent, la vitesse du vent. Les polluants atmosphériques mesurés (en concentration) sont : les particules fines dont le diamètre est inférieur à 2.5 microns, les particules fines dont le diamètre est inférieur à 10 microns, le dioxyde de sodium, le dioxyde d’azote, le monoxyde de carbone, l’ozone.

L’object de cette étude est de déterminer quelle approche de modélisation prédictive (entre le machine learning et le deep learning) permettrait d'améliorer la précision des prévisions de l'indice de qualité de l'air, et ainsi de mieux protéger la santé globale dans le cadre d’une approche One Health.

Nous comparerons trois modèles prédictifs : deux modèles de machine learning, Random Forest et ARIMA, ainsi qu'un modèle de deep learning, le réseau de neurones récurrent LSTM (Long Short-Term Memory). Chaque modèle sera évalué en termes de précision, robustesse, et capacité à capturer les dynamiques complexes des séries temporelles.

Les résultats permettront d'identifier les points forts et les limites de chaque méthode pour orienter les choix futurs en fonction des besoins spécifiques en termes de précision et de performance de calcul.


# Mots-clés : pollution, prévision, séries temporelles, réseaux de neurones récurrents. 

Données : https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data

Articles : https://www.semanticscholar.org/paper/Cautionary-tales-on-air-quality-improvement-in-Zhang- Guo/59c99a7bf19617b43be0aa9f492def8c80ffae19
https://www.semanticscholar.org/paper/Evaluation-of-Time-Series-Forecasting-Models-for-of-Garg- Jindal/3b81dbabec2b6f32153d59784e7bcb2249ce8091

Derya Kapisiz, Amélie Brejot, Achille Gausserès.
