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

