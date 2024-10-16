## L'objectif de ce code est d'importer les differents dataset de la base de donnee 
## puis le reorganiser et enfin imputer les valeurs manquantes. 

library(dplyr)
library(zoo)
library(VIM)
library(missRanger)

df1<-read.table("PRSA_Data_Aotizhongxin_20130301-20170228.csv",sep = ",",header = TRUE,
               stringsAsFactors = TRUE)
df2<-read.table("PRSA_Data_Changping_20130301-20170228.csv",sep=",",header = TRUE,stringsAsFactors = TRUE)
df3<-read.table("PRSA_Data_Dingling_20130301-20170228.csv",sep = ",",header = TRUE,
               stringsAsFactors = TRUE)
df4<-read.table("PRSA_Data_Dongsi_20130301-20170228.csv",sep=",",header = TRUE,stringsAsFactors = TRUE)
df5<-read.table("PRSA_Data_Guanyuan_20130301-20170228.csv",sep = ",",header = TRUE,
                stringsAsFactors = TRUE)
df6<-read.table("PRSA_Data_Gucheng_20130301-20170228.csv",sep=",",header = TRUE,stringsAsFactors = TRUE)
df7<-read.table("PRSA_Data_Huairou_20130301-20170228.csv",sep = ",",header = TRUE,
                stringsAsFactors = TRUE)
df8<-read.table("PRSA_Data_Nongzhanguan_20130301-20170228.csv",sep=",",header = TRUE,stringsAsFactors = TRUE)
df9<-read.table("PRSA_Data_Shunyi_20130301-20170228.csv",sep = ",",header = TRUE,
                stringsAsFactors = TRUE)
df10<-read.table("PRSA_Data_Tiantan_20130301-20170228.csv",sep=",",header = TRUE,stringsAsFactors = TRUE)
df11<-read.table("PRSA_Data_Wanliu_20130301-20170228.csv",sep = ",",header = TRUE,
                 stringsAsFactors = TRUE)
df12<-read.table("PRSA_Data_Wanshouxigong_20130301-20170228.csv",sep=",",header = TRUE,stringsAsFactors = TRUE)
df<-bind_rows(df1,df2,df3,df4,df5,df6,df7,df8,df9,df10,df11,df12)


## On veut maintenant réorganiser les données en fonction du temps

# On crée une colonne datetime en combinant year, month, day, hour
df <- df %>%
  mutate(datetime = as.POSIXct(paste(year, month, day, hour, sep = "-"), format = "%Y-%m-%d-%H"))

# On trie d'abord par datetime, puis par station
df_sorted <- df %>%
  arrange(datetime, station)

print(df_sorted)
df_sorted$year <- as.factor(df_sorted$year)
df_sorted$wd <- as.factor(df_sorted$wd)
df_sorted$station <- as.factor(df_sorted$station)
summary(df_sorted)

##On veut maintenant s'interesser aux valeurs manquantes 

aggr(df_sorted)

#matrixplot(df_sorted, sortby = 2) Pour une visualisation plus fine

df_sorted <- df_sorted[,-19]

## On passe à l'imputation des données manquantes avec des forets aléatoires. 

df_imput <- missRanger(df_sorted, pmm.k = 5, num.trees = 100, seed = 1)
sum(is.na(df_imput))




#la partie suivante créer des colonnes dans le data frame original
#qui indiquent les IQA pour chaques polluants et un IQA pour chaque heure

#Indice de la Qualité de l'Air (IQA)

data <- df_imput

df_IQA <- read.table("AQI_INDEX.csv",sep=",",header=TRUE,stringsAsFactors = TRUE)
df_IQA <- df_IQA[1:35,1:5]
summary(df_IQA)


#IQA pour PM2.5
df_polluant <- df_IQA[df_IQA$polluant == "pm2_5",]#data_frame créé à partir de csv AQI_INDEX spécifique au seul polluant pm2.5
get_first_index <- function(pm_value) {
  index <- which(pm_value > c(df_polluant$concentration_inf))[1]
  return(index)
}
data$row_aqi<-sapply(data$PM2.5, get_first_index)#index du premier TRUE
data$index_high_pm25 <- sapply(data$row_aqi, function(index) {
  return(df_polluant$aqi_range_sup[index])
})
data$index_low_pm25 <- sapply(data$row_aqi, function(index) {
  return(df_polluant$aqi_range_inf[index])
})
data$C_high_pm25 <- sapply(data$row_aqi, function(index) {
  return(df_polluant$concentration_sup[index])
})
data$C_low_pm25 <- sapply(data$row_aqi, function(index) {
  return(df_polluant$concentration_inf[index])
})

#formule pour calculer IQA(PM2.5)
data$IQAPM2.5 <- (data$index_high_pm25 - data$index_low_pm25)/(data$C_high_pm25-data$C_low_pm25)*(data$PM2.5-data$C_low_pm25)+data$index_low_pm25

#IQA pour PM10
df_polluant <- df_IQA[df_IQA$polluant == "pm10",]
get_first_index <- function(pm_value) {
  index <- which(pm_value > c(df_polluant$concentration_inf))[1]
  return(index)
}
data$row_aqipm10<-sapply(data$PM10, get_first_index)#index du premier TRUE
data$index_high_pm10 <- sapply(data$row_aqipm10, function(index) {
  return(df_polluant$aqi_range_sup[index])
})
data$index_low_pm10 <- sapply(data$row_aqipm10, function(index) {
  return(df_polluant$aqi_range_inf[index])
})
data$C_high_pm10 <- sapply(data$row_aqipm10, function(index) {
  return(df_polluant$concentration_sup[index])
})
data$C_low_pm10 <- sapply(data$row_aqipm10, function(index) {
  return(df_polluant$concentration_inf[index])
})

data$IQAPM10 <- (data$index_high_pm10 - data$index_low_pm10)/(data$C_high_pm10-data$C_low_pm10)*(data$PM10-data$C_low_pm10)+data$index_low_pm10


#IQA pour SO2
df_polluant <- df_IQA[df_IQA$polluant == "so2",]
get_first_index <- function(pm_value) {
  index <- which(pm_value > c(df_polluant$concentration_inf))[1]
  return(index)
}
data$row_aqi<-sapply(data$SO2, get_first_index)#index du premier TRUE
data$index_high <- sapply(data$row_aqi, function(index) {
  return(df_polluant$aqi_range_sup[index])
})
data$index_low <- sapply(data$row_aqi, function(index) {
  return(df_polluant$aqi_range_inf[index])
})
data$C_high <- sapply(data$row_aqi, function(index) {
  return(df_polluant$concentration_sup[index])
})
data$C_low <- sapply(data$row_aqi, function(index) {
  return(df_polluant$concentration_inf[index])
})

data$IQA_SO2 <- (data$index_high - data$index_low)/(data$C_high-data$C_low)*(data$SO2-data$C_low)+data$index_low

#IQA pour CO
df_polluant <- df_IQA[df_IQA$polluant == "co",]
get_first_index <- function(pm_value) {
  index <- which(pm_value > c(df_polluant$concentration_inf))[1]
  return(index)
}
data$row_aqi<-sapply(data$CO/100, get_first_index)#index du premier TRUE
data$index_high <- sapply(data$row_aqi, function(index) {
  return(df_polluant$aqi_range_sup[index])
})
data$index_low <- sapply(data$row_aqi, function(index) {
  return(df_polluant$aqi_range_inf[index])
})
data$C_high <- sapply(data$row_aqi, function(index) {
  return(df_polluant$concentration_sup[index])
})
data$C_low <- sapply(data$row_aqi, function(index) {
  return(df_polluant$concentration_inf[index])
})

data$IQACO <- (data$index_high - data$index_low)/(data$C_high-data$C_low)*(data$CO/100-data$C_low)+data$index_low


#IQA pour NO2
df_polluant <- df_IQA[df_IQA$polluant == "no2",]
df_polluant[1,3]<-0#correction d'une mauvaise entrée de données
get_first_index <- function(pm_value) {
  index <- which(pm_value < c(df_polluant$concentration_sup))[1]
  return(index)
}
data$row_aqi<-sapply(data$NO2, get_first_index)#index du premier TRUE
data$index_high <- sapply(data$row_aqi, function(index) {
  return(df_polluant$aqi_range_sup[index])
})
data$index_low <- sapply(data$row_aqi, function(index) {
  return(df_polluant$aqi_range_inf[index])
})
data$C_high <- sapply(data$row_aqi, function(index) {
  return(df_polluant$concentration_sup[index])
})
data$C_low <- sapply(data$row_aqi, function(index) {
  return(df_polluant$concentration_inf[index])
})

data$IQANO2 <- (data$index_high - data$index_low)/(data$C_high-data$C_low)*(data$NO2-data$C_low)+data$index_low


#IQA pour 03
df_polluant <- df_IQA[df_IQA$polluant == "O3",]
get_first_index <- function(pm_value) {
  index <- which(pm_value > c(df_polluant$concentration_inf))[1]
  return(index)
}
data$row_aqi<-sapply(data$O3, get_first_index)#index du premier TRUE
data$index_high <- sapply(data$row_aqi, function(index) {
  return(df_polluant$aqi_range_sup[index])
})
data$index_low <- sapply(data$row_aqi, function(index) {
  return(df_polluant$aqi_range_inf[index])
})
data$C_high <- sapply(data$row_aqi, function(index) {
  return(df_polluant$concentration_sup[index])
})
data$C_low <- sapply(data$row_aqi, function(index) {
  return(df_polluant$concentration_inf[index])
})

data$IQAO3 <- (data$index_high - data$index_low)/(data$C_high-data$C_low)*(data$O3-data$C_low)+data$index_low


View(data)

save(data, file="data_.RData")


data$IQA <- sapply(1:nrow(data), function(i) {
  # création de la colonne IQA selon les critères de classfication
  if (!(data$IQAPM2.5[i] > 50 | data$IQAPM10[i] > 50 | data$IQA_SO2[i] > 50 | 
        data$IQACO[i] > 50 | data$IQANO2[i] > 50 | data$IQAO3[i] > 50)) {
    return("bon")
  } else if (!(data$IQAPM2.5[i] > 100 | data$IQAPM10[i] > 100 | data$IQA_SO2[i] > 100 | 
               data$IQACO[i] > 100 | data$IQANO2[i] > 100 | data$IQAO3[i] > 100)) {
    return("modéré")
  } else if (!(data$IQAPM2.5[i] > 150 | data$IQAPM10[i] > 150 | data$IQA_SO2[i] > 150 | 
               data$IQACO[i] > 150 | data$IQANO2[i] > 150 | data$IQAO3[i] > 150)) {
    return("non-sain pour sensibles")
  } else if (!(data$IQAPM2.5[i] > 200 | data$IQAPM10[i] > 200 | data$IQA_SO2[i] > 200 | 
               data$IQACO[i] > 200 | data$IQANO2[i] > 200 | data$IQAO3[i] > 200)) {
    return("non-sain")
  } else if (!(data$IQAPM2.5[i] > 300 | data$IQAPM10[i] > 300 | data$IQA_SO2[i] > 300 | 
               data$IQACO[i] > 300 | data$IQANO2[i] > 300 | data$IQAO3[i] > 300)) {
    return("très non-sain")
  } else {
    return("dangereux")
  }
})

data$IQA <- as.factor(data$IQA)

summary(data)

print(prop.table(table(data$IQA)) * 100)
#On remarque que le jeu est très déséquilibré

data<-data[,-c(19:23,25:29,31:34)]#efface les colonnes inutiles utilisées pour les calculs

## FIN DU TRAITEMENT PRELIMINAIRE 
## ENREGISTREMENT FINAL 

write.csv(data, file = "dfplusIQA_NAcomplet.csv", row.names = FALSE)



