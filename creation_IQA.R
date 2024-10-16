#création de dfplusIQA.csv
load("dta_final.RData")


#réorganisation
data<-impl#data frame imputée par missRanger
sum(is.na(data))#aucune valeur manquante


#la partie suivante créer des colonnes dans le data frame original
#qui indiquent les IQA pour chaques polluants et un IQA pour chaque heure

#Indice de la Qualité de l'Air (IQA)
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

save(data, file="data_.RData")

df$IQA_PM2.5 <- data$IQAPM2.5
df$IQA_PM10 <- data$IQAPM10
df$IQA_SO2 <- data$IQA_SO2
df$IQA_CO <- data$IQACO
df$IQA_NO2 <- data$IQANO2
df$IQA_O3 <- data$IQAO3

data$IQA <- sapply(1:nrow(data), function(i) {
  # Check conditions
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
sum(is.na(data))
summary(data)
data<-data[,-c(19:24,26:30,32:35)]
write.csv(data, file = "dfplusIQA_NAcomplet.csv", row.names = FALSE)




