require("missForest")
require(missMDA)
df_og <- read.table("/Users/achillegausseres/OneDrive/PRO/ACO/3A/Conf_Machine_Learning/Conf_Machine_Learning_repo/df.csv",
                    sep=",",
                    header=T,
                    stringsAsFactors = T)#jeu de données original
str(df_og)
df_imp<-missForest(df_og)
class(df_imp$ximp)#le data frame imputé


nb <- estim_ncpFAMD(mydata) ## tps de calcul long
res.imp <- imputeFAMD(mydata, ncp = nb$ncp)
res.famd <- FAMD(mydata, ,tab.disj = res.imp$tab.disj)