# Cargo librerias
library(dplyr)
library(modeest)
library(rpart)
library(rpart.plot)
library(ggplot2)
library(ISLR)
library(randomForest)

# Ejercicio 2 ####
 
# Creo dataset con los id, y los valores de respuesta y clasificiacion, ademas los agrupo segun la visualizacion
data <- tibble(
id = as.character(1:20),
 yr = c(5, 7, 33, 22, 6, 12, 16, 2, 17, 9, 1, 3, 5, 7, 5, 3, 16, 12, 9, 1),
 yc = c("A", "A", "C", "B", "B", "B", "C", "B", "A", "C", "C", "C", "B", "B", "A","A", "A", "B", "C", "C")
) %>% 
  mutate(grupo = case_when(id %in% c(3, 6, 14, 18) ~ "Grupo 1",
                           id %in% c(2, 10, 16)    ~ "Grupo 2",
                           id %in% c(9, 11, 19)    ~ "Grupo 3",
                           id %in% c(12, 17, 20)   ~ "Grupo 4",
                           id %in% c(1, 7, 8, 13)  ~ "Grupo 5",
                           id %in% c(4, 5, 15)     ~ "Grupo 6"))

# 1) Valores promedio de respuesta y moda de clasificacion de cada grupo
data %>% 
  group_by(grupo) %>% 
  summarise(valor  = mean(yr),
            clasif = mfv(yc))

# 2) X1=10, X2=13 ---> la observacion pertenece al grupo 1, YR=16

# 3) X1=4, X2=8   ---> la observacion pertenece al grupo 2, YC=A


# Ejercicio 3 ####
datos = 1:10
una.muestra.bootstrap <- sample(1:10, 10, replace = TRUE)
# 1) La probabilidad de que coincidan la primera observacion de la muestra y del bootrsap es 1/n,
# ya que se trata de un muestreo aleatorio. 

# 2) Como la muestra es con reposicion la probabilidad de que una observacion j no este en la muestra es igual 
# a la probabilidad de elegir cualquier otra observacion = (n-1)/n, repetido cada vez que se selecciona una observacion,
# es decir n veces: [(n-1)/n]^n=(1-1/n)^n.

# 3) Genere un grafico que muestre para cada valor de n de 1 a 100, la probabilidad de que la
# observacion j-esima este en la muestra bootstrap. Comente lo que se observa.

otra.muestra.bootstrap <- tibble(
  n   = 1:100,
  prob_no = ((n-1)/n)^n,
  prob_si = 1-prob_no
)


ggplot(data=otra.muestra.bootstrap, aes(x=n, y=prob_si))+
  geom_point(alpha = 0.5)+
  scale_y_continuous(limits = c(0,1))+
  theme_bw()+
  ggtitle('Probabilidad de aparecer en el remuestreo')
  
# Ejercicio 4 ####
data_cs <- ISLR::Carseats

# Subseteo primeras 300 observaciones para train y ultimas 100 para test
set.seed(2021)
intrain <- sample(x=1:400, size=300)

# Creo dataset train y test
train <- data_cs[intrain,]
test  <- data_cs[-intrain,]  

# Estimo modelo lineal con todas las variables
model  <- lm(data=train, Sales ~ CompPrice + Income + Advertising + Population + Price + ShelveLoc + Age + Education + Urban + US)
summary(model)

# Arbol de regresion con todas las variables
sales_tree <- rpart(Sales ~ CompPrice + Income + Advertising + Population + Price + ShelveLoc + Age + Education + Urban + US, 
                    data=train)
rpart.plot(sales_tree, digits=3, cex=.6)

# Calculo predicciones
pred_train <- predict(sales_tree, newdata=train)
pred_test  <- predict(sales_tree, newdata=test)

# Estimo ECM
ecm_train <- sum((pred_train-train$Sales)^2)/nrow(train)
ecm_test  <- sum((pred_test-test$Sales)^2)/nrow(test)

# Arbol maximal
sales_tree <- rpart(Sales ~ CompPrice + Income + Advertising + Population + Price + ShelveLoc + Age + Education + Urban + US,
                    data=train, control=rpart.control(cp = 0))

plotcp(sales_tree)
# El mayor cp=0.02 con error a menos de un desvío del menor error

# Podo el arbol
sales_tree_prune <- prune(sales_tree, cp = 0.02)
rpart.plot(sales_tree_prune, digits=3)

# vuelvo a calcular valores predichos y ecm
pred_train_prune <- predict(sales_tree_prune, newdata=train)
pred_test_prune  <- predict(sales_tree_prune, newdata=test)

ecm_train_prune <- sum((pred_train_prune-train$Sales)^2)/nrow(train)
ecm_test_prune  <- sum((pred_test_prune-test$Sales)^2)/nrow(test)


# Validación cruzada
# Voy a armar 5 modelos para cada mtry entre 1 y 11 y promediar el ecm, k=5
# k grupos con k < n: estimo con n-n/k y se predice con n/k, repito el procedimiento k veces.

ecm_rf <- tibble()
k <- 5

for (i in 1:ncol(train)) {
  
  ecm_aux <- tibble()
  
  for (j in 1:k) {
    
    sample   <- sample(x=1:400, size=400-(400/k))
    
    train_rf <- data_cs[sample,] 
    test_rf  <- data_cs[-sample,] 
    
    rf_modelo <- randomForest(Sales ~ CompPrice + Income + Advertising + Population + Price + ShelveLoc + Age + Education + Urban + US,
                              data = train_rf, 
                              mtry = i)
    
    pred_train_rf <- predict(rf_modelo, newdata=train_rf)
    pred_test_rf  <- predict(rf_modelo, newdata=test_rf)
    
    ecm_train_rf <- mean((pred_train_rf-train$Sales)^2)
    ecm_test_rf  <- mean((pred_test_rf-test$Sales)^2)
    
    aux <- tibble(
      mtry = i,
      rep  = j,
      ecm_train = ecm_train_rf,
      ecm_test  = ecm_test_rf
    )
    
    ecm_aux <- bind_rows(ecm_aux, aux)
  }
  
  ecm_rf <- bind_rows(ecm_rf, ecm_aux)
  
}


#Estimaré la validación cruzada para un *k=5*. El siguiente código intenta estimar para todas las posibles cantidad de variables (*11 variables*), *k* submuestras aleatorias de tamaño $n-(n/k)$. Para cada una de estas submuestras calcula además el error del modelo en el cojunto de entrenamiento y en el de control. Luego se muestra el promedio de estas *k* submuestras:
#```{r}
# ecm_rf %>% 
#   group_by(mtry) %>% 
#   summarise(ecm_train = mean(ecm_train), 
#             ecm_test  = mean(ecm_test))
#```

# Promedio del error de pronostico para cada grupo, la submuestra que tiene menor ECM es la de 2 variables
ecm_rf %>% 
  group_by(mtry) %>% 
  summarise(ecm_train = mean(ecm_train), 
            ecm_test  = mean(ecm_test))

# usando el comando tuneRf
tune <- tuneRF(x=data_cs[,2:11], y=data_cs[,1])

rf_modelo <- randomForest(Sales ~ CompPrice + Income + Advertising + Population + Price + ShelveLoc + Age + Education + Urban + US,
                          data = train, 
                          mtry = 2)
rf_modelo$forest

