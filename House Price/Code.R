library(tidyverse)
library(knitr)
library(dplyr)
library(tidyr)
library(magrittr)
library(scales)
library(ggplot2)
library(purrr)
library(GGally)
library(gtools)
library(faraway)
library(ellipse)
library(pander)

library(ISLR, quietly = TRUE)
library(glmnet, quietly = TRUE)

DF <- read.csv("~/housing.txt")
summary(DF)
str(DF)
#View(DF)
na_pct = colMeans(is.na(DF))
#View(data.frame(t(na_pct)))

DF1 <- DF[,-1] #remove 'id'
for (i in 1:ncol(DF1)) {
  if(class(DF1[,i]) == 'factor' & sum(is.na(DF1[,i])) > 0){
    print(i)
    # Change NA to str::NA
    DF1[,i] <- addNA(DF1[,i])
  }
}
sum(is.na(DF1))

for (i in 1:ncol(DF1)) {
  if(class(DF1[,i]) != 'factor' & sum(is.na(DF1[,i])) > 0){
    print(i)
  }
}

#names(DF1)[c(4,27,60)]
#"LotFrontage" "MasVnrArea"  "GarageYrBlt"
# put explanation later on
DF1$LotFrontage[is.na(DF$LotFrontage)] <- mean(DF1$LotFrontage, na.rm=T)
# change 8 missing MarVnrArea to zero
DF1$MasVnrArea[is.na(DF$MasVnrArea)] <- 0
# and its corresponding missing type to 'None'
DF1$MasVnrType[is.na(DF$MasVnrType)] <- 'None'
# sum(is.na(DF$GarageType)) = 81 so all flat with a missing garage will have no garageyrblt
DF1$GarageYrBlt[is.na(DF$GarageType)] <- min(DF$GarageYrBlt, na.rm = T)

#standardize
DF2 <- DF1
for (i in 1:ncol(DF2)) {
  if(class(DF2[,i]) != 'factor'){
    DF2[,i] <- scale(DF1[,i])
  }
}
#taking log for sale price
DF2$SalePrice = log(DF1$SalePrice)

DF1[,-80] <- DF2[,-80]


x <- model.matrix(SalePrice ~., data = DF1)
y <- DF1$SalePrice

# df <- read.csv("~/Desktop/College.csv")
# X <- df[, c("Accept", "Enroll", "Outstate", "Books", "Grad.Rate")]
# y <- df["Room.Board"]

#######ridge#######

#First, set a grid of lambda to search over. We want to include
#lambda = 0 for standard linear regression
grid.lambda <- 10^seq(5, -2, length = 100)





set.seed(1) #for reproducability
#Randomly select a training and test set.
#Here, we leave half of the data out for later model assessment
train <- sample(1:nrow(x), nrow(x)*.8)
test <- (-train)
y.train <- y[train]
y.test <- y[test]


alpha = seq(0, 1, length = 101)
best.lambda = rep(NA,100)
mspe = rep(NA,100)
#Now, fit a Lasso regression model to the training data
for (i in seq(100)){
  model.train <- glmnet(x[train, ], y.train, alpha = alpha[i], lambda = grid.lambda)
  set.seed(1) #for reproducability
  cv.out <- cv.glmnet(x[train, ], y.train, alpha = alpha[i])
  best.lambda[i] <- cv.out$lambda.min
  model.pred <- predict(model.train, s = best.lambda[i], newx = x[test,])
  mspe[i] <- mean((model.pred - y.test)^2)
  }





#######forward/backward#######################

null=lm(log(SalePrice)~1, data=DF1)
full=lm(log(SalePrice)~., data=DF1)
step(null, scope = list(upper=fit0), data=DF1, direction="both")
fit1 <- lm(formula = log(SalePrice) ~ OverallQual + GrLivArea + Neighborhood +
             BsmtQual + RoofMatl + BsmtFinSF1 + MSSubClass + BsmtExposure +
             KitchenQual + Condition2 + SaleCondition + OverallCond +
             YearBuilt + LotArea + PoolQC + ExterQual + GarageArea + TotalBsmtSF +
             BldgType + Functional + BedroomAbvGr + Condition1 + PoolArea +
             ScreenPorch + LowQualFinSF + LandContour + Street + LandSlope +
             KitchenAbvGr + MasVnrArea + Exterior1st + TotRmsAbvGrd +
             LotConfig + MSZoning + GarageCars + Fireplaces + YearRemodAdd +
             GarageQual + GarageCond + WoodDeckSF + BsmtFullBath + X1stFlrSF +
             Fence + MoSold, data = DF2[-A,])
summary(fit1)
e <- fit1$residuals
std.residuals <- (e-mean(e))/sd(e)
qqnorm(std.residuals)
qqplot(rnorm(1000),std.residuals)
ks.test(std.residuals, rnorm(1000))


step(full, data=DF1, direction="backward")
fit2 <- lm(formula = log(SalePrice) ~ MSZoning + LotFrontage + LotArea + 
             Street + LotConfig + LandSlope + Neighborhood + Condition1 + 
             Condition2 + BldgType + OverallQual + OverallCond + YearBuilt + 
             YearRemodAdd + RoofMatl + Exterior1st + ExterCond + Foundation + 
             BsmtQual + BsmtExposure + BsmtFinSF1 + BsmtFinSF2 + BsmtUnfSF + 
             Heating + HeatingQC + CentralAir + X1stFlrSF + X2ndFlrSF + 
             LowQualFinSF + BsmtFullBath + FullBath + HalfBath + KitchenAbvGr + 
             KitchenQual + TotRmsAbvGrd + Functional + Fireplaces + GarageYrBlt + 
             GarageCars + GarageArea + GarageQual + GarageCond + WoodDeckSF + 
             EnclosedPorch + X3SsnPorch + ScreenPorch + PoolArea + PoolQC + 
             SaleType + SaleCondition, data = DF2[-A,])
summary(fit2)
e <- fit2$residuals
std.residuals <- (e-mean(e))/sd(e)
qqnorm(std.residuals)
qqplot(rnorm(1000),std.residuals)
ks.test(std.residuals, rnorm(1000))


fit3 <- lm(formula = log(SalePrice) ~ GrLivArea + YearBuilt + KitchenQual + 
             OverallCond + FireplaceQu + GarageArea + MSZoning + BsmtFullBath + 
             BsmtFinType2 + PoolQC + PoolArea + ExterQual + Functional + 
             ScreenPorch + HalfBath + KitchenAbvGr + YearRemodAdd + Heating + 
             EnclosedPorch + BedroomAbvGr + GarageQual + BsmtHalfBath, 
             data = DF2[-A,]) 
summary(fit3)
e <- fit3$residuals
std.residuals <- (e-mean(e))/sd(e)
qqnorm(std.residuals)
qqplot(rnorm(1000),std.residuals)
ks.test(std.residuals, rnorm(1000))
##############
inflm.SR <- influence.measures(fit0)
inflp <- which(apply(inflm.SR$is.inf, 1, any))
noninfluDF <- DF1[-inflp,]
fit_noninf <- lm(SalePrice~., data = noninfluDF)

# qqplot
qqnorm(summary(fit1)$residuals)
qqplot(rnorm(1460),summary(fit1)$residuals)
# standardized residual 
stResidual <- scale(summary(fit1)$residuals)
qqnorm(stResidual)
shapiro.test(stResidual)

## we can see that it is not normal
##########part II q2########
Morty <- data.frame(DF[6,])
