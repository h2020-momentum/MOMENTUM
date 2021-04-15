############################################################################################################
##SCRIPT DETAILS##
############################################################################################################
#Input: Synthetic population
#Output: No. of HH cars
#Type of model: Multinomial logit model
#The coefficients are read from the file 'coefficients.csv'. This file is provided to allow custom values for the coefficients
#NoCar is the base category and other choices are: OneCar, TwoCars and ThreeOrMoreCars

#############################################################################################################
##READING INPUT##
#############################################################################################################
#Change the separator (sep) and decimal (dec) format according to input data 
InputData = read.csv2("SyntheticPopulation.csv", sep = ',', dec = ".") #Socio-demographics data
CoefficientValues = read.csv2("CoefficientValues.csv", sep = ',', dec = ".", header = FALSE) #Coefficient values

#############################################################################################################
##PREDICTION##
#############################################################################################################
SharesPrediction = function(InputData) {
  Utility = data.frame(matrix(NA, nrow = nrow(InputData), ncol = 3))
  colnames(Utility) = c("OneCar","TwoCars","ThreeOrMoreCars")
  Utility$OneCar = CoefficientValues[1,2] +
    CoefficientValues[4,2]*InputData$NativeCitizen +
    CoefficientValues[7,2]*InputData$LowIncome +
    CoefficientValues[10,2]*InputData$MediumIncome +
    CoefficientValues[13,2]*I(InputData$Age^3) +
    CoefficientValues[16,2]*InputData$Age +
    CoefficientValues[19,2]*InputData$HHSize +
    CoefficientValues[22,2]*InputData$CargoBike +
    CoefficientValues[25,2]*InputData$PTPass +
    CoefficientValues[28,2]*InputData$UnwillingToUseCSInFuture +
    CoefficientValues[31,2]*InputData$CSSupplySubscriptionInteraction +
    CoefficientValues[34,2]*InputData$CSSupply +
    CoefficientValues[37,2]*InputData$CommuteSpeed
  
  
  Utility$TwoCars = CoefficientValues[2,2] +
    CoefficientValues[5,2]*InputData$NativeCitizen +
    CoefficientValues[8,2]*InputData$LowIncome +
    CoefficientValues[11,2]*InputData$MediumIncome +
    CoefficientValues[14,2]*I(InputData$Age^3) +
    CoefficientValues[17,2]*InputData$Age +
    CoefficientValues[20,2]*InputData$HHSize +
    CoefficientValues[23,2]*InputData$CargoBike +
    CoefficientValues[26,2]*InputData$PTPass +
    CoefficientValues[29,2]*InputData$UnwillingToUseCSInFuture +
    CoefficientValues[32,2]*InputData$CSSupplySubscriptionInteraction +
    CoefficientValues[35,2]*InputData$CSSupply +
    CoefficientValues[38,2]*InputData$CommuteSpeed
  
  
  Utility$ThreeOrMoreCars = CoefficientValues[3,2] +
    CoefficientValues[6,2]*InputData$NativeCitizen +
    CoefficientValues[9,2]*InputData$LowIncome +
    CoefficientValues[12,2]*InputData$MediumIncome +
    CoefficientValues[15,2]*I(InputData$Age^3) +
    CoefficientValues[18,2]*InputData$Age +
    CoefficientValues[21,2]*InputData$HHSize +
    CoefficientValues[24,2]*InputData$CargoBike +
    CoefficientValues[27,2]*InputData$PTPass +
    CoefficientValues[30,2]*InputData$UnwillingToUseCSInFuture +
    CoefficientValues[33,2]*InputData$CSSupplySubscriptionInteraction +
    CoefficientValues[36,2]*InputData$CSSupply +
    CoefficientValues[39,2]*InputData$CommuteSpeed
  
  #Prediction
  Utility$expOneCar = exp(Utility$OneCar)
  Utility$expTwoCars = exp(Utility$TwoCars)
  Utility$expThreeOrMoreCars = exp(Utility$ThreeOrMoreCars)
  Utility$TotalUtility = rowSums(Utility[,4:6])
  Utility$TotalUtility = Utility$TotalUtility + exp(0)
  
  Utility$probOneCar = Utility$expOneCar/Utility$TotalUtility
  Utility$probTwoCars = Utility$expTwoCars/Utility$TotalUtility
  Utility$probThreeOrMoreCars = Utility$expThreeOrMoreCars/Utility$TotalUtility
  Utility$probNoCar = exp(0)/Utility$TotalUtility
  Utility$totalProb = rowSums(Utility[,8:11])
  
  shares = colMeans(Utility[,8:11]) #, na.rm = TRUE
  return(Utility[,c(11, 8:10)])
}

PredictedShares = SharesPrediction(InputData)
InputDataWithCarOwnership = InputData
InputDataWithCarOwnership[(ncol(InputDataWithCarOwnership)+1):(ncol(InputDataWithCarOwnership)+4)] = PredictedShares
InputDataWithCarOwnership$CarOwnership = max.col(PredictedShares, "first") - 1

#############################################################################################################
##SAVING OUTPUT##
#############################################################################################################
#Change the separator (sep) and decimal (dec) format according to your needs
write.table(InputDataWithCarOwnership, file = "SyntheticPopulationWithCarOwnership.csv", row.names=FALSE, sep=",", dec = ".")

