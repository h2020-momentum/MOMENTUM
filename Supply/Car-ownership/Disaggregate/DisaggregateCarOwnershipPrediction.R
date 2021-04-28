############################################################################################################
##SCRIPT DETAILS##
############################################################################################################
#Input: Synthetic population
#Output: No. of HH cars
#Type of model: Multinomial logit model
#The coefficients are read from the file 'coefficients.csv'. This file is provided to allow custom values for the coefficients
#NoCar is the base category and other choices are: OneCar, TwoCars and ThreeOrMoreCars
#Note 1: Age is an ordinal variable - <8 = 1, 18-24 = 2,...., 65-74 = 7, >74 = 8
#Note 2: Commute speed is in kmph

#############################################################################################################
##READING INPUT##
#############################################################################################################
#Change the separator (sep) and decimal (dec) format according to input data 
InputData = read.csv2("InputSampleFile.csv", sep = ',', dec = ".") #Socio-demographics data
CoefficientValues = read.csv2("CoefficientValues.csv", sep = ',', dec = ".", header = FALSE) #Coefficient values

#############################################################################################################
##DATA TRANSFORMATION##
#############################################################################################################
InputData$LowIncome = ifelse(InputData$HHIncome=="Low",1,0)
InputData$MediumIncome = ifelse(InputData$HHIncome=="Medium",1,0)
InputData$CargoBike = ifelse(InputData$HHCargoBike>=1,1,0)
InputData$PTPass = ifelse(InputData$HHPTPass>=1,1,0)
InputData$UnwillingToUseCSInFuture = ifelse(InputData$CarSharingWillingness=="0",1,0)
InputData$CSSubscription = ifelse(InputData$HHCarSharingSubscription==1,1,0)
InputData$CSSupplySubscriptionInteraction = I(InputData$CSSubscription * InputData$CarSharingSupply)

InputData$Age_Old = InputData$Age
InputData$Age = 1
InputData$Age[InputData$Age_Old>=18 & InputData$Age_Old<=24] = 2
InputData$Age[InputData$Age_Old>=25 & InputData$Age_Old<=34] = 3
InputData$Age[InputData$Age_Old>=35 & InputData$Age_Old<=44] = 4
InputData$Age[InputData$Age_Old>=45 & InputData$Age_Old<=54] = 5
InputData$Age[InputData$Age_Old>=55 & InputData$Age_Old<=64] = 6
InputData$Age[InputData$Age_Old>=658 & InputData$Age_Old<=74] = 7
InputData$Age[InputData$Age_Old>74] = 8

#############################################################################################################
##PREDICTION##
#############################################################################################################
SharesPrediction = function(InputData) {
  Utility = data.frame(matrix(NA, nrow = nrow(InputData), ncol = 3))
  colnames(Utility) = c("OneCar","TwoCars","ThreeOrMoreCars")
  Utility$OneCar = CoefficientValues[1,2] +
    CoefficientValues[4,2]*InputData$Citizen +
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
    CoefficientValues[5,2]*InputData$Citizen +
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
    CoefficientValues[6,2]*InputData$Citizen +
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
##AGGREGATION TO HOUSEHOLD##
#############################################################################################################
#Based on household leader (usually the one with highest age)
CarOwnershipBasedOnIndividualWithHighestAge = aggregate(InputDataWithCarOwnership[, 26], list(InputDataWithCarOwnership$HH_ID), max)
colnames(CarOwnershipBasedOnIndividualWithHighestAge)= c("HH_ID", "CarOwnership")

#Based on mean of probabilities, calculated across the individuals from the households
CarOwnershipBasedOnMeanOfProbabilities = aggregate(InputDataWithCarOwnership[, 22:25], list(InputDataWithCarOwnership$HH_ID), mean)
names(CarOwnershipBasedOnMeanOfProbabilities)[names(CarOwnershipBasedOnMeanOfProbabilities) == 'Group.1'] <- 'HH_ID'
CarOwnershipBasedOnMeanOfProbabilities$CarOwnership = max.col(CarOwnershipBasedOnMeanOfProbabilities[,2:5], "first") - 1

#Based on mean of assigned choice, calculated across the individuals from the households
CarOwnershipBasedOnMeanOfAssignedChoice = aggregate(InputDataWithCarOwnership[, 26], list(InputDataWithCarOwnership$HH_ID), mean)
colnames(CarOwnershipBasedOnMeanOfAssignedChoice)= c("HH_ID", "CarOwnership")
CarOwnershipBasedOnMeanOfAssignedChoice$CarOwnership = round(CarOwnershipBasedOnMeanOfAssignedChoice$CarOwnership)

#############################################################################################################
##SAVING OUTPUT##
#############################################################################################################
#Change the separator (sep) and decimal (dec) format according to your needs
#Original car ownership calculation
write.table(InputDataWithCarOwnership, file = "SyntheticPopulationWithCarOwnership.csv", row.names=FALSE, sep=",", dec = ".")

#Car ownership based on household leader (usually the one with highest age)
write.table(CarOwnershipBasedOnIndividualWithHighestAge, file = "CarOwnershipBasedOnIndividualWithHighestAge.csv", row.names=FALSE, sep=",", dec = ".")

#Car ownership based on mean of probabilities, calculated across the individuals from the households
write.table(CarOwnershipBasedOnMeanOfProbabilities, file = "CarOwnershipBasedOnMeanOfProbabilities.csv", row.names=FALSE, sep=",", dec = ".")

#Car ownership based on mean of assigned choice, calculated across the individuals from the households
write.table(CarOwnershipBasedOnMeanOfAssignedChoice, file = "CarOwnershipBasedOnMeanOfAssignedChoice.csv", row.names=FALSE, sep=",", dec = ".")

