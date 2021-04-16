############################################################################################################
##SCRIPT DETAILS##
############################################################################################################
#Input: Synthetic population
#Output: Mode choice
#Type of model: Multinomial logit model
#The coefficients are read from the file 'coefficients.csv'. This file is provided to allow custom values for the coefficients
#Conventional system-as-a-whole(0) is the base category and other choices are: bike-sharing(1), car-sharing(2) and ride-sharing(3)

#############################################################################################################
##READING INPUT##
#############################################################################################################
#Change the separator (sep) and decimal (dec) format according to input data 
InputData <- read.csv2("SyntheticPopulation.csv", sep = ',', dec = ".") #Socio-demographics data
CoefficientValues <- read.csv2("CoefficientValues.csv", sep = ',', dec = ".", header = FALSE) #Coefficient values

#############################################################################################################
##PREDICTION##
#############################################################################################################
SharesPrediction <- function(InputData) {
  Utility <- data.frame(matrix(NA, nrow = nrow(InputData), ncol = 3))
  colnames(Utility) <- c("Bikesharing","Carsharing","Ridesharing")
  Utility$Bikesharing <- CoefficientValues[1,2] +
    CoefficientValues[4,2]*InputData$Age20To44 +
    CoefficientValues[7,2]*InputData$Male +
    CoefficientValues[11,2]*InputData$UniversityOrVocationalDegree +
    CoefficientValues[14,2]*InputData$PTPass +
    CoefficientValues[17,2]*InputData$HHCars +
    CoefficientValues[19,2]*InputData$TripDistKM2 +
    CoefficientValues[20,2]*InputData$TripDistKM2To5 +
    CoefficientValues[29,2]*InputData$TravelTimeMins30 +
    CoefficientValues[30,2]*InputData$BikesharingSupply +
    CoefficientValues[31,2]*InputData$VehicleNonAvailabilityBikesharing
    
  
  Utility$Carsharing <- CoefficientValues[2,2] +
    CoefficientValues[5,2]*InputData$Age20To44 +
    CoefficientValues[8,2]*InputData$Male +
    CoefficientValues[9,2]*InputData$NoBLicense +
    CoefficientValues[12,2]*InputData$UniversityOrVocationalDegree +
    CoefficientValues[15,2]*InputData$PTPass +
    CoefficientValues[18,2]*InputData$HHCars +
    CoefficientValues[21,2]*InputData$TripDistKM2To5 +
    CoefficientValues[23,2]*InputData$TripDistKM5To15 +
    CoefficientValues[25,2]*InputData$TravelTimeMins15 +
    CoefficientValues[27,2]*InputData$TravelTimeMins15To30 +
    CoefficientValues[32,2]*InputData$VehicleNonAvailabilityCarsharing
  
  Utility$Ridesharing <- CoefficientValues[3,2] +
    CoefficientValues[6,2]*InputData$Age20To44 +
    CoefficientValues[10,2]*InputData$hasLicense +
    CoefficientValues[13,2]*InputData$UniversityOrVocationalDegree +
    CoefficientValues[16,2]*InputData$PTPass +
    CoefficientValues[22,2]*InputData$TripDistKM2To5 +
    CoefficientValues[24,2]*InputData$TripDistKM5To15 +
    CoefficientValues[26,2]*InputData$TravelTimeMins15 +
    CoefficientValues[28,2]*InputData$TravelTimeMins15To30 +
    CoefficientValues[33,2]*InputData$VehicleNonAvailabilityRidesharing
  
  #Prediction
  Utility$expBikesharing <- exp(Utility$Bikesharing)
  Utility$expCarsharing <- exp(Utility$Carsharing)
  Utility$expRidesharing <- exp(Utility$Ridesharing)
  Utility$TotalUtility <- rowSums(Utility[,4:6])
  Utility$TotalUtility <- Utility$TotalUtility + exp(0)
  
  Utility$probBikesharing <- Utility$expBikesharing/Utility$TotalUtility
  Utility$probCarsharing <- Utility$expCarsharing/Utility$TotalUtility
  Utility$probRidesharing <- Utility$expRidesharing/Utility$TotalUtility
  Utility$probConventionalSystems <- exp(0)/Utility$TotalUtility
  Utility$totalProb = rowSums(Utility[,8:11])
  
  shares <- colMeans(Utility[,8:11]) #, na.rm = TRUE
  return(Utility[,c(11, 8:10)])
}

PredictedShares <- SharesPrediction(InputData)
InputDataWithModeChoice <- InputData
InputDataWithModeChoice[(ncol(InputDataWithModeChoice)+1):(ncol(InputDataWithModeChoice)+4)] <- PredictedShares
InputDataWithModeChoice$ModeChoice = max.col(PredictedShares, "first") - 1

#############################################################################################################
##SAVING OUTPUT##
#############################################################################################################
#Change the separator (sep) and decimal (dec) format according to your needs
write.table(InputDataWithModeChoice, file = "SyntheticPopulationWithModeChoice.csv", row.names=FALSE, sep=",", dec = ".")

