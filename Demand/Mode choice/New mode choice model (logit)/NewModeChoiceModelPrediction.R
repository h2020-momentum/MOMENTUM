############################################################################################################
##SCRIPT DETAILS##
############################################################################################################
#Input: Synthetic population, travel times, sharing vehicle availability, bike-sharing supply (in hundreds)
#Output: Mode choice
#Type of model: Multinomial logit model
#The coefficients are read from the file 'coefficients.csv'. This file is provided to allow custom values for the coefficients
#Conventional system-as-a-whole(0) is the base category and other choices are: bike-sharing(1), car-sharing(2) and ride-sharing(3)

#############################################################################################################
##READING INPUT##
#############################################################################################################
#Change the separator (sep) and decimal (dec) format according to input data 
InputData <- read.csv2("InputSampleFile.csv", sep = ',', dec = ".") 
CoefficientValues <- read.csv2("CoefficientValues.csv", sep = ',', dec = ".", header = FALSE) #Coefficient values

#############################################################################################################
##DATA TRANSFORMATION##
#############################################################################################################
InputData$Age20To44 = ifelse(InputData$Age>=20 & InputData$Age<=44,1,0)
InputData$Male = ifelse(InputData$Gender=="Male",1,0)
InputData$UniversityOrVocationalDegree = ifelse(InputData$Education == "University" | InputData$Education == "Vocational",1,0)

InputData$TripDistKM2 = ifelse(InputData$TripDistanceKM <= 2,1,0) 
InputData$TripDistKM2To5 = ifelse(InputData$TripDistanceKM > 2 & InputData$TripDistanceKM <= 5,1,0)
InputData$TripDistKM5To15 = ifelse(InputData$TripDistanceKM > 5 & InputData$TripDistanceKM <= 15,1,0)

InputData$TravelTimeMins30_Bikesharing = ifelse(InputData$TravelTimeMins_Bikesharing <= 30,1,0)
InputData$TravelTimeMins15_Carsharing = ifelse(InputData$TravelTimeMins_Carsharing <= 15,1,0)
InputData$TravelTimeMins15To30_Carsharing = ifelse(InputData$TravelTimeMins_Carsharing > 15 & InputData$TravelTimeMins_Carsharing <= 30,1,0)
InputData$TravelTimeMins15_Ridesharing = ifelse(InputData$TravelTimeMins_Ridesharing <= 15,1,0)
InputData$TravelTimeMins15To30_Ridesharing = ifelse(InputData$TravelTimeMins_Ridesharing > 15 & InputData$TravelTimeMins_Ridesharing <= 30,1,0)

InputData$NoBLicense = 1
InputData$NoBLicense[InputData$LicenseType=="B" | InputData$LicenseType=="A,B"] = 0
InputData$BikesharingSupply = InputData$BikesharingSupply/100

InputData$VehicleNonAvailability_Bikesharing = ifelse(InputData$VehicleAvailability_Bikesharing==0,1,0)
InputData$VehicleNonAvailability_Carsharing = ifelse(InputData$VehicleAvailability_Carsharing==0,1,0)
InputData$VehicleNonAvailability_Ridesharing = ifelse(InputData$VehicleAvailability_Ridesharing==0,1,0)
  
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
    CoefficientValues[29,2]*InputData$TravelTimeMins30_Bikesharing +
    CoefficientValues[30,2]*InputData$BikesharingSupply +
    CoefficientValues[31,2]*InputData$VehicleNonAvailability_Bikesharing
    
  
  Utility$Carsharing <- CoefficientValues[2,2] +
    CoefficientValues[5,2]*InputData$Age20To44 +
    CoefficientValues[8,2]*InputData$Male +
    CoefficientValues[9,2]*InputData$NoBLicense +
    CoefficientValues[12,2]*InputData$UniversityOrVocationalDegree +
    CoefficientValues[15,2]*InputData$PTPass +
    CoefficientValues[18,2]*InputData$HHCars +
    CoefficientValues[21,2]*InputData$TripDistKM2To5 +
    CoefficientValues[23,2]*InputData$TripDistKM5To15 +
    CoefficientValues[25,2]*InputData$TravelTimeMins15_Carsharing +
    CoefficientValues[27,2]*InputData$TravelTimeMins15To30_Carsharing +
    CoefficientValues[32,2]*InputData$VehicleNonAvailability_Carsharing
  
  Utility$Ridesharing <- CoefficientValues[3,2] +
    CoefficientValues[6,2]*InputData$Age20To44 +
    CoefficientValues[10,2]*InputData$hasLicense +
    CoefficientValues[13,2]*InputData$UniversityOrVocationalDegree +
    CoefficientValues[16,2]*InputData$PTPass +
    CoefficientValues[22,2]*InputData$TripDistKM2To5 +
    CoefficientValues[24,2]*InputData$TripDistKM5To15 +
    CoefficientValues[26,2]*InputData$TravelTimeMins15_Ridesharing +
    CoefficientValues[28,2]*InputData$TravelTimeMins15To30_Ridesharing +
    CoefficientValues[33,2]*InputData$VehicleNonAvailability_Ridesharing
  
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

