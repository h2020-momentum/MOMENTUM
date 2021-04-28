############################################################################################################
##SCRIPT DETAILS##
############################################################################################################
#Input: Aggregate socio-demographics
#Output: Numbers of cars in thousands
#Type of model: Linear regression
#The coefficients are read from the file 'coefficients.csv'. This file is provided to allow custom values for the coefficient

#############################################################################################################
##READING INPUT##
#############################################################################################################
#Change the separator (sep) and decimal (dec) format according to input data 
InputData <- read.csv2("InputSampleFile.csv", sep = ',', dec = ".") 
CoefficientValues <- read.csv2("CoefficientValues.csv", sep = ',', dec = ".") #Coefficient values

#Data conversion
InputData$total_population_thousands <- InputData$total_population/1000
InputData$population_from_65_years_age_percent <-	InputData$population_from_65_years_age*100/InputData$total_population
InputData$household_size_from_3_percent <- InputData$household_size_from_3*100/InputData$total_households
InputData$medium_density_districts <- (InputData$population_density	> 20 & InputData$population_density	<= 40)

#############################################################################################################
##PREDICTION##
#############################################################################################################
CarOwnershipPrediction <- function(InputData, CoefficientValues) {
  InputData$CarOwnership = CoefficientValues[,1] + 
    CoefficientValues[,2] * InputData$total_population_thousands +
    CoefficientValues[,3] * InputData$population_from_65_years_age_percent +
    CoefficientValues[,4] * InputData$household_size_from_3_percent +
    CoefficientValues[,5] * InputData$medium_density_districts +
    CoefficientValues[,6] * InputData$car_sharing_vehicles 
  return(InputData)
}
InputDataWithCarOwnership = CarOwnershipPrediction(InputData, CoefficientValues)

#############################################################################################################
##SAVING OUTPUT##
#############################################################################################################
#Change the separator (sep) and decimal (dec) format according to your needs
write.table(InputDataWithCarOwnership, file = "SociodemographicsWithCarOwnership.csv", row.names=FALSE, sep=",", dec = ".")

