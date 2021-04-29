############################################################################################################
##Load required libraries##
############################################################################################################
library(dplyr)
library(DirichletReg)

############################################################################################################
##Total Demand Per Day##
############################################################################################################
##Script details##
#Input: Number of stations and analysis period (day & month)
#Output: Total number of demand per day
#Type of model: Regression model
#The coefficients are read from the file 'CoefficientValues_TotalDemand.csv'. This file is provided to allow custom values for the coefficients

##READING INPUT##
#Change the separator (sep) and decimal (dec) format according to input data 
InputData_Demand = read.csv2("InputData_Demand.csv", sep = ',', dec = ".") 
CoefficientValues_TotalDemand = read.csv2("CoefficientValues_TotalDemand.csv", sep = ',', dec = ".", header = FALSE) #Coefficient values

##PREDICTION##
TotalDemandPrediction <- function(InputData, CoefficientValues) {
  TotalDemandPredicted = CoefficientValues[1,2] + 
    CoefficientValues[2,2] * InputData[1,1] +
    CoefficientValues[3,2] * InputData[1,2] +
    CoefficientValues[4,2] * InputData[1,3] +
    CoefficientValues[5,2] * InputData[1,4] +
    CoefficientValues[6,2] * InputData[1,5] + 
    CoefficientValues[7,2] * InputData[1,6] +
    CoefficientValues[8,2] * InputData[1,7]
  return(ceiling(TotalDemandPredicted))
}
InputData_Demand$TotalDemandPredicted = TotalDemandPrediction(InputData_Demand, CoefficientValues_TotalDemand)

############################################################################################################
##Total Demand Per Day Per Station##
############################################################################################################
##Script details##
#Input: Total demand and analysis period (day)
#Output: Total number of demand per day per station
#Type of model: Dirichlet regression
#This model is specific to the city of Regensburg, since it is built specific to the car-sharing stations in Regensburg
#Due to the above reason, the possibility to provide custom coefficients are not coded

##READING INPUT##
#Change the separator (sep) and decimal (dec) format according to input data
InputData_Stations = read.csv2("Stations.csv", sep = ',', dec = ".") #Station names and locations (traffic zone)
load("StationSharesPrediction.rda")

##PREDICTION##
StationSharesPredicted = round(predict(StationSharesPrediction,InputData_Demand)*InputData_Demand$TotalDemandPredicted)


#############################################################################################################
##Assignment of trip Distance##
#############################################################################################################
##Script details##
#Input: Distribution of trip distance of the sharing system
#Output: Trip distance
#Type of model: Statistical sampling


##READING INPUT##
#Change the separator (sep) and decimal (dec) format according to input data 
TripDistanceDistribution = read.csv2("Input_TripDistanceDistribution.csv", sep = ',', dec = ".", header = FALSE) 
TripDistanceDistribution = as.data.frame(t(as.matrix(TripDistanceDistribution)))

##SAMPLING DISTANCES##
TripDistances = Map(function(x, y) sample(x[!is.na(x)], size = y, TRUE), 
                    TripDistanceDistribution[c(-1),], StationSharesPredicted) 
TripDistances = as.data.frame(unlist(TripDistances))
names(TripDistances)[names(TripDistances) == 'unlist(TripDistances)'] <- 'TripDistance'
TripDistances$TripDistance = as.integer(TripDistances$TripDistance)
  
##ORIGIN ASSIGNMENT##
TripDistances$TripNumber = 1:nrow(TripDistances)
TripDistances$Station = rep(InputData_Stations[,1], StationSharesPredicted) 
TripDistances$OriginZone = rep(InputData_Stations[,2], StationSharesPredicted)

#############################################################################################################
##Assigning Destination##
#############################################################################################################
##Script details##
#Input: Distance between ODs, sampled distances for car-sharing trips
#Output: Trip origin and destination
#Type of model: Nearest value matching

##READING INPUT##
#Change the separator (sep) and decimal (dec) format according to input data
ODDistances = read.csv2("Input_ODDistances.csv", sep = ',', dec = ".") #Columns - Origin Zone, Desti zone, Distance
ODDistances = ODDistances[ODDistances$OriginZone!=ODDistances$DestinationZone,]

TripRequests = TripDistances%>%
  full_join(ODDistances, by = c("OriginZone")) %>%
  group_by(OriginZone,TripNumber) %>%
  mutate(myDiff = abs(TripDistance - ZoneDistance)) %>%
  slice(which.min(myDiff))
TripRequests = TripRequests %>% select(TripNumber, Station, OriginZone, DestinationZone, ZoneDistance)


#############################################################################################################
##Use frequency for the synthetic population##
#############################################################################################################
##Script details##
#Input: Synthetic population
#Output: Use frequency for each individual
#Type of model: Multinomial logit
#The coefficients are read from the file 'CoefficientValues_UseFrequency.csv'. This file is provided to allow custom values for the coefficients
#Input synthetic population should be only those in zones where car-sharing station is available
#Choices: Medium frequency (2), low frequency (1), never (0)

##READING INPUT##
#Change the separator (sep) and decimal (dec) format according to input data 
InputData_Population = read.csv2("SyntheticPopulationSample.csv", sep = ',', dec = ".")
CoefficientValues_UseFrequency = read.csv2("CoefficientValues_UseFrequency.csv", sep = ',', dec = ".", header = FALSE) #Coefficient values

##DATA PREPARATION##
InputData_Population$IsStudent = ifelse(InputData_Population$Employment == "Student",1,0)
InputData_Population$IsFullyEmployed = ifelse(InputData_Population$Employment == "Full time",1,0)
InputData_Population$IsHalfEmployed = ifelse(InputData_Population$Employment == "Half time",1,0)
InputData_Population$HasUniversityDegree = ifelse(InputData_Population$Education == "University",1,0)

InputData_Population$IsBicycleHighFrequencyUser = ifelse(InputData_Population$BicycleUseFrequency == "High",1,0)
InputData_Population$IsBicycleMediumFrequencyUser = ifelse(InputData_Population$BicycleUseFrequency == "Medium",1,0)
InputData_Population$IsPTHighFrequencyUser = ifelse(InputData_Population$PTUseFrequency == "High",1,0)
InputData_Population$IsPrivateCarLowFrequencyUser = ifelse(InputData_Population$PrivateCarUseFrequency == "Low" |
                                                             InputData_Population$PrivateCarUseFrequency == "Never",1,0)
InputData_Population$HasLowIncome = ifelse(InputData_Population$Income == "Low",1,0)
InputData_Population$IsPTAndCarUser = ifelse((InputData_Population$PTUseFrequency == "High" | InputData_Population$PTUseFrequency == "Medium") &
                                               (InputData_Population$PrivateCarUseFrequency == "High" | InputData_Population$PrivateCarUseFrequency == 
                                                  "Medium") ,1,0)
InputData = InputData_Population
CoefficientValues = CoefficientValues_UseFrequency
##PREDICTION##
SharesPrediction = function(InputData, CoefficientValues) {
  Utility = data.frame(matrix(NA, nrow = nrow(InputData), ncol = 3))
  colnames(Utility) = c("MediumFreq","LowFreq")
  
  Utility$MediumFreq = CoefficientValues[1,2] +
    CoefficientValues[3,2]*InputData$Age +
    CoefficientValues[5,2]*InputData$IsStudent +
    CoefficientValues[7,2]*InputData$IsFullyEmployed +
    CoefficientValues[9,2]*InputData$IsHalfEmployed +
    CoefficientValues[15,2]*InputData$IsBicycleHighFrequencyUser +
    CoefficientValues[17,2]*InputData$IsPTHighFrequencyUser +
    CoefficientValues[20,2]*InputData$NumberOfSharingVehiclesInZone
  
  Utility$LowFreq = CoefficientValues[2,2] +
    CoefficientValues[4,2]*InputData$Age +
    CoefficientValues[6,2]*InputData$IsStudent +
    CoefficientValues[8,2]*InputData$IsFullyEmployed +
    CoefficientValues[10,2]*InputData$IsHalfEmployed +
    CoefficientValues[11,2]*InputData$HasLowIncome +
    CoefficientValues[12,2]*InputData$HasUniversityDegree +
    CoefficientValues[13,2]*InputData$HHBicycles +
    CoefficientValues[14,2]*InputData$HHPrivateCars +
    CoefficientValues[16,2]*InputData$IsBicycleMediumFrequencyUser +
    CoefficientValues[18,2]*InputData$IsPrivateCarLowFrequencyUser +
    CoefficientValues[19,2]*InputData$IsPTAndCarUser +
    CoefficientValues[21,2]*InputData$NumberOfSharingVehiclesInZone
  
  Utility$expMediumFreq = exp(Utility$MediumFreq)
  Utility$expLowFreq = exp(Utility$LowFreq)
  Utility$TotalUtility = rowSums(Utility[,4:5])
  Utility$TotalUtility = Utility$TotalUtility + exp(0)
  
  Utility$probMediumFreq = Utility$expMediumFreq/Utility$TotalUtility
  Utility$probLowFreq = Utility$expLowFreq/Utility$TotalUtility
  Utility$probNever = exp(0)/Utility$TotalUtility
  Utility$totalProb = rowSums(Utility[,7:9])
  
  shares = colMeans(Utility[,7:9]) #, na.rm = TRUE
  return(Utility[,c(9,8,7)])
}

PredictedShares = SharesPrediction(InputData_Population,CoefficientValues_UseFrequency)
InputData_Population_UseFrequency = InputData_Population
InputData_Population_UseFrequency[(ncol(InputData_Population_UseFrequency)+1):
                                    (ncol(InputData_Population_UseFrequency)+3)] = PredictedShares
InputData_Population_UseFrequency$UseFrequency = max.col(PredictedShares, "first") - 1


#############################################################################################################
##SAVING OUTPUT##
#############################################################################################################
#Change the separator (sep) and decimal (dec) format according to your needs
write.table(TripRequests, file = "TripRequests.csv", row.names=FALSE, sep=",", dec = ".")

write.table(TripRequests, file = "SyntheticPopulationWithUseFrequency.csv", row.names=FALSE, sep=",", dec = ".")