function [ link_emissions ] = calculate_link_emissions( length, PCE, speed, emission_factors, speed_correction_factor_functions )
%CALCULATE_LINK_EMISSIONS Calculate the emissions for a link
%SYNTAX
%   [ link_emissions ] = calculate_link_emissions( length, PCE, speed,
%   emission_factors, speed_correction_factor_functions )
%
%DESCRIPTION
%   Calculates the emissions for different pollutants caused by traffic on
%   a link
%
%INPUTS
%   length: length of the link
%   PCE: passenger car equivalent that traversed the link
%   speed: speed that is reached on the link
%   emission_factors: emission factors for the different pollutants
%   speed_correction_factor_functions: functions that give the speed
%   correction factor for the different pollutants as a function of speed.
%
% OUTPUT
%   Emissions of the different pollutants caused by traffic on the link
link_emissions = length * PCE * emission_factors.*speed_correction_factor_functions(speed);