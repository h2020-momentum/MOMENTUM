function [ speed_correction_function ] = construct_speed_correction_function( pollutant, speed_correction_data )
% CONSTRUCT_SPEED_CORRECTION_FUNCTION Construct a function that interpolates
% the speed correction factors for the pollutant emission factor to 
% intermediate speeds.
%
% SYNTAX
%   [speed_correction_function] = construct_speed_correction_function(
%   my_pollutant, speed_correction_data )
%
% DESCRIPTION
%   Returns a function that interpolates the speed correction
%   factors for the given vecotr of pollutant.
%
% INPUTS
%   - pollutant: Vector of strings specifying pollutants. Supported 
%       choices are 'CO', 'NOX', 'PM', 'VOC' and 'CO2'
%   - speed_correction_data: Table with speed correction factors.
%
% OUTPUT
%   speed_correction_function: Function that takes a speed as argument, 
%   corresponding to a cubic spline interpolation to the data.
speeds = speed_correction_data.colheaders;

speed_correction_factors = cellfun( @(poll) get_element_from_header_names(speed_correction_data, poll, speeds), pollutant, 'UniformOutput',false);
speeds = str2double(speeds);

% make structures containing piecewise polynomials (pp)
pp_structs = cellfun( @(cf) spline(speeds, cf), speed_correction_factors,'UniformOutput',false);

% Convert pps to functions that can be evaluated for a given speed.
speed_correction_function = @(speed) cellfun(@(pp) ppval(pp, speed), pp_structs);

% In the vectorized variant, speed can be a vector itself.
%vectorized_correction_factor_f = @(speed) cell2mat(arrayfun(correction_factor_f, speed,'UniformOutput',false));

