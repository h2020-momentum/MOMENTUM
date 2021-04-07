function [ demand ] = logit_based_variable_demand( utility, cal_param, base_utility, base_demand )
%LOGIT_BASED_VARIABLE_DEMAND Calculate new variable demand value, using the
%logit-based approach
%
% SYNTAX
%   [new_demand] = logit_based_variable_demand( utility, cal_param,
%   base_utility, base_demand )
%
% DESCRIPTION
%   Returns the new demand as calculated through the logit-based formula
%   for variable demand calibration. One is free to choose the method for
%   determining the calibration parameter.
%
%INPUTS
%   Utility, cal_param, base_utility and base_demand can be either single
%   values or matrices where each element contains an entry that is related
%   to the elements of the OD-matrix:
%       - utility: The utility of the OD pair in the situation for which 
%       the demand needs to be calculated.
%       - cal_param: Properly calculated calibration parameter for each OD
%       pair. One is free to choose how the calibration parameter is
%       calculated. One option is to use demand elasticities. For
%       consistency with the spirit of the model, cal_param must be
%       strictly positive.
%       - base_utility: The utility of the OD pair in the base-scenario
%       - base_demand: The demand of the OD pair in the base-scenario
%
% OUTPUT
%   The new demand, either for a single OD pair or for the OD matrix
%   corresponding to the input parameters.
exp_U_0 = exp(base_utility);
exp_U = exp(utility);

demand = (exp_U ./ exp_U_0) .* (cal_param + exp_U_0) ./ (cal_param + exp_U) .* base_demand;

