function costs = calculateCostBPR(alpha,beta,flows,lengths,speeds,caps,cost_extra)
%Calculates the costs on a network according to the BPR curve
%
%SYNTAX
%   [costs] = calculateCostBPR(alpha,beta,flows,lengths,speeds,caps) 
%
%DESCRIPTION
%   returns the costs on a network according to the BPR curve. Note that
%   the free flow travel time is calculated within the function.
%
%INPUTS
%   alpha: parameter that captures the additional travel time at capacity
%   beta: parameter that handles the slope of the increase in travel time
%   flows: total flow over each link
%   lengths: length of each link
%   speeds: maximum speed of each link
%   caps: capacity of each link
%   cost_extra: extra cost of each link (expreseed in seconds)
%
% OUTPUT
%   costs: travel cost expressed in minutes

costs = lengths./speeds*60.*(1+alpha.*(flows./caps).^beta)+cost_extra/60;