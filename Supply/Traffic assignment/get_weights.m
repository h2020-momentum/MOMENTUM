function [nodeWeights,linkWeights] = get_weights(rp,ci,ai,a,costTree,theta)
%GET_WEIGHTS Method for calculating the weights of a stochastic network loading
%
%
%SYNTAX
%   [nodeWeights,linkWeights] = get_weights(rp,ci,ai,a,costTree,theta)
%
%DESCRIPTION
%   Calculate node weights & link weights according to Dial's procedure
%   (1971). The weights can be used to assign flow in the network according
%   to a logit distribution over the links that are accessible through the 
%   topological order of the upstream and downstream nodes.
%
%INPUTS
%   rp: dense notation for outgoing links of each node
%   ci: dense notation for the end nodes of the outgoing links of each node
%   ai: dense notation for outgoing link ids
%   a: array of link costs
%   costTree: ordered list of nodes ascending according to distance
%   theta: logit scaling parameter
%
% See also STOCH_NOCON_SQZ

%Initialization
nodeWeights = zeros(size(costTree,1),1);
linkWeights = zeros(length(a),1);
nodeWeights(end) = 1;

%visit nodes in costTree stareting from k=2
for k = 2:size(costTree,1)
    nodeInd = costTree(k,1);
    %get which links enter node k
    nodeWeights(nodeInd) = 0; %?
    for ei = rp(nodeInd):rp(nodeInd+1)-1
        l=ai(ei);%edge
        downNodeInd = ci(ei); %get node id
        
        linkWeights(l) = nodeWeights(downNodeInd) * exp(-a(l)*theta);
        nodeWeights(nodeInd) = nodeWeights(nodeInd) + linkWeights(l);
    end
end

end