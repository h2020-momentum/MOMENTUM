function [linkFlows,accessFlows] = propagate_flows_SQZ(linkFlows,nodeFlows,linkWeights,nodeWeights,costTree,rp,ci,ai,totLinks,accN,reduction)
%PROPAGATE_FLOWS_SQZ Method for propagating linkflows to the destination zone
%
%
%SYNTAX
%   [linkFlows,accessFlows] = propagate_flows_SQZ(linkFlows,nodeFlows,linkWeights,nodeWeights,costTree,rp,ci,ai,totLinks,accN,reduction)
%
%DESCRIPTION
%   Processes the links flows after they are initilized at the sources by 
%   propagating them over the network until they reach the destination at 
%   the access nodes. 
%
%INPUTS
%   linkFlows: array of link flows in the network
%   nodeFlows: array of node flows in the network
%   linkWeights: array of link weights in the network
%   nodeWeights: array of node weights in the network
%   costTree: ordered list of nodes ascending according to distance
%   rp: dense notation for outgoing links of each node
%   ci: dense notation for the end nodes of the outgoing links of each node
%   ai: dense notation for outgoing link ids
%   totLinks: total number links in the network
%   accN: list of access nodes
%   reduction: factor for enforcing link capacity
%
% See also INIT_FLOWS_SQZ, STOCH_NOCON_SQZ

accessFlows = zeros(size(accN));
%beginning from n
for  k = size(costTree,1):-1:2
    nodeInd = costTree(k,1); %get node Id
    %get which links enter node k
    for ei = rp(nodeInd):rp(nodeInd+1)-1
        l=ai(ei);
        downNodeInd = ci(ei); %get node id
        if ( nodeWeights(nodeInd) == 0 )
            continue;
        end
        
        if l < totLinks + 1
            %link is in actual network
            linkFlows(l) = linkFlows(l) + nodeFlows(nodeInd)*linkWeights(l)/nodeWeights(nodeInd);
            nodeFlows(downNodeInd) = nodeFlows(downNodeInd) + reduction(l)*nodeFlows(nodeInd)*linkWeights(l)/nodeWeights(nodeInd);
        else
            %link is hypotethical connection to destination zone 
            accessFlows(l-totLinks) = accessFlows(l-totLinks) + nodeFlows(nodeInd)*linkWeights(l)/nodeWeights(nodeInd);
            nodeFlows(downNodeInd) = nodeFlows(downNodeInd) + nodeFlows(nodeInd)*linkWeights(l)/nodeWeights(nodeInd);
        end
        
    end
end
end