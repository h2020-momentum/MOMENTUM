function [linkFlows,sink] = distribute_flows_SQZ(linkFlows,ind_destination_links,strN,endN,len,travelCosts,type,accessFlows,accN,totNodes,totLinks,theta,reduction)
%DISTRIBUTE_FLOWS_SQZ Method for distributing the linkflows over the links in the destination zone
%
%
%SYNTAX
%   [linkFlows,sink] = distribute_flows_SQZ(linkFlows,ind_destination_links,strN,endN,len,travelCosts,type,dFl_con,conN,maxNodes,totLinks,idN,theta,red_fact)
%
%DESCRIPTION
%   Processes the links flows after they are initilized at the sources by 
%   propagating them over the network until they reach the destination at 
%   the access nodes. 
%
%INPUTS
%   linkFlows: array of link flows in the network
%   ind_destination_links: list of indices of all links that act as a sink
%   in the destination
%   strN: array of the upstream nodes of each link
%   endN: array of the downstream nodes of each link
%   len: array of link lengths
%   travelCosts: array of link costs
%   type: array with the class number of each link reduction
%   accessFlows: array of access node flows
%   accN: list of access nodes
%   totNodes: total number of nodes in the network
%   totLinks: total number of links in the network
%   theta: logit scaling parameter
%   reduction: factor for enforcing link capacity
%
% See also PROPAGATE_FLOWS_SQZ, STOCH_NOCON_SQZ
sink = zeros(totNodes,1);

[rp_tmp,ci_tmp,ai_tmp] = sparse_to_csr(strN(ind_destination_links),endN(ind_destination_links),ind_destination_links,totNodes);
[rp_tmp_,ci_tmp_,ai_tmp_] = sparse_to_csr(endN(ind_destination_links),strN(ind_destination_links),ind_destination_links,totNodes);
a_tmp = travelCosts(ai_tmp);
a_tmp_ = travelCosts(ai_tmp_);

for ind = 1:length(accN)
    if accessFlows(ind)==0
        continue;
    end
    d = accN(ind);
    distance = dijkstra(rp_tmp,ci_tmp,a_tmp,d);
    
    %node order for stochastic loads
    costTree = [1:totNodes;distance']';
    costTree = sortrows(costTree,2);
    
    %visit nodes in costTree starting from k=2
    nodeWeights = zeros(totNodes,1);
    linkWeights = zeros(totLinks,1);
    nodeFlows = zeros(totNodes,1);
    nodeWeights(d) = 1;
    totNodes_tmp = find(isinf(costTree(:,2)),1,'first')-1;
    for k = 2:totNodes_tmp
        nodeInd = costTree(k,1);
        %get which links enter node k
        nodeWeights(nodeInd) = 0; %?
        for ei = rp_tmp_(nodeInd):rp_tmp_(nodeInd+1)-1
            l=ai_tmp_(ei);%edge
            upNodeInd = ci_tmp_(ei); %get node id
            linkWeights(l) = nodeWeights(upNodeInd) * exp(-travelCosts(l)*theta);
            nodeWeights(nodeInd) = nodeWeights(nodeInd) + linkWeights(l);
        end
    end
    
    %Build flows vector, starting from the destinations propegate backwards to the boundary node(s)
    ind_dest_links = ind_destination_links(type(ind_destination_links) > 2);
    ind_dest_links(distance(endN(ind_dest_links))<distance(strN(ind_dest_links)))=[];
    ind_dest_links(isinf(distance(endN(ind_dest_links))))=[];
    if isempty(ind_dest_links)
        ind_dest_links = ind_destination_links;
        ind_dest_links(distance(endN(ind_dest_links))<distance(strN(ind_dest_links)))=[];
        ind_dest_links(isinf(distance(endN(ind_dest_links))))=[];
        
    end
    
    for l=ind_dest_links'
        nodeFlows(endN(l)) = nodeFlows(endN(l)) + accessFlows(ind)*len(l)/sum(len(ind_dest_links));
    end
   
    sink = sink + nodeFlows;
    
    %beginning from n
    for  k = totNodes_tmp:-1:2
        nodeInd = costTree(k,1); %get node Id
        %get which links enter node k
        for ei = rp_tmp_(nodeInd):rp_tmp_(nodeInd+1)-1
            l=ai_tmp_(ei);
            upNodeInd = ci_tmp_(ei); %get node id
            if ( nodeWeights(nodeInd) == 0 )
                continue;
            end
            linkFlows(l) = linkFlows(l) + nodeFlows(nodeInd)*linkWeights(l)/nodeWeights(nodeInd);
            nodeFlows(upNodeInd) = nodeFlows(upNodeInd) + reduction(l)*nodeFlows(nodeInd)*linkWeights(l)/nodeWeights(nodeInd);
        end
    end
end