function [nodeFlows,linkFlows,source] = init_flows_SQZ(odmatrix,destination,strN,endN,len,zone,type,distance,totLinks,totNodes)
%INIT_FLOWS_SQZ Method for initializing linkflows and nodeflows
%
%
%SYNTAX
%   [nodeFlows,linkFlows,source] = init_flows_SQZ(odmatrix,destination,strN,endN,len,zone,type,distance,totLinks,maxNodes)
%
%DESCRIPTION
%   Retuns an initialized array of node flows and links flows for every 
%   source of flow towards the destination.
%
%INPUTS
%   ODmatrix: static origin destination matrix
%   destination: destination id
%   strN: array of the upstream nodes of each link
%   endN: array of the downstream nodes of each link
%   len: array of link lengths
%   zone: array with the zone id of each link
%   type: array with the class number of each link reduction
%   distance: array of distances of each node to the destination
%   totLinks: total number links in the network
%   totNodes: total number nodes in the network
%
% See also STOCH_NOCON_SQZ


%Initialization
nodeFlows = zeros(totNodes+1,1);
source = zeros(totNodes,1);
linkFlows = zeros(totLinks,1);

%Go over every origin zone and set the initial link & node flows
for origin = find(sum(odmatrix,2))'
    if origin~=destination
        ind_origin_links = find(zone==origin & type > 2);
        ind_origin_links(distance(strN(ind_origin_links))<distance(endN(ind_origin_links)))=[];
        ind_origin_links(isinf(distance(strN(ind_origin_links))))=[];
        if isempty(ind_origin_links)
            ind_origin_links = find(zone==origin);
            ind_origin_links(distance(strN(ind_origin_links))<distance(endN(ind_origin_links)))=[];            
        end
        
        %weigth for origin links based on links length
        linkFlows(ind_origin_links) = linkFlows(ind_origin_links)+odmatrix(origin,destination)*len(ind_origin_links)/sum(len(ind_origin_links));
        for l=ind_origin_links'
            source(strN(l)) = source(strN(l)) + odmatrix(origin,destination)*len(l)/sum(len(ind_origin_links));
            nodeFlows(endN(l)) = nodeFlows(endN(l)) + odmatrix(origin,destination)*len(l)/sum(len(ind_origin_links));
        end
    end
end

end