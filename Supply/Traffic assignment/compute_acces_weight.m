function [accW] = compute_acces_weight(accN,ind_destination_links,strN,endN,len,travelCosts,type,totNodes)
%COMPUTE_ACCESS_WEIGHT Method for computing the access weight towards the destination zone
%
%
%SYNTAX
%   [accW] = compute_acces_weight(accN,ind_destination_links,strN,endN,len,travelCosts,type,totNodes)
%
%DESCRIPTION
%   Returns the access weight of each access node to a destination. Note 
%   that every link in the destination is a potential sink. The flow is 
%   propegated from the access point towards each of the links and it is
%   attracted proportional to the length it represents in the destination
%   zone
%
%INPUTS
%   accN: list of access nodes
%   ind_destination_links: list of indices of all links that act as a sink
%   in the destination
%   strN: array of the upstream nodes of each link
%   endN: array of the downstream nodes of each link
%   len: array of link lengths
%   travelCosts: array of link costs
%   type: array with the class number of each link reduction
%   totNodes: total number nodes in the network
%
% See also STOCH_NOCON_SQZ

%Initialization
accW = inf(size(accN));
cnt_dist = zeros(size(accN));
w_dist = zeros(size(accN));
con_imp = ones(size(accN))/numel(accN);

%Compute cost & flow to boundary links from every node
%first adjust link connection mapping (only use subnetwork in zone)
%should be utility to be fully consistent (now just based on shortest path
%distance)
[rp_tmp,ci_tmp,ai_tmp] = sparse_to_csr(strN(ind_destination_links),endN(ind_destination_links),ind_destination_links,totNodes);
a_tmp = travelCosts(ai_tmp);

if any(type(ind_destination_links) > 2)
    %compute average distance to boundary nodes
    for ind = 1:length(accN)
        d = accN(ind);
        distance = dijkstra(rp_tmp,ci_tmp,a_tmp,d);
        %find links that could sent flow out of zone
        if any(~isinf(distance(endN(ind_destination_links))) & distance(endN(ind_destination_links)) > distance(strN(ind_destination_links)) & type(ind_destination_links) > 2) && con_imp(ind)>0
            accW(ind) = sum(distance(endN(ind_destination_links(~isinf(distance(endN(ind_destination_links)))& distance(endN(ind_destination_links)) > distance(strN(ind_destination_links)) & type(ind_destination_links) > 2))));
            cnt_dist(ind)= nnz(~isinf(distance(endN(ind_destination_links)))& distance(endN(ind_destination_links)) > distance(strN(ind_destination_links)) & type(ind_destination_links) > 2);
            w_dist(ind)= sum(len(ind_destination_links).*(~isinf(distance(endN(ind_destination_links)))& distance(endN(ind_destination_links)) > distance(strN(ind_destination_links)) & type(ind_destination_links) > 2));
        end
    end
else    
    for ind = 1:length(accN)
        d = accN(ind);
        distance = dijkstra(rp_tmp,ci_tmp,a_tmp,d);
        %find links that could sent flow out of zone
        if any(~isinf(distance(endN(ind_destination_links))) & distance(endN(ind_destination_links)) > distance(strN(ind_destination_links))) && con_imp(ind)>0
            accW(ind) = sum(distance(endN(ind_destination_links(~isinf(distance(endN(ind_destination_links)))& distance(endN(ind_destination_links)) > distance(strN(ind_destination_links))))));
            cnt_dist(ind)= nnz(~isinf(distance(endN(ind_destination_links)))& distance(endN(ind_destination_links)) > distance(strN(ind_destination_links)));
            w_dist(ind)= sum(len(ind_destination_links).*(~isinf(distance(endN(ind_destination_links)))& distance(endN(ind_destination_links)) > distance(strN(ind_destination_links))));
        end
    end
end

accW(accW>0)=accW(accW>0)./(cnt_dist(accW>0).*w_dist(accW>0));

end