function [destinationFlows,source,sink] = Stoch_noCon_SQZ(odmatrix, nodes, links, travelCosts, theta, reduction)
%STOCH_NOCON_SQZ Method for calculating a stochastic network loading
%
%
%SYNTAX
%   [destinationFlows,source,sink] = Stoch_noCon_SQZ(odmatrix, nodes, links, travelCosts, theta, reduction)
%
%DESCRIPTION
%   Returns the flow on the network as a bush for each destination. Every
%   link is a potential source or sink. The flow is propegated through the
%   network with weights according to Dial's methods. The capacity of links
%   can be enforced by applying appropriate link based reduction factors. 
%   These are determined using the generalized node model of Tampère et al.
%   and the squeezing faze part of the STAQ assignment procedure.
%
%INPUTS
%   ODmatrix: static origin destination matrix
%   nodes: list of all the nodes in the network.
%   links: list of all the links in the network
%   travelCosts: array of link costs
%   theta: logit scaling parameter
%   reduction: factor for enforcing link capacity
%
% See also SUE

%Initilization
totNodes = length(nodes.id);
totLinks = length(links.ID);
maxNodes = max([nodes.id]);

lNum = 1:totLinks;

strN=[links.fromNode]';
endN=[links.toNode]';
zone=[links.zone]';
type=[links.type]';
len=[links.length]';

destinationFlows = zeros(totLinks,size(odmatrix,2));
source = zeros(totNodes,size(odmatrix,2));
sink = zeros(totNodes,size(odmatrix,2));

[rp,ci,ai]=sparse_to_csr(strN,endN,lNum,maxNodes); %forward star
[rp_,ci_,ai_]=sparse_to_csr(endN,strN,lNum,maxNodes); %backward star

%Compute destination based link flows for each zone
for destination = 1:size(odmatrix,2)
    if sum(odmatrix(:,destination)) > 0
        %first find all destination links that attract flow
        bln_destination_links = find_destination_links(destination,zone,type);
        ind_destination_links = find(bln_destination_links);
                       
        %find access nodes to the destination zone
        accN = find_access_nodes(bln_destination_links,strN(bln_destination_links),rp,ai,rp_,ai_);
        
        %adjust network by removing links within the destination zone 
        %replace these links by dummy links at boundary that attract flow
        %each dummy link is assigned an attraction weight similar to a
        %distance function
        [accW]=compute_acces_weight(accN,ind_destination_links,strN,endN,len,travelCosts,type,totNodes);
        [rp_tmp,ci_tmp,ai_tmp]=sparse_to_csr([strN(~bln_destination_links);accN],[endN(~bln_destination_links);maxNodes*ones(size(accN))+1],[lNum(~bln_destination_links),totLinks+1:totLinks+numel(accN)]',maxNodes+1); %
        [rp_tmp_,ci_tmp_,ai_tmp_]=sparse_to_csr([endN(~bln_destination_links);maxNodes*ones(size(accN))+1],[strN(~bln_destination_links);accN],[lNum(~bln_destination_links),totLinks+1:totLinks+numel(accN)]',maxNodes+1); %
       
        %Compute dijkstra costs of remaining nodes and store them in a cost-ordered structure
        a = [travelCosts;accW];
        distance = dijkstra(rp_tmp_,ci_tmp_,a(ai_tmp_),maxNodes+1);
        costTree = [[nodes.id';maxNodes+1],distance];
        costTree = sortrows(costTree,2);
        
        %compute node weights according to Dials procedure
        [nodeWeights,linkWeights] = get_weights(rp_tmp,ci_tmp,ai_tmp,a,costTree,theta);
                
        %initialize node flows & link flows
        %every link in each origin zone will act as a source
        [nodeFlows,linkFlows,srcFlw] = init_flows_SQZ(odmatrix,destination,strN,endN,len,zone,type,distance,totLinks,maxNodes);
        source(:,destination)=srcFlw;
        
        %build flows vector, starting from origin(s) and downwind to the
        %destination (also nodeflows vector is used for dial)
        [linkFlows,accessFlows] = propagate_flows_SQZ(linkFlows,nodeFlows,linkWeights,nodeWeights,costTree,rp_tmp,ci_tmp,ai_tmp,totLinks,accN,reduction);
             
        %update flows of links within the destination zone
        [linkFlows,desFlw] = distribute_flows_SQZ(linkFlows,ind_destination_links,strN,endN,len,travelCosts,type,accessFlows,accN,maxNodes,totLinks,theta,reduction);
        sink(:,destination)=desFlw;
        destinationFlows(:,destination)=linkFlows;
    end
    
end
end