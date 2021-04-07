function access_nodes = find_access_nodes(bln_destination_links,strN,rp,ai,rp_,ai_)
%Method for finding access nodes to the destination
%
%
%SYNTAX
%   access_nodes = find_access_nodes(bln_destination_links,strN,rp,ai,rp_,ai_)
%
%DESCRIPTION
%   Returns an array of access nodes.
%
%INPUTS
%   bln_destination_links: array of booleans with true for each link in the
%   destination
%   strN: array with upstream node id of each link in the destination
%   rp: dense notation for outgoing links of each node
%   ai: dense notation for outgoing link ids
%   rp_: dense notation for incoming links of each node
%   ai_: dense notation for incoming links ids


totN = length(strN);
flag = false(totN,1);
for i = 1:length(strN)
    k=strN(i);
    ei_ = rp_(k):rp_(k+1)-1;
    l_in=ai_(ei_);%incoming links
    ei = rp(k):rp(k+1)-1;
    l_out=ai(ei);%outgoing links
    
    %flag link if one outgoing link of upstream node is part of the 
    %destination zone and at least one incoming link is not.
    if any(bln_destination_links(l_out)) && any(~bln_destination_links(l_in))
        flag(i)=true;
    end
end

%
access_nodes = unique(strN(flag));
