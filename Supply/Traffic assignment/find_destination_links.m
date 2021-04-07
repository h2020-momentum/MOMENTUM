function bln_destination_links = find_destination_links(destination,zone,type)
%FIND_DESTINATION_LINKS Method for finding links that act as a sink in a destination zone
%
%
%SYNTAX
%   bln_destination_links = find_destination_links(destination,strN,endN,rp,rp_,zone,type)
%
%DESCRIPTION
%   Returns an array of booleans. True if a link in the destination zone is
%   a sink.
%
%INPUTS
%   destination: destination id
%   zone: array with the zone id of each link
%   type: array with the class number of each link reduction
%
% See also STOCH_NOCON_SQZ

bln_destination_links = zone==destination;

if any(type(bln_destination_links) > 2)
    %only urban links attract flow in this zone
    bln_destination_links = zone==destination & type > 2;
end

