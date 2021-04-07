function [flows,destinationFlows,travelCosts,source,sink] = SUE(odmatrix,nodes,links,theta,varargin)
%Method of successive averages for calculating stochastic user equilibrium
%For fixed ordering of nodes based on free flow conditions
%
%SYNTAX
%   [flows,destinationFlows,travelCosts,source,sink] = SUE(odmatrix,nodes,links,theta,destinationFlows,source,sink,red_factor)
%
%DESCRIPTION
%   returns the flow on each link in the stochastic user equilibrium as
%   calculated by the method of successive averages
%
%INPUTS
%   odmatrix: static origin/destination matrix
%   nodes: table with all the nodes in the network.
%   links: table with all the links in the network
%   theta: stochastic distribution parameter (related to the value of time) 

%setup the output figure
h = figure;
semilogy(0,NaN);
start_time = cputime;
tic;

%Maximum number of iterations
maxIt = 10; 

%Initialize the stop criterion
epsilon = 10^-0;

%Initilization
totLinks = length(links.ID);
totNodes = length(nodes.ID);
strN = [links.fromNode]';
endN = [links.toNode]';
if nargin==7
    destinationFlows=varargin{1};
    source=varargin{2};
    sink=varargin{3};
    red_fact = varargin{4};
else
    destinationFlows=zeros(totLinks,size(odmatrix,1));
    source=zeros(totNodes,size(odmatrix,1));
    sink=zeros(totNodes,size(odmatrix,1));
    red_fact = ones(totLinks,1);
end

%initialize the travel cost
%note that the total flow on a link is the sum of the destination based flow
%on that link
alpha = links.alpha';
beta = links.beta';
travelCosts = calculateCostBPR(alpha,beta,sum(destinationFlows,2),[links.length]',[links.freeSpeed]',[links.capacity]',[links.cost_extra]');
    
%Initialize the iteration numbering
it = 0;
    
%initialize the gap function
gap = inf;

%MAIN LOOP: iterate until convergence is reached or maximum number of
%iterations is reached
while it < maxIt && gap > epsilon 
    it = it+1;
  
    %Compute new flows
    [newDestFlows,newSource,newSink] = Stoch_noCon_SQZ(odmatrix,nodes,links,travelCosts,theta,red_fact);

    %calculate the update step
    update_flows = newDestFlows - destinationFlows;
    update_sink = newSink - sink;
    update_source = newSource - source;
    
    %calculate new flows with Polyak averaging step
    destinationFlows = destinationFlows + 1/it^(2/3)*update_flows;
    source = source + 1/it^(2/3)*update_source;
    sink = sink + 1/it^(2/3)*update_sink;
    
    %update costs
    travelCosts = calculateCostBPR(alpha,beta,sum(destinationFlows,2),[links.length]',[links.freeSpeed]',[links.capacity]',[links.cost_extra]');

    %convergence gap
    gap = max(max(max(update_flows)),-min(min(update_flows)));
        
    %plot convergence
    figure(h) 
    hold on
    semilogy(cputime-start_time,gap,'r.')
    drawnow
end
display(['it: ',num2str(it)]);
display(['gap (veh/h): ', num2str(gap)]);
display(['max update flow (veh/h): ',num2str(max(max(abs(update_flows))))]);
display(['relative gap: ',num2str(gap/sum(sum(odmatrix)))]);
display(['total Time ALGD (CPU) ',num2str(cputime-start_time), 's']);
display(['total Time ALGD (Time) ',num2str(toc), 's']);

%Check for number of iterations until convergence
if it >= maxIt 
    disp(['Maximum Iteration limit reached: ', num2str(maxIt)]);
else
    disp(['Converged in iteration ', num2str(it)]);
end

%Return the total flow for every linnk (sum over all destinations)
flows = sum(destinationFlows,2);

end