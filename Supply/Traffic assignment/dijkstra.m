function [d,pred,list]=dijkstra(rp,ci,ai,u)
% DIJKSTRA Compute shortest paths using Dijkstra's algorithm
%
% d=dijkstra(rp,ci,ai,u) computes the shortest path from vertex u to all nodes 
% reachable from vertex u using Dijkstra's algorithm for the problem.  
% The graph is given by the sparse matrix notation in CRS format (rp,ci,ai). 
%
% See also SPARSE_TO_CSR

% David F. Gleich
% Copyright, Stanford University, 2008-2009
%
% Willem Himpe
% Copyright, KU Leuven, 2019
%
% History
% 2008-04-09: Initial coding
% 2009-05-15: Documentation
% 2019-07-29: CRS input variables

n=length(rp)-1; 
d=Inf(n,1); T=zeros(n,1); L=zeros(n,1);
pred=zeros(1,length(rp)-1);
list=zeros(length(rp)-1,1);
xout=true(length(rp)-1,1);
l_it=1;

n=1; T(n)=u; L(u)=n; % oops, n is now the size of the heap

% enter the main dijkstra loop
d(u) = 0;
while n>0
    v=T(1); ntop=T(n); T(1)=ntop; L(ntop)=1; n=n-1; % pop the head off the heap
    list(l_it)=v;
    xout(v)=false;
    l_it=l_it+1;
    
    k=1; kt=ntop;                   % move element T(1) down the heap
    while 1,
        i=2*k; 
        if i>n, break; end          % end of heap
        if i==n, it=T(i);           % only one child, so skip
        else                        % pick the smallest child
            lc=T(i); rc=T(i+1); it=lc;
            if d(rc)<d(lc), i=i+1; it=rc; end % right child is smaller
        end
        if d(kt)<d(it), break;     % at correct place, so end
        else T(k)=it; L(it)=k; T(i)=kt; L(kt)=i; k=i; % swap
        end
    end                             % end heap down
     
    % for each vertex adjacent to v, relax it
    for ei=rp(v):rp(v+1)-1            % ei is the edge index
        w=ci(ei); ew=ai(ei);          % w is the target, ew is the edge weight
        % relax edge (v,w,ew)
        if d(w)>d(v)+ew
            d(w)=d(v)+ew; pred(w)=v;
            % check if w is in the heap
            k=L(w); onlyup=0; 
            if k==0
                % element not in heap, only move the element up the heap
                n=n+1; T(n)=w; L(w)=n; k=n; kt=w; onlyup=1;
            else kt=T(k);
            end
            % update the heap, move the element down in the heap
            while 1 && ~onlyup,
                i=2*k; 
                if i>n, break; end          % end of heap
                if i==n, it=T(i);           % only one child, so skip
                else                        % pick the smallest child
                    lc=T(i); rc=T(i+1); it=lc;
                    if d(rc)<d(lc), i=i+1; it=rc; end % right child is smaller
                end
                if d(kt)<d(it), break;      % at correct place, so end
                else T(k)=it; L(it)=k; T(i)=kt; L(kt)=i; k=i; % swap
                end
            end
            % move the element up the heap
            j=k; tj=T(j);
            while j>1,                       % j==1 => element at top of heap
                j2=floor(j/2); tj2=T(j2);    % parent element
                if d(tj2)<d(tj), break;      % parent is smaller, so done
                else                         % parent is larger, so swap
                    T(j2)=tj; L(tj)=j2; T(j)=tj2; L(tj2)=j; j=j2;
                end
            end  
        end
    end
end
list(l_it:end)=find(xout);
% figure;plot(N(1:sl_it));
end