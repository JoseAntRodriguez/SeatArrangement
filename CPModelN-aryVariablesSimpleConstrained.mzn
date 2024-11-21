include "alldifferent.mzn"; 

int:k;
set of int:n = 0..(k-1);
array[n,n] of int: V;
array[n,0..1] of int: G;

%array[n] of var n:p;%permutation
array[n] of var n:q;
constraint alldifferent(q);
%constraint forall(i in n, u in n)(if p[i] == u then q[u] == i endif);

%function var int: utility(int: i) = sum([G[p[i],p[j]]*V[i,j] | j in n]);
%function var int: utility(int: i) = V[i,q[G[p[i],0]]] + V[i,q[G[p[i],1]]];
function var int: utility(int: u) = V[q[u],q[G[u,0]]] + V[q[u],q[G[u,1]]];
function array[int] of int:connectedComponent(int: u) = [v|v in (u+1)..(k-1) where sum(w in u..v)(G[w,v]) >= 1];

constraint forall(i in n)(utility(i) <= 2);
constraint forall(i in n)(utility(i) >= 0);
%constraint forall(u in 1..(k-1))(if G[u,u-1] == 0 then forall(v in connectedComponent(u))(q[u] < q[v]) endif);
% In every connected component, the "smallest" agent goes in the "smallest" vertex
constraint forall(u in 0..(k-2))(if G[u,0] > u then forall(v in (u+1)..(G[u,1]))(q[u] < q[v]) endif);
%constraint forall(u in 1..(k-1))(if G[u,0] > u then q[u+1] < q[G[u,1]] endif); % Test with this constraint

solve ::int_search(q, first_fail, indomain_min) maximize sum([utility(i) | i in n]);