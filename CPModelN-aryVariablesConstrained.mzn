include "alldifferent.mzn";
int:k;
set of int:n = 0..(k-1);
array[n,n] of int: V;
array[n,0..1] of int: G;
int: uType; % 1 if S-utility, 2 if B-utility, 3 if W-utility
int: objective; % 1 if MWA, 2 if MUA, 3 if EFA, 4 if STA
int: minVal = min(i in n, j in n)(V[i,j]);
int: maxVal = max(i in n, j in n)(V[i,j]);

function array[int] of int: neighbours(int: u) = [v|v in n where G[u,v] == 1];

array[n] of var n:x;%permutation
constraint alldifferent(x);

function var int: utility(int: u, int: uT, array[n] of var n: p) = if uT == 2 then max(V[p[u],p[G[u,0]]], V[p[u],p[G[u,1]]]) elseif uT == 3 then min(V[p[u],p[G[u,0]]], V[p[u],p[G[u,1]]]) else V[p[u],p[G[u,0]]] + V[p[u],p[G[u,1]]] endif;

function var int: exchangeVariable(int: u, int: v, int: r) = if r == u then x[v] elseif r == v then x[u] else x[r] endif;

function array[n] of var n: exchangeVariables(int: u, int: v) = array1d(n,[exchangeVariable(u,v,r)|r in n]);

function var 0..1: envy(int: u, int: v) = if utility(v, uType, exchangeVariables(u,v)) > utility(u, uType, x) then 1 else 0 endif;

constraint if objective == 3 then forall(u in n)(forall(v in n)(envy(u,v) = 0)) endif;
constraint if objective == 4 then forall(u in n)(forall(v in (u+1)..(k-1))(envy(u,v) + envy(v,u) <= 1)) endif;

% New constraints to account for the fact that valuations are binary and the seat graph is a cycle seat graph
constraint forall(u in n)(if uType == 1 then utility(u, uType, x) <= 2 else utility(u, uType, x) <= 1 endif);
constraint forall(u in n)(utility(u, uType, x) >= 0);

function array[int] of int:connectedComponent(int: u) = [v|v in (u+1)..(k-1) where sum(w in u..v)(G[w,v]) >= 1];
constraint forall(u in 0..(k-2))(if G[u,0] > u then forall(v in (u+1)..(G[u,1]))(x[u] < x[v]) endif);

var int: goal = if objective == 1 then sum([utility(u, uType, x) | u in n]) elseif objective == 2 then min([utility(u, uType, x) | u in n]) else 0 endif;
solve ::int_search(x, first_fail, indomain_min) maximize goal;