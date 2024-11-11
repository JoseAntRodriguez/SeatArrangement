int:k;
set of int:n = 0..(k-1);
array[n,n] of int: V;
array[n,n] of int: G;
int: uType; % 1 if S-utility, 2 if B-utility, 3 if W-utility
int: objective; % 1 if MWA, 2 if MUA, 3 if EFA, 4 if STA
int: minVal = min(p in n, q in n)(V[p,q]);
int: maxVal = max(p in n, q in n)(V[p,q]);
array[n] of var 0..1:degree;
constraint forall(u in n)(if sum(v in n)(G[u,v]) == 0 then degree[u] = 0 else degree[u] = 1 endif);

function array[int] of int: neighbours(int: u) = [v|v in n where G[u,v] == 1];
function int: degree(int: u) = sum(v in n)(G[u,v]);

array[n,n] of var 0..1:x;
constraint forall(p in n)(sum(u in n)(x[p,u]) = 1);
constraint forall(u in n)(sum(p in n)(x[p,u]) = 1);

function var int: utilityAtVertex(int: p, int: u, int: uT) = if degree(u) == 0 then 0 elseif uT == 2 then max(v in neighbours(u))(sum(q in n)(x[q,v]*V[p,q])) elseif uT == 3 then min(v in neighbours(u))(sum(q in n)(x[q,v]*V[p,q])) else sum(q in n, v in neighbours(u))(x[q,v]*V[p,q]*G[u,v]) endif;

function var int: utility(int: p, int: uT) = sum(u in n)(utilityAtVertex(p,u,uT)*x[p,u]);

var int: goal = if objective == 1 then sum([utility(p, uType) | p in n]) else min([utility(p, uType) | p in n]) endif;
solve ::int_search(x, first_fail, indomain_min) maximize goal;