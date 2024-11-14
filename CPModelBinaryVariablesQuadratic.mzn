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

function var int: utilityAtVertex(int: p, int: u, int: uT, array[n,n] of var n: z) = if degree(u) == 0 then 0 elseif uT == 2 then max(v in neighbours(u))(sum(q in n)(z[q,v]*V[p,q])) elseif uT == 3 then min(v in neighbours(u))(sum(q in n)(z[q,v]*V[p,q])) else sum(q in n, v in neighbours(u))(z[q,v]*V[p,q]*G[u,v]) endif;

function var int: utility(int: p, int: uT, array[n,n] of var n: z) = sum(u in n)(utilityAtVertex(p,u,uT,z)*z[p,u]);

function var int: exchangeVariable(int: p, int: q, int: r, int: w) = if r == p then if x[q,w] == 1 then 1 else 0 endif elseif r == q then if x[p,w] == 1 then 1 else 0 endif else x[r,w] endif;

function array[n,n] of var 0..1: exchangeVariables(int: p, int: q) = array2d(n,n,[exchangeVariable(p,q,r,w)|r in n, w in n]);

function var 0..1: envy(int: p, int: q) = if utility(p, uType, exchangeVariables(p,q)) > utility(p, uType, x) then 1 else 0 endif;

constraint if objective == 3 then forall(u in n)(forall(v in n)(envy(u,v) = 0)) endif;
constraint if objective == 4 then forall(u in n)(forall(v in (u+1)..(k-1))(envy(u,v) + envy(v,u) <= 1)) endif;

var int: goal = if objective == 1 then sum([utility(u, uType, x) | u in n]) elseif objective == 2 then min([utility(u, uType, x) | u in n]) else 0 endif;
solve ::int_search(x, first_fail, indomain_min) maximize goal;