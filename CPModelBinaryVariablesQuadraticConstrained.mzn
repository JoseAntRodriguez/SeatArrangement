int:k;
set of int:n = 0..(k-1);
array[n,n] of int: V;
array[n,0..1] of int: G;
int: uType; % 1 if S-utility, 2 if B-utility, 3 if W-utility
int: objective; % 1 if MWA, 2 if MUA, 3 if EFA, 4 if STA
int: minVal = min(p in n, q in n)(V[p,q]);
int: maxVal = max(p in n, q in n)(V[p,q]);

array[n,n] of var 0..1:x;
constraint forall(p in n)(sum(u in n)(x[p,u]) = 1);
constraint forall(u in n)(sum(p in n)(x[p,u]) = 1);

function var int: utilityAtVertex(int: p, int: u, int: uT, array[n,n] of var n: z) = if uT == 2 then max(sum(q in n)(z[q,G[u,0]]*V[p,q]), sum(q in n)(z[q,G[u,1]]*V[p,q])) elseif uT == 3 then min(sum(q in n)(z[q,G[u,0]]*V[p,q]), sum(q in n)(z[q,G[u,1]]*V[p,q])) else sum(q in n)(z[q,G[u,0]]*V[p,q] + z[q,G[u,1]]*V[p,q]) endif;

function var int: utility(int: p, int: uT, array[n,n] of var n: z) = sum(u in n)(utilityAtVertex(p,u,uT,z)*z[p,u]);

function var int: exchangeVariable(int: p, int: q, int: r, int: w) = if r == p then if x[q,w] == 1 then 1 else 0 endif elseif r == q then if x[p,w] == 1 then 1 else 0 endif else x[r,w] endif;

function array[n,n] of var 0..1: exchangeVariables(int: p, int: q) = array2d(n,n,[exchangeVariable(p,q,r,w)|r in n, w in n]);

function var 0..1: envy(int: p, int: q) = if utility(p, uType, exchangeVariables(p,q)) > utility(p, uType, x) then 1 else 0 endif;

constraint if objective == 3 then forall(u in n)(forall(v in n)(envy(u,v) = 0)) endif;
constraint if objective == 4 then forall(u in n)(forall(v in (u+1)..(k-1))(envy(u,v) + envy(v,u) <= 1)) endif;

% New constraints to account for the fact that valuations are binary and the seat graph is a cycle seat graph
constraint forall(p in n)(if uType == 1 then utility(p, uType, x) <= 2 else utility(p, uType, x) <= 1 endif);
constraint forall(p in n)(utility(p, uType, x) >= 0);

function array[int] of int:connectedComponent(int: u) = [v|v in (u+1)..(k-1) where sum(w in u..v)(G[w,v]) >= 1];
constraint forall(u in 0..(k-2))(if G[u,0] > u then forall(v in (u+1)..(G[u,1]))(forall(p in n)(if x[p,u] == 1 then forall(q in 0..(p-1))(x[q,v]=0) endif)) endif);
constraint forall(u in 0..(k-2))(if G[u,0] > u then forall(p in n)(if x[p,G[u,0]] == 1 then forall(q in 0..(p-1))(x[q,G[u,1]]=0) endif) endif);

var int: goal = if objective == 1 then sum([utility(u, uType, x) | u in n]) elseif objective == 2 then min([utility(u, uType, x) | u in n]) else 0 endif;
solve ::int_search(x, first_fail, indomain_min) maximize goal;