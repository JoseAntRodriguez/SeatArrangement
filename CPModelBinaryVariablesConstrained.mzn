int:k;
set of int:n = 0..(k-1);
array[n,n] of int: V;
array[n,0..1] of int: G;
int: uType; % 1 if S-utility, 2 if B-utility, 3 if W-utility
int: objective; % 1 if MWA, 2 if MUA, 3 if EFA, 4 if STA
int: minVal = min(p in n, q in n)(V[p,q]);
int: maxVal = max(p in n, q in n)(V[p,q]);

function int: neighbours(int: u, int: v) = if G[u,0] == v \/ G[u,1] == v then 1 else 0 endif;

array[n,n] of var 0..1:x;
constraint forall(p in n)(sum(u in n)(x[p,u]) = 1);
constraint forall(u in n)(sum(p in n)(x[p,u]) = 1);

array[n,n,n,n] of var 0..1:y;
constraint forall(p in n, q in n, u in n, v in n) (y[p,q,u,v] == if x[p,u] + x[q,v] == 2 then neighbours(u,v) else 0 endif);

function var int: utilityAtVertex(int: p, int: u, int: uT, array[n,n,n,n] of var 0..1:z) = if uT == 2 then max(sum(q in n)(z[p,q,u,G[u,0]]*V[p,q]), sum(q in n)(z[p,q,u,G[u,1]]*V[p,q])) elseif uT == 3 then min(sum(q in n)(z[p,q,u,G[u,0]]*V[p,q]), sum(q in n)(z[p,q,u,G[u,1]]*V[p,q])) else sum(q in n)(z[p,q,u,G[u,0]]*V[p,q]) + sum(q in n)(z[p,q,u,G[u,1]]*V[p,q]) endif;

function var int: utility(int: p, int: uT, array[n,n,n,n] of var 0..1:z) = sum(u in n)(utilityAtVertex(p,u,uT,z));

function var int: exchangeVariable(int: p, int: q, int: r, int: s, int: w, int: w2) =
if r == p \/ s == p then if r == p then if s == q then y[q,p,w,w2] else y[q,s,w,w2] endif else if r == q then y[p,q,w,w2] else y[r,q,w,w2] endif endif
elseif r == q \/ s == q then if r == q then if s == p then y[p,q,w,w2] else y[p,s,w,w2] endif else if r == p then y[q,p,w,w2] else y[r,p,w,w2] endif endif
else y[r,s,w,w2] endif;

function array[n,n,n,n] of var 0..1: exchangeVariables(int: p, int: q) = array4d(n,n,n,n,[exchangeVariable(p,q,r,s,w,w2)|r in n, s in n, w in n, w2 in n]);

function var 0..1: envy(int: p, int: q) = if utility(p, uType, exchangeVariables(p,q)) > utility(p, uType, y) then 1 else 0 endif;

constraint if objective == 3 then forall(u in n)(forall(v in n)(envy(u,v) = 0)) endif;
constraint if objective == 4 then forall(u in n)(forall(v in (u+1)..(k-1))(envy(u,v) + envy(v,u) <= 1)) endif;

% New constraints to account for the fact that valuations are binary and the seat graph is a cycle seat graph
constraint forall(p in n)(utility(p, uType, y) <= 2);
constraint forall(p in n)(utility(p, uType, y) >= 0);

function array[int] of int:connectedComponent(int: u) = [v|v in (u+1)..(k-1) where sum(w in u..v)(G[w,v]) >= 1];
constraint forall(u in 1..(k-1))(if G[u,0] > u then forall(v in (u+1)..(G[u,1]))(forall(p in n)(if x[p,u] == 1 then forall(q in 0..(p-1))(x[q,v]=0) endif)) endif);

var int: goal = if objective == 1 then sum([utility(p, uType, y) | p in n]) elseif objective == 2 then min([utility(p, uType, y) | p in n]) else 0 endif;
solve ::int_search(x, first_fail, indomain_min) maximize goal;