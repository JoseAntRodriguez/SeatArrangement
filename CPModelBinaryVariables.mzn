int:k;
set of int:n = 0..(k-1);
array[n,n] of int: V;
array[n,n] of int: G;
int: uType; % 1 if S-utility, 2 if B-utility, 3 if W-utility
int: objective; % 1 if MWA, 2 if MUA, 3 if EFA, 4 if STA
int: minVal = min(p in n, q in n)(V[p,q]);
int: maxVal = max(p in n, q in n)(V[p,q]);

function array[int] of int: neighbours(int: u) = [v|v in n where G[u,v] == 1];
function int: degree(int: u) = sum(v in n)(G[u,v]);

array[n,n] of var 0..1:x;
constraint forall(p in n)(sum(u in n)(x[p,u]) = 1);
constraint forall(u in n)(sum(p in n)(x[p,u]) = 1);

array[n,n,n,n] of var 0..1:y;
constraint forall(p in n, q in n, u in n, v in n) (y[p,q,u,v] == if x[p,u] + x[q,v] == 2 then G[u,v] else 0 endif);

% We need the utilityAtVertex function to properly take into account isolated vertices; otherwise we take max and min over empty sets, which causes bugs
function var int: utilityAtVertex(int: p, int: u, int: uT, array[n,n,n,n] of var 0..1:z) = if degree(u) == 0 then 0 elseif uT == 2 then max(v in neighbours(u))(sum(q in n)(z[p,q,u,v]*V[p,q])) elseif uT == 3 then min(v in neighbours(u))(sum(q in n)(z[p,q,u,v]*V[p,q])) else sum(q in n, v in n)(z[p,q,u,v]*V[p,q]) endif;

function var int: utility(int: p, int: uT, array[n,n,n,n] of var 0..1:z) = sum(u in n)(utilityAtVertex(p,u,uT,z));

function var int: exchangeVariable(int: p, int: q, int: r, int: s, int: w, int: w2) =
if r == p \/ s == p then if r == p then if s == q then y[q,p,w,w2] else y[q,s,w,w2] endif else if r == q then y[p,q,w,w2] else y[r,q,w,w2] endif endif
elseif r == q \/ s == q then if r == q then if s == p then y[p,q,w,w2] else y[p,s,w,w2] endif else if r == p then y[q,p,w,w2] else y[r,p,w,w2] endif endif
else y[r,s,w,w2] endif;

function array[n,n,n,n] of var 0..1: exchangeVariables(int: p, int: q) = array4d(n,n,n,n,[exchangeVariable(p,q,r,s,w,w2)|r in n, s in n, w in n, w2 in n]);

function var 0..1: envy(int: p, int: q) = if utility(p, uType, exchangeVariables(p,q)) > utility(p, uType, y) then 1 else 0 endif;

constraint if objective == 3 then forall(u in n)(forall(v in n)(envy(u,v) = 0)) endif;
constraint if objective == 4 then forall(u in n)(forall(v in (u+1)..(k-1))(envy(u,v) + envy(v,u) <= 1)) endif;

var int: goal = if objective == 1 then sum([utility(p, uType, y) | p in n]) elseif objective == 2 then min([utility(p, uType, y) | p in n]) else 0 endif;
solve ::int_search(x, first_fail, indomain_min) maximize goal;