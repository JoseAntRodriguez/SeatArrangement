include "alldifferent.mzn"; 
int:k;
set of int:n = 0..(k-1);
array[n,n] of int: V;
array[n,n] of int: G;
int: uType = 1; % 1 if S-utility, 2 if B-utility, 3 if W-utility
int: objective = 1; % 1 if MWA, 2 if MUA, 3 if EFA, 4 if STA
int: minVal = min(i in n, j in n)(V[i,j]);
int: maxVal = max(i in n, j in n)(V[i,j]);
array[n] of var 0..1:degree;
constraint forall(u in n)(if sum(v in n)(G[u,v]) == 0 then degree[u] = 0 else degree[u] = 1 endif);

function array[int] of int: neighbours(int: u) = [v|v in n where G[u,v] == 1];

array[n] of var n:p;%permutation
constraint alldifferent(p);

function var int: utility(int: i, int: u) = if u == 2 then degree[p[i]]*max([G[p[i],p[j]]*V[i,j] + (1-G[p[i],p[j]])*minVal | j in n]) elseif u == 3 then degree[p[i]]*min([G[p[i],p[j]]*V[i,j] + (1-G[p[i],p[j]])*maxVal | j in n]) else sum([G[p[i],p[j]]*V[i,j] | j in n]) endif;

function var int: utility2(int: i, int: u) = if u == 2 then degree[p[i]]*max([G[p[i],p[j]]*V[i,j] + (1-G[p[i],p[j]])*minVal | j in n]) elseif u == 3 then degree[p[i]]*min([G[p[i],p[j]]*V[i,j] + (1-G[p[i],p[j]])*maxVal | j in n]) else sum([G[p[i],p[j]]*V[i,j] | j in n]) endif;


%solve maximize sum([sum([G[p[i],p[j]]*V[i,j] | j in n]) | i in n]);
var int: goal = if objective == 1 then sum([utility(i, uType) | i in n]) else min([utility(i, uType) | i in n]) endif;
solve ::int_search(p, first_fail, indomain_min) maximize goal;