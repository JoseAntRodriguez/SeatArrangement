int:k;
set of int:n = 0..(k-1);
array[n,n] of int: V;
array[n,n] of int: G;

array[n,n] of var 0..1:x;
constraint forall(p in n)(sum(u in n)(x[p,u]) = 1);
constraint forall(u in n)(sum(p in n)(x[p,u]) = 1);

function var int: utilityAtVertex(int: p, int: u) = sum(q in n, v in n)(x[q,v]*V[p,q]*G[u,v]);

function var int: utility(int: p) = sum(u in n)(utilityAtVertex(p,u)*x[p,u]);

solve ::int_search(x, first_fail, indomain_min) maximize sum([utility(p) | p in n]);
output "";