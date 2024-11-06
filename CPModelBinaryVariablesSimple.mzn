int:k;
set of int:n = 0..(k-1);
array[n,n] of int: V;
array[n,n] of int: G;

array[n,n] of var 0..1:x;
constraint forall(p in n)(sum(u in n)(x[p,u]) = 1);
constraint forall(u in n)(sum(p in n)(x[p,u]) = 1);

array[n,n,n,n] of var 0..1:y;
constraint forall(p in n, q in n, u in n, v in n) (y[p,q,u,v] == if x[p,u] + x[q,v] == 2 then G[u,v] else 0 endif);

function var int: utility(int: p) = sum(q in n, u in n, v in n)(y[p,q,u,v]*V[p,q]);

solve ::int_search(x, first_fail, indomain_min) maximize sum([utility(p) | p in n]);
output "";