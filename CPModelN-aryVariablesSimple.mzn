include "alldifferent.mzn";

int:k;
set of int:n = 0..(k-1);
array[n,n] of int: V;
array[n,n] of int: G;

array[n] of var n:p;%permutation
constraint alldifferent(p);

function var int: utility(int: i) = sum([G[p[i],p[j]]*V[i,j] | j in n]);

solve ::int_search(p, first_fail, indomain_min) maximize sum([utility(i) | i in n]);