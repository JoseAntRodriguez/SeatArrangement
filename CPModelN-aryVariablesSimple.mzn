include "alldifferent.mzn"; 
include "valuations9Binary.dzn";
include "seatGraph9Cycle2.dzn";

set of int:n = 0..(k-1);

array[n] of var n:p;%permutation
constraint alldifferent(p);

function var int: utility(int: i) = sum([G[p[i],p[j]]*V[i,j] | j in n]);

solve ::int_search(p, first_fail, indomain_min) maximize sum([utility(i) | i in n]);