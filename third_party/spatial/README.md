# Spatial indexes

GPU k-d tree, linear BVH and uniform grid used by ThinGPU, taken from commit
e02d22920cc32e4f279c617770fc5116c84e1989 of the authors' structures library
(Morera Salas and Asch Burgos, CONCAPAN 2026). Only the files ThinGPU needs are included.

The code is unchanged except that comments and messages were translated to English
and five identifiers were renamed (`contrato` → `contract`, `cumple` → `satisfies`,
`cumple_contrato` → `satisfies_contract`, `medir` → `sizer`, `repartir` → `layout`).
