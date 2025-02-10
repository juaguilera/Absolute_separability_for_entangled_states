import numpy as np
import math
from toqito.channels import partial_transpose
from toqito.perms import symmetric_projection

d=2
N=2

total_dimension = d**N #Dimensió de l'espai de Hilbert Global
dimensions = [d]*N #Dimensions dels espais de Hilbert locals

biggest_bipartition = int(math.floor(N/2)) #Respecte quants qudits es fa la bipartició més gran

symmetric_dimension = math.comb(d+N-1,N) #Dimensió del subespai simètric

IS = symmetric_projection(d,N) #Identitat del Subespai Simètric

symmetric_transpose_dimension = math.comb(d+biggest_bipartition-1,biggest_bipartition)*math.comb(d+N-biggest_bipartition-1,N-biggest_bipartition) 
#La transposada parcial de quelcom que viu en S(C^d * otimes N) en una partició de k vs N-k, viu realment en l'espai S(C^d * otimes k) * S(C^d * otimes N-k)
#La transposta en la base computacional tindrà diversos eigenvalues que seran 0, però no ens interessen!

dim = [d*biggest_bipartition,d*(N-biggest_bipartition)] #Dimensions dels espais de Hilbert de la bipartició més gran

IS_T = partial_transpose(IS, 0, dim) #Transposada Parcial de la identitat del subespai simètric respecte la partició més gran

min_eval_esperat = 1/math.comb(N,biggest_bipartition) #Mínim valor propi que trobem nosaltres

min_eval = np.linalg.eigvalsh(IS_T)[-symmetric_transpose_dimension] #Mínim valor propi que troba numpy en el subespai de S(C^d * \otimes k) * S(C^d * \otimes N-k)

print('Numèric', min_eval, '\n','Esperat', min_eval_esperat) #No hauria de dependre de d.