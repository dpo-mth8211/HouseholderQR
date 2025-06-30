using LinearAlgebra

function my_sign(x::Number)
    if x == zero(x)
      return one(x)
    else
      return sign(x)
    end
end

"""
    myqr_simple(A)

Écrase `A` par le résultat de la factorisation QR compacte de Householder.
"""
function myqr_simple!(A)
  m, n = size(A)
  @assert m ≥ n
  for j = 1:n
    vj = A[j:m,j]
    σj = my_sign(vj[1])
    vj_norm = norm(vj)
    vj[1] += σj * vj_norm
    vj ./= vj[1]
    δj = vj'vj

    A[j:m,j:n] -= 2 * vj * (vj' * A[j:m,j:n]) / δj
    A[j+1:m,j] = vj[2:end]
  end
  A
end 

"""
    Qprod_simple!(A, x)

Écrase `x` par le résultat du produit Q * x, où Q est le facteur unitaire de la factorisation QR compacte de Householder.
On suppose que `A` contient déjà le résultat de cette factorisation QR compacte.
"""
function Qprod_simple!(A, x)
  m, n = size(A)
  for j = n:-1:1
    uj = [1 ; A[j+1:m, j]]
    δj = uj'uj
    x[j:m] -= 2*uj*(uj'x[j:m])/δj
  end
  x
end
