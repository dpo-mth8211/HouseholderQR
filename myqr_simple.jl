using LinearAlgebra


function my_sign(x::Number)
    
    if x == zero(x)
      return one(x)
    elseif abs(real(x)) > abs(imag(x))
      return sign(real(x))
    else 
      return sign(imag(x))
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


"""
    onehot(T, m, k)

Construit un vecteur de taille `m` d'éléments de type `T` composé de zéros et d'un 1 à l'indice `k`.
Si `T` est omis, le type Float64 est utilisé.
"""
function onehot(T, m, k)
  @assert 1 ≤ k ≤ m
  x = zeros(T, m)
  x[k] = 1
  x
end

function check_QR(A)
  m, n = size(A)
  Q, R = qr(A)
  B = copy(A)
  myqr_simple!(B)
  check_Q(B, Q)
  err_R = norm(UpperTriangular(B[1:n, 1:n] - R)) / norm(UpperTriangular(R))
  println("erreur sur R : ", err_R)
end

function check_Q(A, Q)
  m, n = size(A)
  T = eltype(A)
  Qjulia = hcat([Q * onehot(T, m, k) for k = 1:m]...)
  err_Q = norm(hcat([Qprod_simple!(A, onehot(T, m, k)) for k = 1:m]...) - Qjulia)
  println("erreur sur Q : ", err_Q)
end

function check_Qprod_simple(A::Matrix{T}, x::Vector{T}) where {T}
  m, n = size(A)
  m, n = size(A)
  Q, R = qr(A)
  myqr!(A)
  y = copy(x)
  err_Q_x = norm(Qprod_simple!(A, y) - Q * x) / norm(x)

  println("erreur sur Q x : ", err_Q_x)
end