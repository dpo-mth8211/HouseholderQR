using LinearAlgebra

function my_sign(x::Number)
    if x == zero(x)
      return one(x)
    else
      return sign(x)
    end
end

"""
    myqr_simple!(A)
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
onehot(m, k) = onehot(Float64, m, k)

"""
    check_QR(A)
Vérifie la factorisation QR en comparant notre implémentation avec celle de LAPACK.
"""
function check_QR(A)
  m, n = size(A)
  Q, R = qr(A)
  B = copy(A)
  myqr_simple!(B)
  check_Q(B, Q)
  err_R = norm(UpperTriangular(B[1:n, 1:n] - R)) / norm(UpperTriangular(R))
  println("erreur sur R : ", err_R)
end

"""
    check_Q(A, Q)
Vérifie la matrice Q en comparant notre implémentation avec celle de LAPACK.
"""
function check_Q(A, Q)
  m, n = size(A)
  T = eltype(A)
  Qjulia = hcat([Q * onehot(T, m, k) for k = 1:m]...)
  err_Q = norm(hcat([Qprod_simple!(A, onehot(T, m, k)) for k = 1:m]...) - Qjulia)
  println("erreur sur Q : ", err_Q)
end

# Implémentation optimisée (sans allocations)
"""
    myqr_opt!(A)
Version optimisée de la factorisation QR de Householder sans allocations.
"""
function myqr_opt!(A)
  m, n = size(A)
  @assert m ≥ n
  vj = zeros(eltype(A), m)
  
  for j = 1:n
    for i = j:m
      vj[i-j+1] = A[i,j]
    end
    
    σj = my_sign(vj[1])
    vj_norm = norm(view(vj, 1:m-j+1))
    vj[1] += σj * vj_norm
    
    for i = 1:m-j+1
      vj[i] /= vj[1]
    end
    
    δj = 0.0
    for i = 1:m-j+1
      δj += vj[i]^2
    end
    
    for k = j:n
      vjAk = 0.0
      for i = 1:m-j+1
        vjAk += vj[i] * A[i+j-1, k]
      end
      
      vjAk = 2.0 * vjAk / δj
      
      for i = 1:m-j+1
        A[i+j-1, k] -= vj[i] * vjAk
      end
    end
    
    for i = 2:m-j+1
      A[j+i-1, j] = vj[i]
    end
  end
  return A
end

"""
    Qprod_opt!(A, x)
Version optimisée du produit Q * x sans allocations.
"""
function Qprod_opt!(A, x)
  m, n = size(A)
  uj = zeros(eltype(A), m)
  
  for j = n:-1:1
    uj[1] = 1.0
    for i = 2:m-j+1
      uj[i] = A[j+i-1, j]
    end
    
    δj = 0.0
    for i = 1:m-j+1
      δj += uj[i]^2
    end
    
    ujx = 0.0
    for i = 1:m-j+1
      ujx += uj[i] * x[j+i-1]
    end
    
    ujx = 2.0 * ujx / δj
    
    for i = 1:m-j+1
      x[j+i-1] -= uj[i] * ujx
    end
  end
  return x
end

"""
    Qstarprod_opt!(A, x)
Version optimisée du produit Q^* * x sans allocations.
"""
function Qstarprod_opt!(A, x)
  m, n = size(A)
  uj = zeros(eltype(A), m)
  
  for j = 1:n
    uj[1] = 1.0
    for i = 2:m-j+1
      uj[i] = A[j+i-1, j]
    end
    
    δj = 0.0
    for i = 1:m-j+1
      δj += uj[i]^2
    end
    
    ujx = 0.0
    for i = 1:m-j+1
      ujx += uj[i] * x[j+i-1]
    end
    
    ujx = 2.0 * ujx / δj
    
    for i = 1:m-j+1
      x[j+i-1] -= uj[i] * ujx
    end
  end
  return x
end

"""
    check_Q_star(A, Q)
Vérifie le produit Q^* en comparant notre implémentation avec celle de LAPACK.
"""
function check_Q_star(A, Q)
  m, n = size(A)
  T = eltype(A)
  Qstar_julia = hcat([Q' * onehot(T, m, k) for k = 1:m]...)
  err_Qstar = norm(hcat([Qstarprod_opt!(A, onehot(T, m, k)) for k = 1:m]...) - Qstar_julia)
  println("erreur sur Q^* : ", err_Qstar)
end
