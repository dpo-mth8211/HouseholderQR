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

function check_Q(A, Q)
  m, n = size(A)
  T = eltype(A)
  Qjulia = hcat([Q * onehot(T, m, k) for k = 1:m]...)
  err_Q = norm(hcat([Qprod_simple!(A, onehot(T, m, k)) for k = 1:m]...) - Qjulia)
  println("erreur sur Q : ", err_Q)
end

function check_QR(A)
  m, n = size(A)
  Q, R = qr(A)
  B = copy(A)
  myqr_opti!(B)
  check_Q(B, Q)
  err_R = norm(UpperTriangular(B[1:n, 1:n] - R)) / norm(UpperTriangular(R))
  println("erreur sur R : ", err_R)
end

function check_Qconj(A, Q)
  m, n = size(A)
  T = eltype(A)
  Qjulia = hcat([Q' * onehot(T, m, k) for k = 1:m]...)
  err_Q = norm(hcat([Qprod_conjugue!(A, onehot(T, m, k)) for k = 1:m]...) - Qjulia)
  println("erreur sur Q : ", err_Q)
end

m, n = 10, 4
A = rand(10, 4)
Q, R = qr(A)
myqr_opti!(A)

check_QR(A)
check_Qconj(A, Q)
