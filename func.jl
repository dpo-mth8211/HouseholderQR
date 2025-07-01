using BenchmarkTools
using LinearAlgebra

function myqr_simple!(A)
    m, n = size(A)
    τ = zeros(eltype(A), n)
    for k = 1:n
        x = view(A, k:m, k)

        σ = norm(x)
        if A[k,k] ≥ 0
            σ = -σ
        end
        v = copy(x)
        v[1] -= σ
        β = σ * (σ - x[1])
        
        if β ≠ 0
            τ[k] = v[1] / σ
            x[1] = σ
            x[2:end] .= 0

            for j = k+1:n
                col = view(A, k:m, j)
                γ = dot(v, col) / β
                col .-= γ .* v
            end
        else
            τ[k] = 0
        end
    end
    return A, τ
end

function Qprod_simple!(A, τ, B)
    m, n = size(A)
    for k = n:-1:1  
        if τ[k] ≠ 0

            v = zeros(eltype(A), m-k+1)
            v[1] = 1
            v[2:end] = view(A, k+1:m, k)

            for j = 1:size(B, 2)
                col = view(B, k:m, j)
                γ = (τ[k] * v[1] + dot(v[2:end], col[2:end])) / (1 + τ[k])
                col .-= γ .* v
            end
        end
    end
    return B
end

function householder_compact!(A)
    m, n = size(A)
    τ = zeros(eltype(A), n)
    for k = 1:min(m, n)
        x = view(A, k:m, k)
        σ = norm(x)

        if σ == 0
            τ[k] = 0.0
            continue
        end
        α = -sign(x[1]) * σ
        v1 = x[1] - α
        x ./= v1      
        x[1] = 1.0     
        τ[k] = 2 / dot(x, x)
        for j = k+1:n
            col = view(A, k:m, j)
            γ = τ[k] * dot(x, col)
            col .-= γ .* x
        end
        A[k, k] = α
    end

    return A, τ
end

function apply_Q_compact!(A, τ, B)
    m, n = size(A)
    for k = min(m, n):-1:1
        v = view(A, k:m, k)
        v_tmp = copy(v)
        v_tmp[1] = 1.0

        if τ[k] != 0
            for j in axes(B, 2)
                col = view(B, k:m, j)
                γ = τ[k] * dot(v_tmp, col)
                col .-= γ .* v_tmp
            end
        end
    end
    return B
end



function apply_Qt_compact!(A, τ, B)
    m, n = size(A)
    for k = 1:min(m, n)
        v = view(A, k:m, k)
        v_tmp = copy(v)
        v_tmp[1] = 1.0
        if τ[k] != 0
            for j in axes(B, 2)
                col = view(B, k:m, j)
                γ = τ[k] * dot(conj(v_tmp), col) 
                col .-= γ .* v_tmp
            end
        end
    end
    return B
end


function build_Q(A, τ)
    m, n = size(A)
    Q = Matrix{eltype(A)}(I, m, m)
    apply_Q_compact!(A, τ, Q)
    return Q
end

function build_R(A)
    m, n = size(A)
    R = zeros(eltype(A), m, n)
    for j = 1:n
        for i = 1:min(j,m)
            R[i,j] = A[i,j]
        end
    end
    return R
end