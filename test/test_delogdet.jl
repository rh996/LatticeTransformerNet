using LinearAlgebra
using Flux.ChainRulesCore
using Flux.Zygote

"""
    logdet_stable(A) -> ComplexF64

Returns complex log(det(A)) robustly. If A is (numerically) singular,
returns -Inf + 0im (you can handle this upstream by rejecting the move).
"""
function logdet_stable(A::AbstractMatrix)
    F = lu(A; check=false)              # partial pivoting, no exception
    d = diag(F.U)
    # detect exact/tiny pivots → treat as node
    atol = eps(real(one(eltype(d)))) * size(A, 1) * 10
    if any(x -> iszero(x) || abs(x) ≤ atol, d)
        # return complex(-Inf, 0)
        return -Inf
    end
    # log det = log(det(P)) + sum(log(diag(U)))
    # det(P) = (-1)^(#swaps): contribute 0 or iπ to the complex log
    # swaps = 0
    # vis = falses(length(F.p))
    # for i in eachindex(F.p)
    #     if !vis[i]
    #         j = i
    #         cyc = 0
    #         while !vis[j]
    #             vis[j] = true
    #             j = F.p[j]
    #             cyc += 1
    #         end
    #         swaps += cyc - 1
    #     end
    # end
    logdetU = sum(log, abs.(d))               # complex log of pivots
    # logdetP = isodd(swaps) ? im * pi : 0  # log(-1)=iπ, log(+1)=0
    return logdetU
end

# ---- Custom rrule so AD uses Tr(A^{-1} dA) instead of backprop through LU ----
function ChainRulesCore.rrule(::typeof(logdet_stable), A::AbstractMatrix)
    y = logdet_stable(A)

    function pullback(Δy)
        # If y = -Inf (singular), do not backprop (treat as zero)
        if !isfinite(real(y))
            return (NoTangent(), ZeroTangent())
        end
        # We need Δy * A^{-H}. Compute A^{-H} stably via solves (no explicit inv).
        # Solve A' X = I  ⇒  X = (A') \ I  = A^{-H}
        n = size(A, 1)
        Iₙ = Matrix{eltype(A)}(I, n, n)
        F = lu(A; check=false)
        A_inv = F \ Iₙ           # uses LU with pivoting under the hood
        A_Hinv = A_inv'          # A^{-H} = (A^{-1})'

        # For complex-valued logdet: ∂L/∂A = Δy * A^{-H}
        # If your loss uses log|det A| (real), you likely want real part only:
        dA = real.(Δy) .* real.(A_Hinv)
        return (NoTangent(), dA)
    end

    return y, pullback
end


using Random

# Fake "network" that outputs a Slater matrix M(θ; R)
struct ToyNet
    W::Matrix{Float64}
end
(net::ToyNet)(R) = R * net.W                 # e.g., R: (N, d), W: (d, N)

# Simple loss: minimize negative real logdet (as a placeholder)
function loss(net::ToyNet, R)
    M = net(R)                               # build Slater matrix
    y = logdet_stable(M)                     # log-det
    return -real(y)                          # example loss
end

function loss2(net::ToyNet, R)
    M = net(R)
    y, _ = logabsdet(M)
    return -y
end

# Tiny demo
Random.seed!(2)
N, d = 8, 8
R = randn(N, d)
net = ToyNet(randn(d, N))

grads1 = Zygote.gradient(n -> loss(n, R), net)
@show grads1[1]
@show grads = Zygote.gradient(n -> loss2(n, R), net)
# grads.W is well-defined and uses A^{-H} via solves, not backprop through LU
