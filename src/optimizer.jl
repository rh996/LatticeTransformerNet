function LocalEstimator(ψ::SlaterDeterminant, x::Configuration)

end


function LocalEstimator(ψ::Gutzwiller, x::Configuration)
    nd = sum((x.Orbitals[1:2:end] + x.Orbitals[2:2:end]) .> 1)
    return -nd
end

function OptimizeEnergy!(ψ::Gutzwiller, EnergyEx::Float64, LocalEstimatorList::Vector, LocalEList::Vector; lr=0.01)
    # Simple gradient descent or stochastic reconfiguration

    Avgradient = 2 * mean((conj.(LocalEList) .- EnergyEx) .* LocalEstimatorList)
    println("Average gradient: ", Avgradient)
    new_g = ψ.g - lr * real(Avgradient)
    println("New g: ", new_g)
    ψ2 = Gutzwiller(new_g, ψ.Mocoeff)
    return ψ2
end

function LocalEstimator(ψ::JastrowLimited, x::Configuration)
    Nd = sum((x.Orbitals[1:2:end] + x.Orbitals[2:2:end]) .> 1)
    NNupup = 0
    NNupdn = 0
    for i in eachindex(x.Electrons)
        for j in i+1:length(x.Electrons)
            if fld1(x.Electrons[j], 2) ∈ ψ.NNind[fld1(x.Electrons[i], 2)]
                # If the two electrons have the same spin
                if mod(x.Electrons[i], 2) == mod(x.Electrons[j], 2)
                    NNupup += 1
                else
                    NNupdn += 1
                end
            end
        end
    end
    return [-Nd, -NNupup, -NNupdn]

end

function OptimizeEnergy!(ψ::JastrowLimited, EnergyEx::Float64, LocalEstimatorList::Vector, LocalEList::Vector; lr=0.04)
    # Simple gradient descent or stochastic reconfiguration
    LocalEstimatorList = hcat(LocalEstimatorList...)

    Avgradient = zeros(Float64, 3)
    for i in 1:3
        Avgradient[i] = real(2 * mean((conj.(LocalEList) .- EnergyEx) .* LocalEstimatorList[i, :]))
    end

    println("Average gradient: ", Avgradient)
    new_g = ψ.g - lr * Avgradient[1]
    new_Vupup = ψ.Vupup - lr * Avgradient[2]
    new_Vupdn = ψ.Vupdn - lr * Avgradient[3]
    println("New g: ", new_g)
    println("New Vupup: ", new_Vupup)
    println("New Vupdn: ", new_Vupdn)
    ψ2 = JastrowLimited(new_g, new_Vupup, new_Vupdn, ψ.Mocoeff, ψ.NNind)
    return ψ2
end

function LocalEstimator(ψ::Union{SlaterNet,TransformerNet}, xs::Vector{Configuration})
    # compute ∂ log|ψ|
    first_config = xs[1]
    _, grads = Flux.withgradient(ψ) do m
        logabsamplitude(m, first_config)[1]
    end

    first_vec, re = destructure(grads[1])
    local_est_list = Vector{typeof(first_vec)}(undef, length(xs))
    local_est_list[1] = first_vec

    for (idx, x) in enumerate(xs[2:end])
        _, grads_i = Flux.withgradient(ψ) do m
            logabsamplitude(m, x)[1]
        end

        v, _ = destructure(grads_i[1])
        local_est_list[idx + 1] = v
    end

    return local_est_list, re

end

function OptimizeEnergy!(ψ::Union{SlaterNet,TransformerNet}, EnergyEx::Float64, LocalEstimatorList::Vector, LocalEList::Vector, re, opt)
    #TODO: clip the gradient


    T = eltype(LocalEstimatorList[1])
    E_m_E_ex = T.(LocalEList .- EnergyEx)
    # @show size(E_m_E_ex)
    Avegradient = T(2) * mean(E_m_E_ex .* LocalEstimatorList)

    Avegradient = re(Avegradient)

    Flux.update!(opt, ψ, Avegradient)
    ψ
end
