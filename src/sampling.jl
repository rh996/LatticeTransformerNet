function metropolis_step(ψ::Wavefunction, x::Configuration; Nup=nothing, Ndn=nothing)
    # Single Metropolis-Hastings update
    #
    # propose a move

    L = length(x.Orbitals)



    new_x = LocalUpdate(x)


    # if Nup !== nothing && Ndn !== nothing
    #     if rand() < 0.5
    #         new_x = initialize_configuration(Nup, Ndn, div(L, 2))

    #     else
    #         new_x = SpinflipUpdate(x)
    #     end
    # else
    #     Nelect = length(x.Electrons)
    #     new_x = initialize_configuration(Nelect, L)
    # end

    old_amp, _ = logabsdet(ψ(x.Electrons))
    new_amp, _ = logabsdet(ψ(new_x.Electrons))

    #accept rate
    acc = 0
    ratio_square = exp(2 * (new_amp - old_amp))

    if rand() < min(1, ratio_square)
        # println("Accepted move")
        acc = 1
        x = new_x
    end




    return x, acc
end


function LocalUpdate(x::Configuration)
    L = length(x.Orbitals)


    Electrons = deepcopy(x.Electrons)
    Nelect = length(x.Electrons)
    ielect = rand(1:Nelect)
    occupied = Set(x.Electrons)
    all_orbs = Set(1:L)
    unoccupied = collect(setdiff(all_orbs, occupied))
    if !isempty(unoccupied)
        new_orb = rand(unoccupied)
        Electrons[ielect] = new_orb
    end
    Orbitals = zeros(Int64, L)
    for i in Electrons
        Orbitals[i] += 1
    end
    return Configuration(Orbitals, Electrons)
end


function SpinflipUpdate(x::Configuration)
    Electrons = deepcopy(x.Electrons)
    Electrons .= Electrons .+ (mod.(x.Electrons, 2) .== 1) .- (mod.(x.Electrons, 2) .== 0)
    Orbitals = deepcopy(x.Orbitals)
    Orbitals[1:2:end] .= x.Orbitals[2:2:end]
    Orbitals[2:2:end] .= x.Orbitals[1:2:end]

    return Configuration(Orbitals, Electrons)
end


function SpinConserveLocalUpdate(x::Configuration, Nup, Ndn)

end


function CombinationUpdate(x::Configuration, Nup, Ndn)

end
