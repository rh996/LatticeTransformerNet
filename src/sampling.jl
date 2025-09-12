function metropolis_step(ψ::Wavefunction, x::Configuration; Nup=nothing, Ndn=nothing)
    # Single Metropolis-Hastings update
    #
    # propose a move

    L = length(x.Orbitals)


    # Electrons = deepcopy(x.Electrons)
    # ielect = rand(1:Nelect)
    # occupied = Set(x.Electrons)
    # all_orbs = Set(1:L)
    # unoccupied = collect(setdiff(all_orbs, occupied))
    # if !isempty(unoccupied)
    #     new_orb = rand(unoccupied)
    #     Electrons[ielect] = new_orb
    # end
    # Orbitals = zeros(Int64, L)
    # for i in Electrons
    #     Orbitals[i] += 1
    # end
    # new_x = Configuration(Orbitals, Electrons)
    if Nup !== nothing && Ndn !== nothing
        new_x = initialize_configuration(Nup, Ndn, div(L, 2))
    else
        Nelect = length(x.Electrons)
        new_x = initialize_configuration(Nelect, L)
    end

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
