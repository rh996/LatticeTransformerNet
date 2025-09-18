function metropolis_step(ψ::Wavefunction, x::Configuration; Nup=nothing, Ndn=nothing)
    # Single Metropolis-Hastings update
    #
    # propose a move

    new_x = if Nup !== nothing && Ndn !== nothing
        CombinationalUpdate(x, Nup, Ndn)
    else
        LocalUpdate(x)
    end

    old_logabs, _ = logabsamplitude(ψ, x)
    new_logabs, _ = logabsamplitude(ψ, new_x)

    #accept rate
    acc = 0
    ratio_square = if !isfinite(new_logabs)
        0.0
    elseif !isfinite(old_logabs)
        1.0
    else
        exp(2 * (new_logabs - old_logabs))
    end

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
    L = length(x.Orbitals)


    Electrons = deepcopy(x.Electrons)
    Nelect = Nup + Ndn
    ielect = rand(1:Nelect)
    occupied = Set(x.Electrons)
    all_orbs = Set(1:L)
    unoccupied = collect(setdiff(all_orbs, occupied))

    spin = mod(Electrons[ielect], 2)
    unoccupied_same_spin = filter(o -> mod(o, 2) == spin, unoccupied)

    if !isempty(unoccupied_same_spin)
        new_orb = rand(unoccupied_same_spin)
        Electrons[ielect] = new_orb
    end
    Orbitals = zeros(Int64, L)
    for i in Electrons
        Orbitals[i] += 1
    end
    return Configuration(Orbitals, Electrons)
end


function CombinationalUpdate(x::Configuration, Nup, Ndn)
    L = div(length(x.Orbitals), 2)
    p = rand()
    if p < 0.7
        return SpinConserveLocalUpdate(x, Nup, Ndn)
    elseif p < 0.8
        return initialize_configuration(Nup, Ndn, L)
    else
        return SpinflipUpdate(x)
    end
end
