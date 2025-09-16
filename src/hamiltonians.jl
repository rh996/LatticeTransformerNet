abstract type Hamiltonian2D end

struct HubbardHamiltonian <: Hamiltonian2D
    t::Float64
    U::Float64
    Nx::Int64
    Ny::Int64
end


function ConstructHoppings(HubbardHamiltonian; spin_explicit=false)
    # Construct the 2D hopping matrix for the Hubbard model
    # Sequence: spin_up, spin_dn, spin_up_spin_dn

    Nx = HubbardHamiltonian.Nx
    Ny = HubbardHamiltonian.Ny
    N_sites = Nx * Ny

    # The hopping matrix for a single spin sector
    function hopping_matrix(Nx, Ny, t)
        N = Nx * Ny
        H = zeros(Float64, N, N)
        for x in 1:Nx
            for y in 1:Ny
                i = (x - 1) * Ny + y
                # right neighbor (periodic)
                xp = x == Nx ? 1 : x + 1
                j = (xp - 1) * Ny + y
                H[i, j] -= t
                H[j, i] -= t
                # up neighbor (periodic)
                yp = y == Ny ? 1 : y + 1
                j = (x - 1) * Ny + yp
                H[i, j] -= t
                H[j, i] -= t
            end
        end
        return H
    end

    t = HubbardHamiltonian.t

    H_up = hopping_matrix(Nx, Ny, t)

    if spin_explicit

        return H_up

    else
        # For spin_up_spin_dn, it's just a block diagonal of H_up and H_dn
        H_total = zeros(Float64, 2 * N_sites, 2 * N_sites)
        H_total[1:2:end, 1:2:end] = H_up
        H_total[2:2:end, 2:2:end] = H_up

        return H_total

    end

end


function local_energy(H::HubbardHamiltonian, tmatrix::Matrix, connection_indx::Vector{CartesianIndex{2}}, ψ::Wavefunction, x::Configuration)
    # Compute local energy for configuration x
    V::ComplexF64 = 0.0
    K::ComplexF64 = 0.0
    #compute V(R)

    V = H.U * sum((x.Orbitals[1:2:end] + x.Orbitals[2:2:end]) .> 1)  # Count double occupancy

    #compute K(R)
    logabs, phase = logabsamplitude(ψ, x)
    if phase == 0 || !isfinite(logabs)
        return V
    end
    for id in connection_indx
        if (x.Orbitals[id[2]] == 1) && (x.Orbitals[id[1]] == 0) # id[1] is unoccupied, id[2] is occupied
            new_electrons = deepcopy(x.Electrons)
            new_orbitals = deepcopy(x.Orbitals)
            new_orbitals[id[1]] = 1
            new_orbitals[id[2]] = 0

            for j in eachindex(new_electrons)
                if x.Electrons[j] == id[2]
                    new_electrons[j] = id[1]
                end
            end
            new_x = Configuration(new_orbitals, new_electrons)
            logabs_m, phase_m = logabsamplitude(ψ, new_x)
            if phase_m == 0 || !isfinite(logabs_m)
                continue
            end
            ratio = exp(logabs_m - logabs) * (phase_m / phase)
            K += tmatrix[id] * ratio

        end

    end

    return V + K
end

function local_energy(ψ::Union{SlaterNet,TransformerNet}, H::HubbardHamiltonian, tmatrix::Matrix, connection_indx::Vector{CartesianIndex{2}}, configs::Vector{Configuration})
    E_locs = [
        let
            x = config
            V = H.U * sum((x.Orbitals[1:2:end] + x.Orbitals[2:2:end]) .> 1)

            logabs, phase = logabsamplitude(ψ, x)
            if phase == 0 || !isfinite(logabs)
                V
            else
                K = 0.0
                for id in connection_indx
                    if (x.Orbitals[id[2]] == 1) && (x.Orbitals[id[1]] == 0) # id[1] is unoccupied, id[2] is occupied
                        new_electrons = copy(x.Electrons)
                        for j in eachindex(new_electrons)
                            if x.Electrons[j] == id[2]
                                new_electrons = [new_electrons[1:j-1]..., id[1], new_electrons[j+1:end]...]
                                break
                            end
                        end
                        new_orbitals = zeros(Int64, length(x.Orbitals))
                        for orb in new_electrons
                            new_orbitals[orb] += 1
                        end
                        logabs_m, phase_m = logabsamplitude(ψ, Configuration(new_orbitals, new_electrons))
                        if phase_m == 0 || !isfinite(logabs_m)
                            continue
                        end
                        ratio = exp(logabs_m - logabs) * (phase_m / phase)
                        K += tmatrix[id] * ratio
                    end
                end
                V + K
            end
        end for config in configs
    ]
    return E_locs
end

# More Hamiltonians (Hubbard, Bose-Hubbard, etc)
