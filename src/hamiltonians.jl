abstract type Hamiltonian2D end

struct HubbardHamiltonian <: Hamiltonian2D
    t::Float64
    U::Float64
    Nx::Int64
    Ny::Int64
end


function ConstructHoppings(HubbardHamiltonian)
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

    # For spin_up_spin_dn, it's just a block diagonal of H_up and H_dn
    H_total = zeros(Float64, 2 * N_sites, 2 * N_sites)
    H_total[1:2:end, 1:2:end] = H_up
    H_total[2:2:end, 2:2:end] = H_up

    return H_total
end


function local_energy(H::HubbardHamiltonian, tmatrix::Matrix, connection_indx::Vector{CartesianIndex{2}}, ψ::Wavefunction, x::Configuration)
    # Compute local energy for configuration x
    V::ComplexF64 = 0.0
    K::ComplexF64 = 0.0
    #compute V(R)

    V = H.U * sum((x.Orbitals[1:2:end] + x.Orbitals[2:2:end]) .> 1)  # Count double occupancy

    #compute K(R)
    amp, s = logabsdet(ψ(x.Electrons))
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
            amp_m, s_m = logabsdet(ψ(new_x.Electrons))
            K += tmatrix[id] * exp(amp_m - amp) * s_m * s

        end

    end

    return V + K
end

function local_energy(ψ::Union{SlaterNet,TransformerNet}, H::HubbardHamiltonian, tmatrix::Matrix, connection_indx::Vector{CartesianIndex{2}}, configs::Vector{Configuration})
    E_locs = [
        let
            x = config
            V = H.U * sum((x.Orbitals[1:2:end] + x.Orbitals[2:2:end]) .> 1)

            amp, s = logabsdet(ψ(x.Electrons))
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
                    amp_m, s_m = logabsdet(ψ(new_electrons))
                    K += tmatrix[id] * exp(amp_m - amp) * s_m * s
                end
            end
            V + K
        end for config in configs
    ]
    return E_locs
end

# More Hamiltonians (Hubbard, Bose-Hubbard, etc)
