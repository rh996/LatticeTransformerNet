include("VMC.jl")
using LinearAlgebra, Tullio

"""
    run_hf(t_matrix, u, mocoeff0, Nelec)

Performs a single iteration of the Hartree-Fock procedure for the Hubbard model.
"""
function run_hf(t_matrix, u, mocoeff0, Nelec)
    L = size(t_matrix, 1)
    L_sites = L ÷ 2

    # 1. Calculate density matrix in AO basis from previous MOs
    # P = C * D * C', where D is diagonal with 1s for occupied MOs
    dm_mo = zeros(ComplexF64, L, L)
    dm_mo[1:Nelec, 1:Nelec] = I(Nelec)
    P = mocoeff0 * dm_mo * mocoeff0' # P is the density matrix in AO basis

    # 2. Build Fock matrix for Hubbard model directly
    # F_ij = t_ij + δ_ij * U * <n_{i,-σ}>
    Fock = copy(t_matrix)
    for i in 1:L_sites
        idx_up = (i - 1) * 2 + 1
        idx_down = (i - 1) * 2 + 2

        # Add mean-field interaction to diagonal elements
        # F_up,up += U * <n_down>
        Fock[idx_up, idx_up] += u * P[idx_down, idx_down]
        # F_down,down += U * <n_up>
        Fock[idx_down, idx_down] += u * P[idx_up, idx_up]
    end

    # 3. Diagonalize the Fock matrix to get new MO coefficients and energies
    v, mocoeff = eigen(Hermitian(Fock))

    # 4. Calculate the total HF energy
    # E = 1/2 * Tr((H_core + F) * P)
    energy = 0.5 * real(tr((t_matrix + Fock) * P))

    return mocoeff, energy
end


let
    # --- Parameters ---
    Nx = 4
    Ny = 4
    Nelec = 4
    t_hopping = 1.0
    u_interaction = 5.0
    mixing_param = 0.1  # Linear mixing parameter for MO coefficients
    convergence_tol = 1e-6 # Energy convergence tolerance

    # --- Setup ---
    L = Nx * Ny * 2
    Hamiltonian = HubbardHamiltonian(t_hopping, u_interaction, Nx, Ny)
    t_matrix = ConstructHoppings(Hamiltonian)

    # --- Initial Guess ---
    # To initialize with a fully spin-polarized state, we apply a small
    # symmetry-breaking magnetic field to the core Hamiltonian for the initial guess.
    # This splits the spin-up/down degeneracy, ensuring the lowest `Nelec` orbitals are spin-polarized.
    @info "Constructing a fully spin-polarized initial state via symmetry breaking..."
    t_perturbed = copy(t_matrix)
    h_field = 1e-4 # Small magnetic field strength

    # Lower the energy of spin-up orbitals and raise spin-down
    # This assumes odd indices are spin-up and even are spin-down.
    for i in 1:L
        if isodd(i)
            t_perturbed[i, i] -= h_field
        else
            t_perturbed[i, i] += h_field
        end
    end

    # Diagonalize the perturbed Hamiltonian. The first `Nelec` eigenvectors
    # will now correspond to the lowest-energy (spin-up) orbitals.
    # v, mo_coeff = eigen(Hermitian(t_perturbed))
    v, mo_coeff = eigen(t_matrix)

    # The SCF loop will now naturally start from a polarized state by occupying
    # these first `Nelec` orbitals. The actual SCF iterations use the original, unperturbed t_matrix.

    # --- SCF Loop ---
    last_energy = 0.0
    for i in 1:1
        mo_coeff_old = deepcopy(mo_coeff)

        # Run one HF iteration
        mo_coeff_new, energy = run_hf(t_matrix, u_interaction, mo_coeff_old, Nelec)

        # Mix new and old MOs to aid convergence
        mo_coeff = mixing_param * mo_coeff_new + (1 - mixing_param) * mo_coeff_old

        @info "Iteration $i, Energy = $energy"
        # @info "Idenpotent $()"
        # Check for convergence
        if abs(energy - last_energy) < convergence_tol
            @info "Hartree-Fock converged in $i iterations."
            break
        end
        last_energy = energy
    end
end
