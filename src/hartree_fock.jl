include("VMC.jl")
using Tullio
using LinearAlgebra

# --- Hartree-Fock Core Functions ---

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
    Fock = complex(t_matrix)
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

"""
    run_scf(t_matrix, u_interaction, Nelec, initial_mo_coeff; ...)

Runs the self-consistent field (SCF) loop for the Hubbard model.
"""
function run_scf(t_matrix, u_interaction, Nelec, initial_mo_coeff;
    mixing_param=0.1, convergence_tol=1e-6, max_iter=5000)

    mo_coeff = deepcopy(initial_mo_coeff)
    last_energy = 0.0
    final_energy = 0.0

    @info "Starting Hartree-Fock SCF..."
    for i in 1:max_iter
        # Run one HF iteration
        mo_coeff_new, energy = run_hf(t_matrix, u_interaction, mo_coeff, Nelec)
        final_energy = energy

        # Mix new and old MOs to aid convergence
        mo_coeff_mixed = mixing_param * mo_coeff_new + (1 - mixing_param) * mo_coeff

        # Re-orthogonalize the mixed coefficients using Löwdin orthogonalization
        overlap_matrix = mo_coeff_mixed' * mo_coeff_mixed
        mo_coeff = mo_coeff_mixed * (overlap_matrix^(-1 / 2))

        @info "Iteration $i, Energy = $energy"

        # Check for convergence
        if abs(energy - last_energy) < convergence_tol
            @info "Hartree-Fock converged in $i iterations."
            return mo_coeff, energy
        end
        last_energy = energy

        if i == max_iter
            @warn "Hartree-Fock did not converge within $max_iter iterations."
        end
    end

    return mo_coeff, final_energy
end


# --- Initial State Generators ---

"""
    generate_random_mo_coeff(L)

Generates a random set of orthonormal molecular orbitals.
"""
function generate_random_mo_coeff(L)
    @info "Constructing a random initial state..."
    random_matrix = rand(ComplexF64, L, L)
    hermitian_matrix = random_matrix + random_matrix'
    _, mo_coeff = eigen(hermitian_matrix)
    return mo_coeff
end

"""
    generate_spin_polarized_mo_coeff(t_matrix; h_field=1e-4)

Generates MO coefficients for a spin-polarized initial state by applying a
symmetry-breaking magnetic field.
"""
function generate_spin_polarized_mo_coeff(t_matrix; h_field=1e-4)
    @info "Constructing a spin-polarized initial state via symmetry breaking..."
    L = size(t_matrix, 1)
    t_perturbed = copy(t_matrix)

    # Lower the energy of spin-up orbitals and raise spin-down
    for i in 1:L
        if isodd(i)
            t_perturbed[i, i] -= h_field
        else
            t_perturbed[i, i] += h_field
        end
    end

    _, mo_coeff = eigen(Hermitian(t_perturbed))
    return mo_coeff
end

"""
    generate_cdw_mo_coeff(t_matrix, Nx, Ny; V_staggered=0.1)

Generates MO coefficients for a charge density wave (CDW) initial state
by applying a staggered potential.
"""
function generate_cdw_mo_coeff(t_matrix, Nx, Ny; V_staggered=0.1)
    @info "Constructing a CDW initial state..."
    L = size(t_matrix, 1)
    L_sites = L ÷ 2
    @assert L_sites == Nx * Ny "Mismatch between lattice size and hopping matrix dimension."

    t_perturbed = copy(t_matrix)

    for i in 1:L_sites
        # Map site index `i` to 2D coordinates (ix, iy)
        # Assuming row-major order for sites: site `i` is at (iy, ix)
        ix = (i - 1) % Nx
        iy = ((i - 1) ÷ Nx) % Ny

        potential = V_staggered * (-1)^(ix + iy)

        # Get spin-up and spin-down indices for site i
        idx_up = (i - 1) * 2 + 1
        idx_down = (i - 1) * 2 + 2

        # Add potential to both spin channels
        t_perturbed[idx_up, idx_up] += potential
        t_perturbed[idx_down, idx_down] += potential
    end

    _, mo_coeff = eigen(Hermitian(t_perturbed))
    return mo_coeff
end


# --- Main Execution Block ---
let
    # --- Parameters ---
    Nx = 6
    Ny = 6
    Nelec = 20
    t_hopping = 1.0
    u_interaction = 5.0

    # --- Setup ---
    L = Nx * Ny * 2
    Hamiltonian = HubbardHamiltonian(t_hopping, u_interaction, Nx, Ny)
    t_matrix = ConstructHoppings(Hamiltonian)

    # --- Initial State Generation ---
    # Choose one of the following methods to generate the initial MO coefficients.
    # Uncomment the desired method.

    # 1. Random initial state
    initial_mo_coeff = generate_random_mo_coeff(L)

    # 2. Spin-polarized initial state
    # initial_mo_coeff = generate_spin_polarized_mo_coeff(t_matrix)

    # 3. Charge Density Wave (CDW) initial state
    # initial_mo_coeff = generate_cdw_mo_coeff(t_matrix, Nx, Ny)

    # --- Run SCF ---
    final_mo_coeff, final_energy = run_scf(t_matrix, u_interaction, Nelec, initial_mo_coeff)

    @info "Final Hartree-Fock Energy: $final_energy"

    # display(final_mo_coeff[:, 1:Nelec])
end
