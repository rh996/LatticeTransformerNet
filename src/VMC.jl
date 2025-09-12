
using Random
using Flux
using LinearAlgebra
using Tullio
using Statistics
using ProgressMeter
using BSON: @save, @load

include("wavefunctions.jl")
include("hamiltonians.jl")
include("sampling.jl")
include("optimizer.jl")
include("observables.jl")
include("utils.jl")



#initialization
function RunVMC(
    Nx::Int,
    Ny::Int,
    Nelec::Int,
    t_hopping::Float64,
    u_interaction::Float64;
    TotalSteps::Int=200000,
    ThermalizationSteps::Int=1000,
    OptimizationSteps::Int=10,
    wf_type::Symbol=:Jastrow,
    seed::Int=1,
    lr::Float64=0.001
)
    SetSeed(seed)
    L = Nx * Ny * 2
    Hamiltonian = HubbardHamiltonian(t_hopping, u_interaction, Nx, Ny)
    t_matrix = ConstructHoppings(Hamiltonian)
    indx = findall(x -> abs(x) > 10^(-6), t_matrix)


    v, mo_coeff = eigen(t_matrix)


    mo_coeff *= exp(58 / 20)

    # Initialize wavefunction based on type
    if wf_type == :SlaterDeterminant
        ψ = SlaterDeterminant(mo_coeff[:, 1:Nelec])
    elseif wf_type == :Gutzwiller
        ψ = Gutzwiller(0.9, mo_coeff[:, 1:Nelec])
    elseif wf_type == :Jastrow
        ψ = Jastrow(ones(L, L), mo_coeff[:, 1:Nelec])
    elseif wf_type == :JastrowLimited
        ψ = JastrowLimited(0.9, 0.0, 0.1, mo_coeff[:, 1:Nelec], GenerateNNind(Nx, Ny))
    else
        error("Unknown wavefunction type: $wf_type")
    end

    if wf_type == :SlaterDeterminant
        OptimizationSteps = 1
    end

    for opt_step in 1:OptimizationSteps

        x_init = initialize_configuration(Nelec, L)
        x = x_init

        e_list = []
        estimator_list = []
        acc_count = 0

        p = Progress(TotalSteps)
        for step in 1:TotalSteps
            next!(p)
            # Perform a single Metropolis step
            x, acc = metropolis_step(ψ, x)
            if step > ThermalizationSteps
                # Collect observables after thermalization
                acc_count += acc
                #compute local energy E(x)
                loc_e = local_energy(Hamiltonian, t_matrix, indx, ψ, x)
                push!(e_list, loc_e)
                push!(estimator_list, LocalEstimator(ψ, x))
            end
        end

        # Calculate and print the average energy
        avg_energy = mean(e_list)
        std_energy = std(e_list)
        println("Average energy after $TotalSteps steps: $avg_energy")
        println("Standard deviation of energy: $std_energy")
        println("Acceptance rate: $(acc_count / (TotalSteps - ThermalizationSteps))")

        # return avg_energy, e_list, acc_count / (TotalSteps - ThermalizationSteps)


        # Optimize the wavefunction parameters
        if wf_type == :Gutzwiller || wf_type == :JastrowLimited
            ψ = OptimizeEnergy!(ψ, real(avg_energy), estimator_list, e_list; lr=lr)
        end

    end
end





function RunVMCNQS(
    Nx::Int,
    Ny::Int,
    Nelec::Int,
    t_hopping::Float64,
    u_interaction::Float64;
    TotalSteps::Int=200000,
    ThermalizationSteps::Int=1000,
    OptimizationSteps::Int=1,
    seed::Int=1,
    lr=1e-3,
    init_params::String="./data/test.bson",
    wf_type::Symbol=:SlaterNet
)
    SetSeed(seed)
    L = Nx * Ny * 2
    Hamiltonian = HubbardHamiltonian(t_hopping, u_interaction, Nx, Ny)
    t_matrix = ConstructHoppings(Hamiltonian)
    indx = findall(x -> abs(x) > 10^(-6), t_matrix)

    #show system infomation
    @info "System Information:"
    @info "Nx: $Nx"
    @info "Ny: $Ny"
    @info "Nelec: $Nelec"
    @info "t_hopping: $t_hopping"
    @info "u_interaction: $u_interaction"

    #initialize the model
    if wf_type == :SlaterNet
        ψ = Flux.f32(SlaterNet(; emb_size=24, Nx=Nx, Ny=Ny, Nelec=Nelec))
        if isfile(init_params)
            @info "Loading pre-trained model parameters from $init_params"
            # ml = nothing
            @load init_params ψ
            # ψ = ml
        else
            @info "Initializing model parameters randomly"
        end
    elseif wf_type == :TransformerNet
        ψ = Flux.f32(TransformerNet(; num_att_block=3, num_heads=4, num_slaters=2, embsize=24, Nx=Nx, Ny=Ny, Nelec=Nelec))
        if isfile(init_params)
            @info "Loading pre-trained model parameters from $init_params"
            # ψ_transform = nothing
            @load init_params ψ
        else
            @info "Initializing model parameters randomly"
        end
        # ψ = ψ_transform
    end

    opt = Flux.setup(Adam(lr), ψ)
    @info "Training model with $wf_type wavefunction"
    for opt_step in 1:OptimizationSteps

        x_init = initialize_configuration(Nelec, L)
        x = x_init

        acc_configs = Configuration[]
        acc_count = 0

        # p = Progress(TotalSteps)
        for step in 1:TotalSteps
            # next!(p)
            # Perform a single Metropolis step
            x, acc = metropolis_step(ψ, x)
            if step > ThermalizationSteps
                # Collect observables after thermalization
                acc_count += acc
                push!(acc_configs, x)
            end
        end

        # Calculate and print the average energy
        #thinning the data every 10 steps
        loc_e_list = local_energy(ψ, Hamiltonian, t_matrix, indx, acc_configs[1:5:end])
        avg_energy = mean(loc_e_list)
        std_energy = std(loc_e_list)

        println("Optimized step: $opt_step")
        println("Average energy after $TotalSteps steps: $avg_energy")
        println("Standard deviation of energy: $std_energy")
        println("Acceptance rate: $(acc_count / (TotalSteps - ThermalizationSteps))")

        #optimize
        estimator_ls, re = LocalEstimator(ψ, acc_configs[1:5:end])
        ψ = OptimizeEnergy!(ψ, avg_energy, estimator_ls, loc_e_list, re, opt)

    end
    @save init_params ψ

end



function RunVMCNQS(
    Nx::Int,
    Ny::Int,
    Nup::Int,
    Ndn::Int,
    t_hopping::Float64,
    u_interaction::Float64;
    TotalSteps::Int=200000,
    ThermalizationSteps::Int=1000,
    OptimizationSteps::Int=1,
    seed::Int=1,
    lr=1e-3,
    init_params::String="./data/test.bson",
    wf_type::Symbol=:SlaterNet
)
    SetSeed(seed)
    L = Nx * Ny
    Hamiltonian = HubbardHamiltonian(t_hopping, u_interaction, Nx, Ny)
    t_matrix = ConstructHoppings(Hamiltonian; spin_explicit=false)
    indx = findall(x -> abs(x) > 10^(-6), t_matrix)

    #show system infomation
    @info "System Information:"
    @info "Nx: $Nx"
    @info "Ny: $Ny"
    @info "seperate spin walkers"
    @info "Nup Ndn: $Nup $Ndn"
    @info "t_hopping: $t_hopping"
    @info "u_interaction: $u_interaction"

    #initialize the model
    if wf_type == :SlaterNet
        ψ = Flux.f32(SlaterNet(; emb_size=24, Nx=Nx, Ny=Ny, Nelec=(Nup + Ndn)))
        if isfile(init_params)
            @info "Loading pre-trained model parameters from $init_params"
            # ml = nothing
            @load init_params ψ
            # ψ = ml
        else
            @info "Initializing model parameters randomly"
        end
    elseif wf_type == :TransformerNet
        ψ = Flux.f32(TransformerNet(; num_att_block=3, num_heads=4, num_slaters=2, embsize=24, Nx=Nx, Ny=Ny, Nelec=(Nup + Ndn)))
        if isfile(init_params)
            @info "Loading pre-trained model parameters from $init_params"
            # ψ_transform = nothing
            @load init_params ψ
        else
            @info "Initializing model parameters randomly"
        end
        # ψ = ψ_transform
    end

    opt = Flux.setup(AdamW(lr), ψ)
    @info "Training model with $wf_type wavefunction"

    for opt_step in 1:OptimizationSteps

        x_init = initialize_configuration(Nup, Ndn, L)
        x = x_init

        acc_configs = Configuration[]
        acc_count = 0

        # p = Progress(TotalSteps)
        for step in 1:TotalSteps
            # next!(p)
            # Perform a single Metropolis step
            x, acc = metropolis_step(ψ, x; Nup=Nup, Ndn=Ndn)
            if step > ThermalizationSteps
                # Collect observables after thermalization
                acc_count += acc
                push!(acc_configs, x)
            end
        end

        # Calculate and print the average energy
        #thinning the data every 10 steps
        loc_e_list = local_energy(ψ, Hamiltonian, t_matrix, indx, acc_configs[1:5:end])
        avg_energy = mean(loc_e_list)
        std_energy = std(loc_e_list)

        println("Optimized step: $opt_step")
        println("Average energy after $TotalSteps steps: $avg_energy")
        println("Standard deviation of energy: $std_energy")
        println("Acceptance rate: $(acc_count / (TotalSteps - ThermalizationSteps))")

        #optimize
        estimator_ls, re = LocalEstimator(ψ, acc_configs[1:5:end])
        ψ = OptimizeEnergy!(ψ, avg_energy, estimator_ls, loc_e_list, re, opt)

    end
    @save init_params ψ



end
