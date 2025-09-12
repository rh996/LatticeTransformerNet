include("../src/VMC.jl")


let
    Nelec = 6
    t_hopping = 1.0
    u_interaction = 4.0
    Nx = 6
    Ny = 6
    L = Nx * Ny * 2
    Hamiltonian = HubbardHamiltonian(t_hopping, u_interaction, Nx, Ny)
    t_matrix = ConstructHoppings(Hamiltonian)
    v, mo_coeff = eigen(t_matrix)

    lr = 0.01
    ψ = SlaterDeterminant(mo_coeff[:, 1:Nelec])
    #prepare training data


    samples = 2000
    xs = Array{Int32}(undef, Nelec, samples)
    ys = Array{Float32}(undef, samples)
    for i in 1:samples
        orbitals = collect(Int32, 1:L)
        selected_orbital = shuffle(orbitals)[1:Nelec]
        xs[:, i] = Int32.(selected_orbital)
        ys[i] = det(10 * real.(ψ.Mocoeff[selected_orbital, :]))
    end

    @show mean(ys)
    # @show ys

    ψ = f32(SlaterNet(; emb_size=5, Nx=Nx, Ny=Ny, Nelec=Nelec))


    model_filename = "./data/test5.bson"

    if isfile(model_filename)
        println("Loading model from $model_filename")
        @load model_filename ψ
    else
        println("No saved model found, starting from scratch.")
    end
    function loss_fn(m, X, Y)
        y = m(X)
        y = reshape(y, Nelec, Nelec, samples)
        logdet1 = Array{Float32}([det(y[:, :, i]) for i in 1:samples])
        # logdet1 = Array{Float32}([logabsdet(y[:, :, i])[1] for i in 1:samples])
        return Flux.Losses.mse(logdet1, Y)
    end

    @show loss_fn(ψ, xs, ys)
    opt_state = Flux.setup(Adam(lr), ψ)


    println("Continuing training for 100 steps...")
    for epoch in 1:500
        loss_val, grads = Flux.withgradient(ψ) do m
            loss_fn(m, xs, ys)
        end
        Flux.update!(opt_state, ψ, grads[1])
        epoch % 10 == 0 && @info "epoch $epoch  loss=$(loss_val)"
    end


    #compare the model with samples
    println("--- Testing the model with 5 OLD samples from the training set ---")
    for i in 1:5
        selected_orbitals = xs[:, i]
        y_exact = (ys[i])
        y_model = det(ψ(selected_orbitals))
        @show logabsdet(y_exact)
        @show logabsdet(y_model)
        # @show size(y_exact)
        # @show size(y_model)
        println("Sample ", i)
        println("Exact value: ", y_exact)
        println("Model prediction: ", y_model)
        println("-------------------")
    end

    # println("--- Testing the model with 5 NEW samples ---")
    # for i in 1:5
    #     orbitals = collect(Int32, 1:L)
    #     selected_orbitals = shuffle(orbitals)[1:Nelec]
    #     y_exact = 100 * real(det(ψ.Mocoeff[selected_orbitals, :]))
    #     y_model = ψ(selected_orbitals)
    #     println("Sample ", i)
    #     println("Exact value: ", y_exact)
    #     println("Model prediction: ", y_model)
    #     println("-------------------")
    # end

    @save model_filename ψ



end
