include("../src/VMC.jl")

let

    num_att_block = 3
    num_heads = 4
    num_slaters = 1
    embsize = 64
    lr = 0.01
    nsample = 2000

    Nx = 4
    Ny = 4
    Nelec = 2
    t_hopping = 1.0
    u_interaction = 5.0

    L = Nx * Ny * 2
    Hamiltonian = HubbardHamiltonian(t_hopping, u_interaction, Nx, Ny)
    t_matrix = ConstructHoppings(Hamiltonian)
    v, mo_coeff = eigen(t_matrix)

    ψ = SlaterDeterminant(mo_coeff[:, 1:Nelec])

    configurations = [initialize_configuration(Nelec, L) for _ in 1:nsample]
    y = Array{Float64}(undef, Nelec, Nelec, length(configurations))
    for i in eachindex(configurations)
        y[:, :, i] = 10 * ψ.Mocoeff[configurations[i].Electrons, :]
    end

    # @show mean_y = mean(log.(abs.(y)))
    x = Array{Int64}(undef, Nelec, length(configurations))
    for i in eachindex(configurations)
        x[:, i] = configurations[i].Electrons
    end
    # x = hcat(x...)
    @show size(x)

    ψ = Flux.f64(TransformerNet(; num_att_block=num_att_block, num_heads=num_heads, num_slaters=num_slaters, embsize=embsize, Nx=Nx, Ny=Ny, Nelec=Nelec))
    model_filename = "./data/transformernet_slater_Nx_$(Nx)_Ny_$(Ny)_Nelec_$(Nelec)_att_$(num_att_block)_heads_$(num_heads)_slaters_$(num_slaters)_emb_$(embsize).bson"

    if isfile(model_filename)
        println("Loading model from $model_filename")
        @load model_filename ψ
    else
        println("No saved model found, starting from scratch.")
    end



    # @show size(ψ(x))
    println("Continuing training for 300 steps...")
    loss_fn(model, X, Y) = begin
        y_pred = model(X)
        # Assuming model output is (Nelec*Nelec, batch_size)
        y_pred_reshaped = reshape(y_pred, Nelec, Nelec, size(X, 2))
        Flux.Losses.huber_loss(y_pred_reshaped, Y)
        # Flux.Losses.mse(y_pred_reshaped, Y)
    end

    opt_state = Flux.setup(AdamW(lr), ψ)

    for epoch in 1:2000
        loss, grads = Flux.withgradient(ψ) do m
            loss_fn(m, x, y)
        end
        Flux.update!(opt_state, ψ, grads[1])
        epoch % 10 == 0 && @info "epoch $epoch  loss=$(loss)"
    end
    @save model_filename ψ

    println("--- Testing the model with 5 OLD samples from the training set ---")
    for i in 1:5
        y_exact = det(y[:, :, i])
        # The model is trained to predict the scaled value, so y_exact is correct.
        # We use slicing (i:i) to keep the batch dimension for the model.
        sample_x = x[:, i:i]
        y_model_matrix = reshape(ψ(sample_x), Nelec, Nelec)
        y_model = det(y_model_matrix)
        println("Sample ", i)
        println("Exact value: ", y_exact)
        println("Model prediction: ", y_model)
        println("-------------------")
    end

    println("--- Testing the model with 5 NEW samples ---")
    for i in 1:5
        new_config = initialize_configuration(Nelec, L)
        # Apply the same scaling factor for a consistent comparison
        y_exact = det(10 * mo_coeff[new_config.Electrons, 1:Nelec])

        # Reshape the new configuration to have a batch dimension of 1
        sample_x = reshape(new_config.Electrons, :, 1)
        y_model_matrix = reshape(ψ(sample_x), Nelec, Nelec)
        y_model = det(y_model_matrix)
        println("Sample ", i)
        println("Exact value: ", y_exact)
        println("Model prediction: ", y_model)
        println("-------------------")
    end
end
