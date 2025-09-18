
include("../src/VMC.jl")

# This script trains a TransformerNet model by using the output matrices from a pre-trained SlaterNet as training data.
# The goal is to see if the TransformerNet can learn to replicate the function of the SlaterNet.

let
    # --- Parameters ---
    # SlaterNet parameters (should match the pre-trained model)
    slater_emb_size = 24
    Nx = 6
    Ny = 6
    Nelec = 20
    slater_model_filename = "./data/test20.bson" # Assumes pre-trained model from pretrain2.jl

    # TransformerNet parameters
    num_att_block = 3
    num_heads = 4
    num_slaters = 1
    transformer_embsize = 24

    # Training parameters
    lr = 0.01
    nsample = 2000
    epochs = 1000

    # --- 1. Load Pre-trained SlaterNet ---
    println("Attempting to load pre-trained SlaterNet...")
    slater_net = f32(SlaterNet(; emb_size=slater_emb_size, Nx=Nx, Ny=Ny, Nelec=Nelec))
    if isfile(slater_model_filename)
        # The variable name in the .bson file is often 'ψ' in other scripts
        ψ = nothing
        @load slater_model_filename ψ
        slater_net = ψ
        println("Successfully loaded SlaterNet model from $slater_model_filename")
    else
        println("Error: Pre-trained SlaterNet model not found at $slater_model_filename")
        println("Please run pretrain2.jl to generate the required model file first.")
        return # Exit the script
    end

    # --- 2. Generate Training Data from SlaterNet ---
    println("Generating $nsample training samples from SlaterNet...")
    xs = Array{Int32}(undef, Nelec, nsample)
    ys = Array{Float32}(undef, Nelec, Nelec, nsample)
    L = Nx * Ny * 2 # Total number of orbitals

    for i in 1:nsample
        orbitals = collect(Int32, 1:L)
        selected_orbitals = shuffle(orbitals)[1:Nelec]
        xs[:, i] = selected_orbitals
        # Generate the target matrix from the pre-trained SlaterNet
        # The output needs to be reshaped from a vector to a matrix
        ys[:, :, i] = reshape(slater_net(selected_orbitals), Nelec, Nelec)
    end
    println("Training data generated.")

    # --- 3. Define and Load TransformerNet ---
    transformer_net = Flux.f32(TransformerNet(; num_att_block=num_att_block, num_heads=num_heads, num_slaters=num_slaters, embsize=transformer_embsize, Nx=Nx, Ny=Ny, Nelec=Nelec))
    transformer_model_filename = "./data/transformernet_slater_Nx_$(Nx)_Ny_$(Ny)_Nelec_$(Nelec)_att_$(num_att_block)_heads_$(num_heads)_slaters_$(num_slaters)_emb_$(transformer_embsize).bson"

    if isfile(transformer_model_filename)

        println("Loading existing TransformerNet model from $transformer_model_filename")
        data = BSON.load(transformer_model_filename)
        for k in keys(data)
            transformer_net = data[k]
        end

    else
        println("No saved TransformerNet model found, starting from scratch.")
    end

    # --- 4. Train the TransformerNet ---
    loss_fn(model, X, Y) = begin
        y_pred = model(X)
        y_pred_tensor = reshape(y_pred, Nelec, Nelec, num_slaters, size(X, 2))
        # Only supervise the first determinant to match the SlaterNet output
        Flux.Losses.mse(@view(y_pred_tensor[:, :, 1, :]), Y)
    end

    opt_state = Flux.setup(AdamW(lr), transformer_net)

    println("Starting training of TransformerNet for $epochs epochs...")
    for epoch in 1:epochs
        loss, grads = Flux.withgradient(transformer_net) do m
            loss_fn(m, xs, ys)
        end
        Flux.update!(opt_state, transformer_net, grads[1])
        epoch % 20 == 0 && @info "Epoch $epoch  Loss=$(loss)"
    end

    println("Training complete. Saving model...")
    ψ_trans = transformer_net
    @save transformer_model_filename ψ_trans
    println("Model saved to $transformer_model_filename")

    # --- 5. Verification ---
    println("\n--- Testing the trained TransformerNet against the SlaterNet using 5 samples ---")
    for i in 1:5
        sample_x = xs[:, i:i] # Keep batch dimension

        # Get matrices from both models
        slater_output_matrix = reshape(slater_net(sample_x), Nelec, Nelec)
        transformer_output_tensor = reshape(transformer_net(sample_x), Nelec, Nelec, num_slaters)

        println("Sample ", i)
        slater_det = logabsdet(slater_output_matrix)
        println("  SlaterNet determinant:      ", slater_det)
        transformer_det = logabsdet(transformer_output_tensor[:, :, 1])
        mse_val = Flux.Losses.mse(transformer_output_tensor[:, :, 1], slater_output_matrix)
        println("  Transformer determinant 1:  ", transformer_det, "  MSE: ", mse_val)
        for s in 2:num_slaters
            transformer_det = logabsdet(transformer_output_tensor[:, :, s])
            mse_val = Flux.Losses.mse(transformer_output_tensor[:, :, s], slater_output_matrix)
            println("  Transformer determinant $(s): ", transformer_det, "  MSE: ", mse_val)
        end
        println("-------------------------------------------------")
    end
end
