
include("../src/VMC.jl")

# This script trains a TransformerNet model by using the output matrices from a pre-trained SlaterNet as training data.
# The goal is to see if the TransformerNet can learn to replicate the function of the SlaterNet.

let
    # --- Parameters ---
    # SlaterNet parameters (should match the pre-trained model)
    slater_emb_size = 10
    Nx = 6
    Ny = 6
    Nelec = 6
    slater_model_filename = "./data/test10.bson" # Assumes pre-trained model from pretrain2.jl

    # TransformerNet parameters
    num_att_block = 3
    num_heads = 4
    num_slaters = 1
    transformer_embsize = 64

    # Training parameters
    lr = 0.001
    nsample = 2000
    epochs = 500

    # --- 1. Load Pre-trained SlaterNet ---
    println("Attempting to load pre-trained SlaterNet...")
    slater_net = f32(SlaterNet(; emb_size=slater_emb_size, Nx=Nx, Ny=Ny, Nelec=Nelec))
    if isfile(slater_model_filename)
        # The variable name in the .bson file is often 'ψ' in other scripts
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
    transformer_model_filename = "./data/transformernet_from_slaternet_Nx_$(Nx)_Ny_$(Ny)_Nelec_$(Nelec).bson"

    if isfile(transformer_model_filename)
        println("Loading existing TransformerNet model from $transformer_model_filename")
        @load transformer_model_filename ψ_trans
        transformer_net = ψ_trans
    else
        println("No saved TransformerNet model found, starting from scratch.")
    end

    # --- 4. Train the TransformerNet ---
    loss_fn(model, X, Y) = begin
        y_pred = model(X)
        y_pred_reshaped = reshape(y_pred, Nelec, Nelec, size(X, 2))
        # Use Mean Squared Error between the predicted and target matrices
        Flux.Losses.mse(y_pred_reshaped, Y)
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
        transformer_output_matrix = reshape(transformer_net(sample_x), Nelec, Nelec)

        # Calculate determinants
        slater_det = det(slater_output_matrix)
        transformer_det = det(transformer_output_matrix)

        println("Sample ", i)
        println("  SlaterNet determinant:      ", slater_det)
        println("  TransformerNet determinant: ", transformer_det)
        println("  Matrix MSE:                 ", Flux.Losses.mse(transformer_output_matrix, slater_output_matrix))
        println("-------------------------------------------------")
    end
end
