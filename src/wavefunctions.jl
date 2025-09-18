using Flux.ChainRulesCore
using Flux.Zygote


struct Configuration
    # Configuration of electrons in orbitals
    Orbitals::Vector{Int64}
    Electrons::Vector{Int64}
end

"""
Initialize a configuration
"""
function initialize_configuration(Nelec::Int64, L::Int64)
    # Randomly select Nelec unique orbitals from L boxes
    orbitals = collect(Int64, 1:L)
    selected_orbitals = shuffle(orbitals)[1:Nelec]
    # Electrons[i] = orbital label for electron i
    selected_electrons = zeros(Int64, L)
    for i in selected_orbitals
        selected_electrons[i] += 1
    end
    config = Configuration(selected_electrons, selected_orbitals)
    return config
end


"""
Initialize a configuration for each spin specie.
The particle numbers are individually conserved.
"""
function initialize_configuration(Nup::Int, Ndn::Int, L::Int)

    orbitals = collect(Int64, 1:L)
    selected_orbital1 = shuffle(orbitals)[1:Nup]
    selected_orbital2 = shuffle(orbitals)[1:Ndn]

    #spin up and spin dn
    selected_orbital1 = selected_orbital1 .* 2 .- 1
    selected_orbital2 = selected_orbital2 .* 2

    selected_orbitals = shuffle(vcat(selected_orbital1, selected_orbital2))
    selected_electrons = zeros(Int64, 2 * L)
    for i in selected_orbitals
        selected_electrons[i] += 1
    end


    config = Configuration(selected_electrons, selected_orbitals)
end



abstract type Wavefunction end

struct SlaterDeterminant <: Wavefunction
    #Mocoeffs defined in L orbitals and Nelec electrons
    Mocoeff::Matrix{ComplexF64}
end



function amplitude(ψ::SlaterDeterminant, x::Configuration)
    @warn "Calling amplitude is not needed, and maybe instable"
    # Compute Ψ(x) for Slater determinant
    determinant = det(ψ.Mocoeff[x.Electrons, :])

    return determinant
end

function logabsamplitude(ψ::SlaterDeterminant, x::Configuration)
    # Compute log|Ψ(x)| and phase for Slater determinant
    logabs, phase = logabsdet(ψ.Mocoeff[x.Electrons, :])
    return logabs, phase
end

(sl::SlaterDeterminant)(x) = sl.Mocoeff[x, :]



function initialize_Slater(Nelec::Int64, L::Int64)
    #generate random Unitary rotation, the matrix is L×L
    U = randn(ComplexF64, L, L)
    U = U + U'
    U = U / norm(U)

    ψ = SlaterDeterminant(U[:, 1:Nelec])

    return ψ, U
end







# Add more wavefunctions here (RBM, Jastrow, etc)
struct Gutzwiller <: Wavefunction
    # Gutzwiller wavefunction parameters
    g::Float64
    Mocoeff::Matrix{ComplexF64}
end

function amplitude(ψ::Gutzwiller, x::Configuration)
    # Compute Ψ(x) for Gutzwiller wavefunction
    @warn "Calling amplitude is not needed, and maybe instable"
    determinant = det(ψ.Mocoeff[x.Electrons, :])
    spactialOrbital = x.Orbitals[1:2:end] + x.Orbitals[2:2:end]
    double_occupancy = sum(spactialOrbital .> 1)  # Count double occupancy

    return exp(-ψ.g * double_occupancy) * determinant

end

function logabsamplitude(ψ::Gutzwiller, x::Configuration)
    logabs_det, phase = logabsdet(ψ.Mocoeff[x.Electrons, :])
    spactialOrbital = x.Orbitals[1:2:end] + x.Orbitals[2:2:end]
    double_occupancy = sum(spactialOrbital .> 1)  # Count double occupancy
    return -ψ.g * double_occupancy + logabs_det, phase
end

(gw::Gutzwiller)(x) = begin
    L = size(gw.Mocoeff, 1)
    Nelec = length(x)
    # Electrons[i] = orbital label for electron i
    Orbitals = zeros(Int64, L)
    for i in x
        Orbitals[i] += 1
    end
    spactialOrbital = Orbitals[1:2:end] + Orbitals[2:2:end]
    double_occupancy = sum(spactialOrbital .> 1)  # Count double occupancy
    gw.Mocoeff[x, :] * exp(-gw.g * double_occupancy / Nelec)
end



struct Jastrow <: Wavefunction
    # Jastrow wavefunction parameters
    g::Matrix{Float64}
    Mocoeff::Matrix{ComplexF64}

end

function amplitude(ψ::Jastrow, x::Configuration)
    # Compute Ψ(x) for Jastrow wavefunction
    determinant = det(ψ.Mocoeff[x.Electrons, :])

    # Assuming g is a vector of parameters for each pair of electrons
    exponent_sum = 0.0
    for i in eachindex(x.Electrons)
        for j in i:length(x.Electrons)

            # Assuming g is a matrix of parameters for each pair of electrons
            exponent_sum += ψ.g[x.Electrons[i], x.Electrons[j]]

        end
    end
    jastrow_factor = exp(-exponent_sum)

    return jastrow_factor * determinant

end

function logabsamplitude(ψ::Jastrow, x::Configuration)
    @error "Not implemented"
end

struct JastrowLimited <: Wavefunction
    # Jastrow wavefunction with limited parameters
    g::Float64
    Vupup::Float64
    Vupdn::Float64
    Mocoeff::Matrix{ComplexF64}
    NNind::Dict{Int64,Vector{Int64}}
end


function amplitude(ψ::JastrowLimited, x::Configuration)
    # Compute Ψ(x) for JastrowLimited wavefunction
    determinant = det(ψ.Mocoeff[x.Electrons, :])

    # Assuming g is a vector of parameters for each pair of electrons
    exponent_sum = 0.0
    for i in eachindex(x.Electrons)
        for j in i+1:length(x.Electrons)
            if fld1(x.Electrons[j], 2) ∈ ψ.NNind[fld1(x.Electrons[i], 2)]
                # If the two electrons have the same spin
                if mod(x.Electrons[i], 2) == mod(x.Electrons[j], 2)
                    exponent_sum += -ψ.Vupup
                else
                    exponent_sum += -ψ.Vupdn
                end
            end
        end
    end

    spactialOrbital = x.Orbitals[1:2:end] + x.Orbitals[2:2:end]
    double_occupancy = sum(spactialOrbital .> 1)  # Count double occupancy
    exponent_sum += -ψ.g * double_occupancy  # Add the Gutzwiller term
    jastrow_factor = exp(exponent_sum)

    return jastrow_factor * determinant

end

function logabsamplitude(ψ::JastrowLimited, x::Configuration)
    logabs_det, phase = logabsdet(ψ.Mocoeff[x.Electrons, :])

    exponent_sum = 0.0
    for i in eachindex(x.Electrons)
        for j in i+1:length(x.Electrons)
            if fld1(x.Electrons[j], 2) ∈ ψ.NNind[fld1(x.Electrons[i], 2)]
                # If the two electrons have the same spin
                if mod(x.Electrons[i], 2) == mod(x.Electrons[j], 2)
                    exponent_sum += -ψ.Vupup
                else
                    exponent_sum += -ψ.Vupdn
                end
            end
        end
    end
    spactialOrbital = x.Orbitals[1:2:end] + x.Orbitals[2:2:end]
    double_occupancy = sum(spactialOrbital .> 1)  # Count double occupancy
    exponent_sum += -ψ.g * double_occupancy  # Add the Gutzwiller term

    return exponent_sum + logabs_det, phase
end

(jl::JastrowLimited)(x) = begin
    L = size(jl.Mocoeff, 1)
    Nelec = length(x)
    Orbitals = zeros(Int64, L)
    for i in x
        Orbitals[i] += 1
    end

    exponent_sum = 0.0
    for i in eachindex(x)
        for j in i+1:length(x)
            if fld1(x[j], 2) ∈ jl.NNind[fld1(x[i], 2)]
                # If the two electrons have the same spin
                if mod(x[i], 2) == mod(x[j], 2)
                    exponent_sum += -jl.Vupup
                else
                    exponent_sum += -jl.Vupdn
                end
            end
        end
    end
    spactialOrbital = Orbitals[1:2:end] + Orbitals[2:2:end]
    double_occupancy = sum(spactialOrbital .> 1)  # Count double occupancy
    exponent_sum += -jl.g * double_occupancy

    return jl.Mocoeff[x, :] * exp(exponent_sum / Nelec)
end

rng = MersenneTwister(114)                # optional, for reproducibility
μ = 1.0f0                             # desired mean
σ = 0.1f0
myinit = (dims...) -> μ .+ σ .* randn(rng, Float32, dims...)




struct ResNet
    perceptron::Dense
end

Flux.@layer ResNet

function ResNet(in_size::Int, out_size::Int)
    ResNet(Dense(in_size, out_size, selu; init=Flux.kaiming_normal))
end

(rs::ResNet)(x) = begin
    x = x + rs.perceptron(x)
end

struct Encoder
    G1::Vector{Float32}
    G2::Vector{Float32}
    PositionVectors::Matrix{Float32}
end

struct SlaterNet <: Wavefunction
    encoder::Encoder
    layernorm::LayerNorm
    W0::Dense
    mlp::Vector{ResNet}
    ReProj::Dense
    # ImProj::Dense
end

Flux.@layer SlaterNet

function create_position_vectors(Nx, Ny)
    L = Nx * Ny * 2
    pos_vecs = zeros(Float32, 2, L)
    for i in 1:L
        orb = fld1(i, 2)
        x = (orb - 1) % Nx + 1
        y = div(orb - 1, Nx) + 1
        pos_vecs[:, i] = [x - 1, y - 1]
    end
    return pos_vecs
end

function SlaterNet(; emb_size, Nx, Ny, Nelec)
    SlaterNet(
        Encoder(; Nx, Ny),
        LayerNorm(7),
        Dense(7 => emb_size; init=Flux.kaiming_normal),
        map(_ -> ResNet(emb_size, emb_size), 1:3),
        Dense(emb_size, Nelec; init=Flux.kaiming_normal),
        # Dense(emb_size, Nelec)
    )

end

(sl::SlaterNet)(x_in) = begin
    feature = Flux.Zygote.@ignore sl.encoder(x_in)
    Nelec = size(feature, 2)
    batch_size = size(feature, 3)
    feature_reshaped = reshape(feature, 7, Nelec * batch_size)
    # x_processed = sl.layernorm(feature_reshaped)
    x_processed = sl.W0(feature_reshaped)

    for layer in sl.mlp
        x_processed = layer(x_processed)
    end

    RealOrbitals = sl.ReProj(x_processed)

    RealOrbitals_reshaped = reshape(RealOrbitals, Nelec, Nelec * batch_size)

    # dets = [det(realamp_reshaped[:, :, i] + 1e-6 * I) for i in 1:batch_size]


    return RealOrbitals_reshaped
end

function logabsamplitude(ψ::SlaterNet, x::Configuration)
    logabs, phase = logabsdet(ψ(x.Electrons))
    return logabs, phase
end




struct AttentionBlock
    ln::LayerNorm
    mha::MultiHeadAttention
    ffn::Dense
end

Flux.@layer AttentionBlock

function AttentionBlock(; emb_size, num_heads)
    AttentionBlock(
        LayerNorm(emb_size),
        MultiHeadAttention(emb_size, nheads=num_heads, dropout_prob=0.1),
        Dense(emb_size, emb_size, tanh; bias=true, init=Flux.Flux.kaiming_normal)
    )
end


function (ab::AttentionBlock)(x)
    x_norm = ab.ln(x)
    mhaout, _ = ab.mha(x_norm, x_norm, x_norm; mask=nothing)
    x = x + mhaout
    x = x + ab.ffn(x)
    x
end




function Encoder(; Nx, Ny)
    Encoder(
        [2π / Nx, 0],
        [0, 2π / Ny],
        create_position_vectors(Nx, Ny)
    )
end


(encoder::Encoder)(x_in) = begin
    is_single_input = ndims(x_in) == 1
    x = is_single_input ? reshape(x_in, :, 1) : x_in

    #dimension of x (Nelec,batch_size)
    Nelec, batch_size = size(x)

    Rvec = reshape(encoder.PositionVectors[:, x[:]], 2, Nelec, batch_size)

    # manual broadcast of matrix multiplication
    inner1 = reshape(encoder.G1' * reshape(Rvec, 2, :), 1, Nelec, batch_size)
    inner2 = reshape(encoder.G2' * reshape(Rvec, 2, :), 1, Nelec, batch_size)

    #feature dimension (feature_dim,Nelec,batch_size)
    feature = vcat(sin.(inner1), sin.(inner2), cos.(inner1), cos.(inner2))

    feature2 = zeros(Float32, 3, Nelec, batch_size)

    # Fill spin information
    for b in 1:batch_size
        for n in 1:Nelec
            # Check if spin up (odd index) or spin down (even index)
            if mod(x[n, b], 2) == 1  # Spin up
                feature2[1, n, b] = 1.0
            else  # Spin down
                feature2[2, n, b] = 1.0
            end

            # Check for doubly occupied orbitals
            spatial_orbital = fld1(x[n, b], 2)  # Get spatial orbital index

            # Count electrons in this spatial orbital
            orbital_count = count(i -> fld1(x[i, b], 2) == spatial_orbital, 1:Nelec)

            # If orbital is doubly occupied, set feature2[3, n, b] = 1
            if orbital_count > 1
                feature2[3, n, b] = 1.0
            end
        end
    end

    # Concatenate features
    feature = vcat(feature, feature2)






    return feature
end


struct TransformerNet <: Wavefunction
    encoder::Encoder
    W0::Dense
    attblock::Vector{AttentionBlock}
    layernorm::LayerNorm
    ReProj::Dense
    # ImProj::Dense
end


function TransformerNet(; num_att_block, num_heads, num_slaters, embsize, Nx, Ny, Nelec)
    TransformerNet(
        Encoder(; Nx, Ny),
        Dense(7 => embsize),
        map(_ -> AttentionBlock(emb_size=embsize, num_heads=num_heads), 1:num_att_block),
        LayerNorm(embsize),
        Dense(embsize => Nelec * num_slaters)
    )
end

(TransNet::TransformerNet)(x_in) = begin
    x_in = Flux.Zygote.@ignore TransNet.encoder(x_in)
    Nelec = size(x_in, 2)
    batch_size = size(x_in, 3)
    #feature dimension (featuredim,Nelec,batch_size)
    x_proc = TransNet.W0(x_in)
    for block in TransNet.attblock
        x_proc = block(x_proc)
    end
    x_proc = TransNet.layernorm(x_proc)

    x_proc = TransNet.ReProj(x_proc)

    num_slaters = div(size(x_proc, 1), Nelec)

    # Reshape to (Nelec, num_slaters, Nelec, batch_size) which is (orbital_dim, slater_idx, electron_idx, batch_idx)
    x_reshaped = reshape(x_proc, Nelec, num_slaters * Nelec * batch_size)

    # Permute to (Nelec, Nelec, num_slaters, batch_size) which is (electron_idx, orbital_dim, slater_idx, batch_idx)
    # x_permuted = permutedims(x_reshaped, [1, 3, 2, 4])


    return x_reshaped
end
amplitude(tn::TransformerNet, x::Configuration) = begin
    logabs, phase = logabsamplitude(tn, x)
    if !isfinite(logabs) || phase == 0
        return zero(phase)
    end
    return phase * exp(logabs)
end

function logabsamplitude(ψ::TransformerNet, x::Configuration)
    raw = ψ(x.Electrons)
    Nelec = length(x.Electrons)
    num_slaters = div(size(raw, 2), Nelec)
    raw_stack = reshape(raw, Nelec, num_slaters, Nelec)
    orbital_stack = permutedims(raw_stack, (1, 3, 2))

    num_slaters >= 1 || error("TransformerNet requires at least one Slater determinant")

    first_log, first_phase = logabsdet(@view orbital_stack[:, :, 1])
    max_log = first_log
    scaled_sum = first_phase

    for idx in 2:num_slaters
        logval, phase = logabsdet(@view orbital_stack[:, :, idx])
        if logval > max_log
            scaled_sum = phase + scaled_sum * exp(max_log - logval)
            max_log = logval
        else
            scaled_sum = scaled_sum + phase * exp(logval - max_log)
        end
    end

    if !isfinite(max_log)
        return -Inf, zero(first_phase)
    end

    total_amp = exp(max_log) * scaled_sum
    abs_total = abs(total_amp)
    if abs_total == 0
        return -Inf, zero(total_amp)
    end
    return log(abs_total), total_amp / abs_total
end







struct LinearAttentionBlock
    mask::Dense
    Wk::Dense
    Wq::Dense
    Wv::Dense

end
