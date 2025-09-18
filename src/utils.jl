function SetSeed(seed::Int)
    Random.seed!(seed)
end

function GenerateNNind(Nx::Int64, Ny::Int64)
    # Generate the nearest neighbor indices for a 2D lattice
    # Nx: number of sites in x direction
    # Ny: number of sites in y direction
    # Returns a dictionary, key is first index, value is a vector of neighbors

    nn_dict = Dict{Int,Vector{Int}}()

    for i in 1:Nx
        for j in 1:Ny
            site = (i - 1) * Ny + j
            neighbors = Vector{Int}()

            # Right neighbor (periodic boundary)
            right_i = i == Nx ? 1 : i + 1
            push!(neighbors, (right_i - 1) * Ny + j)

            # Left neighbor (periodic boundary)
            left_i = i == 1 ? Nx : i - 1
            push!(neighbors, (left_i - 1) * Ny + j)

            # Up neighbor (periodic boundary)
            up_j = j == Ny ? 1 : j + 1
            push!(neighbors, (i - 1) * Ny + up_j)

            # Down neighbor (periodic boundary)
            down_j = j == 1 ? Ny : j - 1
            push!(neighbors, (i - 1) * Ny + down_j)

            nn_dict[site] = neighbors
        end
    end

    return nn_dict

end




# Returns (log|det(A)|, sign(det(A))) for real CuArray A
function logabsdet_cuda(A::CuArray{T,2}) where {T<:AbstractFloat}
    # lu! overwrites; copy if you need A later
    F = lu!(copy(A); check=false)               # cuSOLVER getrf! on device

    # Diagonal of U lives on the diagonal of the packed factors
    d = diag(F.factors)                         # CuVector on device

    # log|det(A)| = sum(log(abs(diag(U))))
    logabs = sum(log.(abs.(d)))                 # GPU reduction -> host scalar

    # sign(det(A)) = (-1)^(#row-swaps) * prod(sign(diag(U)))
    sgnU = prod(sign.(d))                       # in {-1, 0, +1} (0 if singular)

    n = size(A, 1)
    idx = CUDA.CuArray(collect(Int32, 1:n))    # [1,2,...,n] on device
    swaps = sum(F.ipiv .!= idx)                 # GPU reduction counting swaps
    pivotsgn = ifelse(isodd(swaps), -one(T), one(T))

    sgn = pivotsgn * sgnU

    return logabs, sgn
end
