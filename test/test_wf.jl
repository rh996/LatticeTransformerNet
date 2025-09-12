include("../src/VMC.jl")

ml = TransformerNet(; num_att_block=3, num_heads=4, num_slaters=3, embsize=24, Nx=6, Ny=6, Nelec=2)

# encoder = Encoder(; Nx=4, Ny=4)

# xs = [[1, 2, 5, 7], [2, 1, 5, 7], [5, 3, 1, 7], [3, 5, 1, 7]]
# Generate Nelect integers from 1:Nx*Ny*2
Nx, Ny, Nelec = 6, 6, 2
# xs = [randperm(Nx * Ny * 2)[1:Nelec] for _ in 1:3]
xs = [[3, 5], [5, 3], [7, 9], [9, 7], [11, 2], [2, 11]]
xs = hcat(xs...)


# @load "transformernet_slater_Nx_4_Ny_4_Nelec_4_att_3_heads_4_slaters_3_emb_24.bson" ψ
# ml = ψ
display(permutedims(reshape(ml([3, 5]), Nelec, 3, Nelec), [1, 3, 2]))
display(permutedims(reshape(ml([5, 3]), Nelec, 3, Nelec), [1, 3, 2]))

# display(encoder(xs))
