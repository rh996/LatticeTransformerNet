include("../src/VMC.jl")


let

    num_att_block = 3
    num_heads = 4
    num_slaters = 3
    embsize = 12
    lr = 1e-3

    Nx = 4
    Ny = 4
    Nelec = 4
    Nup = 2
    Ndn = 2
    t = 1.0
    U = 5.0
    RunVMCNQS(
        Nx,
        Ny,
        Nup,
        Ndn,
        t,
        U;
        TotalSteps=11000,
        ThermalizationSteps=1000,
        OptimizationSteps=100,
        seed=11,
        lr=lr,
        # init_params="./data/transformernet_from_slaternet_Nx_6_Ny_6_Nelec_20.bson",
        init_params="./data/transformernet_slater_Nx_$(Nx)_Ny_$(Ny)_Nelec_$(Nelec)_att_$(num_att_block)_heads_$(num_heads)_slaters_$(num_slaters)_emb_$(embsize).bson",
        wf_type=:TransformerNet
    )

end
