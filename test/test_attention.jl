include("../src/VMC.jl")


let

    num_att_block = 3
    num_heads = 4
    num_slaters = 1
    embsize = 24
    lr = 1e-3

    Nx = 6
    Ny = 6
    Nelec = 6
    t = 1.0
    U = 5.0
    RunVMCNQS(
        Nx,
        Ny,
        Nelec,
        t,
        U;
        TotalSteps=101000,
        ThermalizationSteps=1000,
        OptimizationSteps=100,
        seed=114514,
        lr=lr,
        init_params="transformernet_slater_Nx_$(Nx)_Ny_$(Ny)_Nelec_$(Nelec)_att_$(num_att_block)_heads_$(num_heads)_slaters_$(num_slaters)_emb_$(embsize).bson",
        wf_type=:TransformerNet
    )

end
