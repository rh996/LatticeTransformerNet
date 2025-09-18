include("../src/VMC.jl")

using CUDA

CUDA.allowscalar(false)


let
    Nx = 4
    Ny = 4
    Nelec = 4
    batch_size = 3

    RunVMCNQS(
        Nx,
        Ny,
        2,
        2,
        1.0,
        5.0;
        TotalSteps=101000,
        ThermalizationSteps=1000,
        OptimizationSteps=1,
        seed=1,
        init_params="./data/testgpu.bson",
        wf_type=:SlaterNet,
        lr=1e-3
    )
end
