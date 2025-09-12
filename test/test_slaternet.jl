include("../src/VMC.jl")


let
    # RunVMCNQS(
    #     6,
    #     6,
    #     10,
    #     1.0,
    #     5.0;
    #     TotalSteps=51000,
    #     ThermalizationSteps=1000,
    #     OptimizationSteps=20,
    #     seed=114,
    #     init_params="./data/test_10.bson",
    #     wf_type=:SlaterNet,
    #     lr=1e-2
    # )

    RunVMCNQS(
        6,
        6,
        10,
        10,
        1.0,
        5.0;
        TotalSteps=51000,
        ThermalizationSteps=1000,
        OptimizationSteps=20,
        seed=1,
        init_params="./data/test_sz.bson",
        wf_type=:SlaterNet,
        lr=1e-2
    )
end
