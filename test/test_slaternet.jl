include("../src/VMC.jl")


let
    # RunVMCNQS(
    #     6,
    #     6,
    #     20,
    #     1.0,
    #     5.0;
    #     TotalSteps=51000,
    #     ThermalizationSteps=1000,
    #     OptimizationSteps=20,
    #     seed=114,
    #     init_params="./data/test2.bson",
    #     wf_type=:SlaterNet,
    #     lr=1e-3
    # )

    RunVMCNQS(
        4,
        4,
        2,
        2,
        1.0,
        5.0;
        TotalSteps=51000,
        ThermalizationSteps=1000,
        OptimizationSteps=50,
        seed=1,
        init_params="./data/test4.bson",
        wf_type=:SlaterNet,
        lr=1e-3
    )
end
