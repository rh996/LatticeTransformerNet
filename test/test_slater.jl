include("../src/VMC.jl")
let
    RunVMC(
        6,
        6,
        20,
        1.0,
        5.0;
        TotalSteps=101000,
        ThermalizationSteps=1000,
        OptimizationSteps=50,
        wf_type=:JastrowLimited,
        seed=11,
        lr=0.01
    )

end
