include("../src/VMC.jl")


let
    # Nelec = 3
    # ψ = Flux.f32(SlaterNet(; emb_size=5, Nx=6, Ny=6, Nelec=Nelec))

    # xs = Array{Int32}(undef, Nelec, 2)
    # for i in 1:2
    #     o = collect(Int, 1:72)
    #     x = shuffle(o)[1:Nelec]
    #     xs[:, i] .= x
    # end
    # @show m1 = reshape(ψ(xs), Nelec, Nelec, 2)[:, :, 2]
    # @show m2 = ψ(xs[:, 2])
    # @show Flux.Losses.mse(m1, m2)
    # # l, u = lu(ψ(x))
    # @show diag(u)
    # @show ψ(x) \ I
    # @show logabsamplitude(ψ(x))
    #TODO: test derivative of log det

    RunVMCNQS(
        6,
        6,
        20,
        1.0,
        5.0;
        TotalSteps=51000,
        ThermalizationSteps=1000,
        OptimizationSteps=20,
        seed=1,
        init_params="test4.bson",
        wf_type=:SlaterNet,
        lr=1e-2
    )

end
