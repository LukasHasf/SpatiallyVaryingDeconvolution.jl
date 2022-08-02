@testset "Temporary functions" begin
    @testset "linshift!" begin
        A = [1, 2, 3, 4]
        B = similar(A)
        linshift!(B, A, [0])
        @test B == A
        linshift!(B, A, [1])
        @test B == [0, 1, 2, 3]
        linshift!(B, A, [-1])
        @test B == [2, 3, 4, 0]

        A = [1 2 3 4; 5 6 7 8; 9 10 11 12; 13 14 15 16]
        B = similar(A)
        linshift!(B, A, [0, 0])
        @test B == A
        linshift!(B, A, [1, 1])
        @test B == [0 0 0 0; 0 1 2 3; 0 5 6 7; 0 9 10 11]
        linshift!(B, A, [-1, -1])
        @test B == [6 7 8 0; 10 11 12 0; 14 15 16 0; 0 0 0 0]
    end

    @testset "padND" begin
        Ny, Nx = 101, 100
        x = rand(Float64, Ny, Nx)
        @test size(padND(x, 2)) == 2 .* size(x)

        Ny, Nx, Nz = 101, 100, 39
        x = rand(Float64, Ny, Nx, Nz)

        @test size(padND(x, 3)) == 2 .* size(x)

        nrPSFs = 10
        x = rand(Float64, Ny, Nx, Nz, nrPSFs)
        @test size(padND(x, 3)) == ((2 .* size(x)[1:3])..., nrPSFs)
    end

    @testset "Test shift registration" begin
        function calc_shifts(shifts)
            ND = size(shifts)[1]
            Ny = 51
            Nx = 51
            Nz = 50
            Ns = [Ny, Nx, Nz][1:ND]
            nrPSFs = size(shifts)[2]
            psfs = zeros(Float32, Ns..., nrPSFs)
            center = Int.(Ns .÷ 2 .+ 1)
            for shift_index in 1:nrPSFs
                psfs[(center .- shifts[:, shift_index])..., shift_index] = one(Float64)
            end
            _, shifts = registerPSFs(psfs, selectdim(psfs, ndims(psfs), nrPSFs ÷ 2 + 1))
            return shifts
        end

        @testset "Test shift is zero for exact same PSF" begin
            in_shifts = zeros(Int32, (2, 9))
            @test in_shifts ≈ calc_shifts(in_shifts)
            in_shifts = zeros(Int32, (3, 9))
            @test in_shifts ≈ calc_shifts(in_shifts)
        end

        @testset "Test random shifts are registered correctly" begin
            in_shifts = rand(-20:20, (2, 17))
            in_shifts[:, size(in_shifts)[2] ÷ 2 + 1] .= 0
            @test in_shifts ≈ calc_shifts(in_shifts)
            in_shifts = rand(-20:20, (3, 9))
            in_shifts[:, size(in_shifts)[2] ÷ 2 + 1] .= 0
            @test in_shifts ≈ calc_shifts(in_shifts)
        end
    end
end
