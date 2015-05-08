module MonteCarloCPU

open Alea.CUDA
open Alea.CUDA.Utilities
open Alea.CUDA.Unbound.Rng
open MonteCarlo

/// Sample loop on CPU
let forAll (n : Sample) (f : Sample -> unit) =
    for i = 0 to n - 1 do
        f i

/// Simulation kernel
let mcSim 
        (nmc   : Sample)
        (nt    : Step)
        (dt    : Real [])
        (s0    : Real)
        (mu    : Real)
        (sigma : Real)
        (dW    : Real [])
        (st    : Real []) =

    let go sample =
        st.[sample] <- mcLoop 0 nt (fun t -> dt.[t]) s0 (fun _ -> mu) (fun _ _ -> sigma) (fun t -> dW.[nt * sample + t])

    forAll nmc go
    
let example () =

    // model parameters
    let T = 1.0f
    let s0 = 1.0f
    let mu = 0.02f
    let sigma = 0.20f

    // simulation parameters
    let nt = 40
    let nmc = 1000000
    let dt = Array.create nt (T / float32 nt)

    // parameters of random number generator     
    let numDimensions = nt
    let seed = 1u

    // determine worker
    let worker = Worker.Default
    let target = GPUModuleTarget.Worker(worker)

    // setup random number generator        
    use cudaRandom = (new XorShift7.CUDA.DefaultNormalRandomModuleF32(target)).Create(1, numDimensions, seed) :> IRandom<float32>
    use prngBuffer = cudaRandom.AllocCUDAStreamBuffer (nmc)

    // create random numbers
    cudaRandom.Fill(0, nmc, prngBuffer)
    let prng = prngBuffer.Gather()

    // Run on CPU
    let st = Array.zeroCreate nmc
    mcSim nmc nt dt s0 mu sigma prng st

    st