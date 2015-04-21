module MonteCarlo

open Alea.CUDA
open Alea.CUDA.Utilities
open Alea.CUDA.Unbound.Rng

type Step   = int
type Real   = float32
type Sample = int

/// Performs an Euler step in log-coordinate space
[<ReflectedDefinition>]
let eulerLogStep (dt : Real) (mu : Real) (sigma : Real) (dW : Real) s =
    s * (exp <| dt * (mu - (1G / 2G) * sigma * sigma) + sqrt dt * sigma * dW )

/// Time loop    
[<ReflectedDefinition>]
let mcLoop
        (tStart : Step)
        (tEnd   : Step)
        (dt     : Step -> Real)
        (s0     : Real)
        (mu     : Step -> Real)
        (sigma  : Step -> Real -> Real)
        (dW     : Step -> Real) =

    let mutable t = tStart
    let mutable s = s0
    
    while t < tEnd do
        let s' = eulerLogStep (dt t) (mu t) (sigma t s) (dW t) s
        s <- s'
        t <- t + 1

    s

/// Sample loop    
[<ReflectedDefinition>]
let forAll (n : Sample) (f : Sample -> unit) =
    let iStart = blockIdx.x * blockDim.x + threadIdx.x
    let iStep  = gridDim.x * blockDim.x
    let mutable i = iStart
    while i < n do
        f i
        i <- i + iStep

/// Simulation kernel
[<ReflectedDefinition;AOTCompile>]
let mcSim 
        (nmc   : Sample)
        (nt    : Step)
        (dt    : deviceptr<Real>)
        (s0    : Real)
        (mu    : Real)
        (sigma : Real)
        (dW    : deviceptr<Real>)
        (st    : deviceptr<Real>) =

    let go sample =
        st.[sample] <- mcLoop 0 nt (fun t -> dt.[t]) s0 (fun _ -> mu) (fun _ _ -> sigma) (fun t -> dW.[nt * sample + t])

    forAll nmc go
    
let launchParam (worker : Worker) n =
    let blockSize = 256
    let numSm = worker.Device.Attributes.MULTIPROCESSOR_COUNT
    let gridSize = min (16 * numSm) (divup n blockSize)
    LaunchParam(gridSize, blockSize)

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

    // transfer to device
    let devDt = worker.Malloc dt
    let devSt = worker.Malloc nmc

    // determine launch parameters
    let lp = launchParam worker nmc

    // launch kernel
    worker.Launch <@ mcSim @> lp nmc nt devDt.Ptr s0 mu sigma prngBuffer.Ptr devSt.Ptr

    // transfer results from device to host 
    devSt.Gather()
