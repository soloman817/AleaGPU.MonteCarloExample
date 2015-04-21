module Randoms

open Alea.CUDA
open Alea.CUDA.Utilities
open Alea.CUDA.Unbound.Rng

let example () =

    // parameters of random number generator     
    let numDimensions = 1
    let seed = 1u
    let nmc = 100000

    // determine worker
    let worker = Worker.Default
    let target = GPUModuleTarget.Worker(worker)

    // setup random number generator        
    use cudaRandom = (new XorShift7.CUDA.DefaultNormalRandomModuleF32(target)).Create(1, numDimensions, seed) :> IRandom<float32>
    use prngBuffer = cudaRandom.AllocCUDAStreamBuffer (nmc)

    // create random numbers
    cudaRandom.Fill(0, nmc, prngBuffer)

    // transfer results from device to host 
    prngBuffer.Gather()
