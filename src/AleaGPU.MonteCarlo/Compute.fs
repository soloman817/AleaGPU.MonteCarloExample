module Compute

open Alea.CUDA
open Alea.CUDA.Utilities

[<ReflectedDefinition>]
let apply f (n : int) (input : deviceptr<float32>) (output : deviceptr<float32>) = 
    let iStart = blockIdx.x * blockDim.x + threadIdx.x
    let iStep  = gridDim.x * blockDim.x
    let mutable i = iStart
    while i < n do
        output.[i] <- f input.[i]
        i <- i + iStep 

[<ReflectedDefinition;AOTCompile>]
let applySin n input output = apply sin n input output

let launchParam (worker : Worker) n =
    let blockSize = 256
    let numSm = worker.Device.Attributes.MULTIPROCESSOR_COUNT
    let gridSize = min (16 * numSm) (divup n blockSize)
    LaunchParam(gridSize, blockSize)

let example () =
    
    let n = 500

    // determine worker
    let worker = Worker.Default

    // allocate and transfer to device
    let xs = Histogram.linbin -4.0 4.0 (n-1) |> Array.map float32
    let input = worker.Malloc xs

    // Allocate on device
    let output = worker.Malloc n

    // determine launch parameters
    let lp = launchParam worker n

    // launch kernel
    worker.Launch <@ applySin @> lp n input.Ptr output.Ptr
    
    // transfer results from device to host 
    xs, output.Gather()
