module Compute

open Alea.CUDA
open Alea.CUDA.Utilities

[<ReflectedDefinition>]
let apply f (n : int) (input : deviceptr<float32>) (output : deviceptr<float32>) = 
    // initial index of thread
    let iThread = blockIdx.x * blockDim.x + threadIdx.x
    // number of active threads
    let nThread  = gridDim.x * blockDim.x
    // iterate until the threads have processed all elements 
    let mutable i = iThread
    while i < n do
        output.[i] <- f input.[i]
        i <- i + nThread

/// Apply `sin` to all elements of `input` and write to `output`.
///
/// The inputs of the function are of type `int` and `deviceptr<float32>` so that we can
/// compile the function to a kernerl that can be run on the GPU. The annotation `AOTCompile`
/// requests the creation of the Kernel at the creation time of the assembly. If the option
/// is omitted then the Kernel will be compiled just-in-time when it is called for the first
/// time.
/// 
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
