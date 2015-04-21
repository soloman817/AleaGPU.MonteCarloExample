module Histogram

/// Create `n` bins between `xmin` and `xmax` that are of equal size 
let linbin (xmin : float) (xmax : float) (n : int) =
    let dx = (xmax - xmin) / float n
    Array.init (n+1) (fun i -> xmin + float i * dx)

/// Bin `data` into bins given by `xlim`. Returns the counts and the midpoints. 
let histogram (xlim : float[]) (data : float[]) : int[] * float[] =
    let bin i = data |> Array.fold (fun c d -> c + if xlim.[i] <= d && d < xlim.[i+1] then 1 else 0) 0
    let count = Array.init (xlim.Length - 1) bin
    let xmid = Array.init (xlim.Length - 1) (fun i -> (xlim.[i] + xlim.[i+1]) / 2.0)
    count, xmid
