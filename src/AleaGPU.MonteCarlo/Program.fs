module Program

open FSharp.Charting
open System.Windows.Forms

let showAndSave (chart : ChartTypes.GenericChart) (filename : Option<string>) =
    let form = chart.ShowChart()
    Option.iter (fun f -> chart.SaveChartAs(f,ChartTypes.ChartImageFormat.Png)) filename
    form.Show()
    

[<EntryPoint>]
let main args =

    // generate random numbers on device
    let res = Randoms.example ()       

    let xlim = Histogram.linbin -4.0 4.0 40
    let count, xmid = Histogram.histogram xlim (res |> Array.map float)

    let chart = FSharp.Charting.Chart.Column (Seq.zip xmid count) 
    showAndSave chart (Some "randoms.png")

    // compute on device
    let xs, res = Compute.example ()       

    let chart = FSharp.Charting.Chart.Point (Seq.zip xs res) 
    showAndSave chart (Some "compute.png")

    // Monte Carlo simulation on device
    let res = MonteCarlo.example ()       

    let xlim = Histogram.linbin 0.4 2.5 40
    let count, xmid = Histogram.histogram xlim (res |> Array.map float)

    let chart = FSharp.Charting.Chart.Column (Seq.zip xmid count) 
    showAndSave chart (Some "montecarlo.png")

    // Monte Carlo simulation on CPU
    let res = MonteCarloCPU.example ()       

    let xlim = Histogram.linbin 0.4 2.5 40
    let count, xmid = Histogram.histogram xlim (res |> Array.map float)

    let chart = FSharp.Charting.Chart.Column (Seq.zip xmid count) 
    showAndSave chart (Some "montecarlo_cpu.png")

    Application.Run()

    0
