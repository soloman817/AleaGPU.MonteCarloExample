open FSharp.Charting

[<EntryPoint>]
let main args =

    // generate random numbers on device
    let res = Randoms.example ()       

    let xlim = Histogram.linbin -4.0 4.0 40
    let count, xmid = Histogram.histogram xlim (res |> Array.map float)

    let chart = FSharp.Charting.Chart.Column (Seq.zip xmid count) 
    chart.ShowChart() |> ignore
    chart.SaveChartAs("randoms.png",ChartTypes.ChartImageFormat.Png)

    // compute on device
    let xs, res = Compute.example ()       

    let chart = FSharp.Charting.Chart.Point (Seq.zip xs res) 
    chart.ShowChart() |> ignore
    chart.SaveChartAs("compute.png",ChartTypes.ChartImageFormat.Png)

    // Monte Carlo simulation on device
    let res = MonteCarlo.example ()       

    let xlim = Histogram.linbin 0.4 2.5 40
    let count, xmid = Histogram.histogram xlim (res |> Array.map float)

    let chart = FSharp.Charting.Chart.Column (Seq.zip xmid count) 
    chart.ShowChart() |> ignore
    chart.SaveChartAs("montecarlo.png",ChartTypes.ChartImageFormat.Png)

    0
