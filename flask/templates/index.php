<!DOCTYPE html>
<html lang="en">
<head>
    <!-- Required meta tags -->
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/2.9.4/Chart.js"></script>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <!-- Bootstrap CSS -->
    <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.1.3/css/bootstrap.min.css"
          integrity="sha384-MCw98/SFnGE8fJT3GXwEOngsV7Zt27NXFoaoApmYm81iuXoPkFOJwJ8ERdknLPMO" crossorigin="anonymous">

    <title>Live Streaming Demonstration</title>
</head>
<body>
<div class="container">
    <div class="row">
      <table align="center">
        <tr>
          <td><div class="live_feed">
            <h3 class="mt-5">Live Streaming</h3>
            <img src="{{ url_for('video_feed') }}" width="60%">
        </div></td>
          <td><div>
            <canvas id="myChart" style="width:100%;max-width:600px;margin-top:20%"></canvas>
            <script>
                var xValues = [7,8,8,9,9,9,10,11,14,14,15];
                var yValues = [50,60,70,80,90,100,110,120,130,140,150];
                
                new Chart("myChart", {
                  type: "line",
                  data: {
                    labels: xValues,
                    datasets: [{
                      fill: false,
                      lineTension: 0,
                      backgroundColor: "rgba(0,0,255,1.0)",
                      borderColor: "rgba(0,0,255,0.1)",
                      data: yValues
                    }]
                  },
                  options: {
                    legend: {display: false},
                    scales: {
                      yAxes: [{ticks: {min: 6, max:16}}],
                    }
                  }
                });
                </script>
        </div></td>
        </tr>
      </table>
        
    </div>
</div>
</body>
</html>