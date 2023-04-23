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
<body bg-info>
<div class="container">
    <div class="row">
      <table align="center">
        <tr>
          <td><div class="live_feed">
            <h3 class="mt-5">Capture Image</h3>
            <img src="{{ url_for('video_feed') }}" width="60%">
        </div></td>
          <td><div>
            <canvas id="myChart" style="width:100%;max-width:600px;margin-top:20%"></canvas>
            <script>
                var xValues = [50,60,70,80,90,100,110,120,130,140,150];
                var yValues = [7,8,8,9,9,9,10,11,14,14,15];
                
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

<?php

require_once('./conn.php');

$sql = "SELECT * FROM crack_data";
$result = mysqli_query($conn,$sql);

echo "<div class='container-three'>";

if($result->num_rows > 0) {
    echo "<div class='container'>
    <table border = 1 style='width:100%;'><tr>
          <th>Original Image</th>
          <th>Length</th>
          <th>Width</th>
          <th>Depth</th>";
    while($row = $result->fetch_assoc()) {
      echo "<td>".$row['length']."</td>
            <td>".$row['length']."</td>
            <td>".$row['width']."</td>
            <td>".$row['depth']."</td></tr>";
    }
    echo "</table></div>";
  } else {
    echo "0 results";
  }

$conn->close();
?>