def write_message(words, scores, str_words, cluster, count):
    message = """ <!DOCTYPE html>
    <meta charset="utf-8">
    <style>

    header {
    background-color: #99ccff;
    padding: 40px;
    text-align: center;
    font-size: 25px;
    color: black;
    }

    div#title {
        font-family: "Lucida Console";
        color: black;
        width: 36%;
        padding-left: 2%;
        float: left;
    }
    div#title2 {
        font-family: "Lucida Console";
        color: black;
        width: 62%;
        float: right;
        position:relative; left:250px;
    }
    div#table {
        font-family: "Lucida Console";
        color: black;
        width: 62%;
        height: 500px;
        float: right;
        position:relative; left:250px; }

    footer {
        background-color: #777;
        padding: 10px;
        text-align: center;
        color: white;
    }
    </style>
    </head>
    <body>
      <header>
        <h2>PRI Project 2019/2020</h2>
        <h3>Group 21</h3>
      </header>

      <div id="title" >
      <h3>Keyword Cloud</h3>
      </div>
      <div id="title2">
      <h3>Top 10 Keywords</h3>
      </div>
      <div id="table">
        <style type="text/css">
          .tg  {border-collapse:collapse;border-spacing:0;border-color:#aaa;}
          .tg td{font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:#aaa;color:#333;background-color:#fff;}
          .tg th{font-family:Arial, sans-serif;font-size:14px;font-weight:normal;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:#aaa;color:#fff;background-color:#99ccff;}
          .tg .tg-baqh{text-align:center;vertical-align:top}
          .tg .tg-amwm{font-weight:bold;text-align:center;vertical-align:top}
          </style>
          <table class="tg">
            <tr>
              <th class="tg-amwm"> Keywords </th>
              <th class="tg-amwm"> PageRank </th>
              <th class="tg-amwm"> Articles Count </th>
              <th class="tg-amwm"> News Articles Titles </th>
            </tr>
            <tr>
              <td class="tg-baqh">""" + words[0] + """</td>
              <td class="tg-baqh">""" + scores[0] + """</td>
              <td class="tg-baqh">""" + str(count[0]) + """</td>
              <td class="tg-baqh">
              <button type="button" onclick="alert(""" + cluster[0] + """)">Show Cluster</button>
              </td>
            </tr>
            <tr>
              <td class="tg-baqh">""" + words[1] + """</td>
              <td class="tg-baqh">""" + scores[1] + """</td>
              <td class="tg-baqh">""" + str(count[1]) + """</td>
              <td class="tg-baqh">
              <button type="button" onclick="alert(""" + cluster[1] + """)">Show Cluster</button>
              </td>
            </tr>
            <tr>
              <td class="tg-baqh">""" + words[2] + """</td>
              <td class="tg-baqh">""" + scores[2] + """</td>
              <td class="tg-baqh">""" + str(count[2]) + """</td>
              <td class="tg-baqh">
              <button type="button" onclick="alert(""" + cluster[2] + """)">Show Cluster</button>
              </td>
            </tr>
            <tr>
              <td class="tg-baqh">""" + words[3] + """</td>
              <td class="tg-baqh">""" + scores[3] + """</td>
              <td class="tg-baqh">""" + str(count[3]) + """</td>
              <td class="tg-baqh">
              <button type="button" onclick="alert(""" + cluster[3] + """)">Show Cluster</button>
              </td>
            </tr>
            <tr>
              <td class="tg-baqh">""" + words[4] + """</td>
              <td class="tg-baqh">""" + scores[4] + """</td>
              <td class="tg-baqh">""" + str(count[4]) + """</td>
              <td class="tg-baqh">
              <button type="button" onclick="alert(""" + cluster[4] + """)">Show Cluster</button>
              </td>
            </tr>
              <tr>
              <td class="tg-baqh">""" + words[5] + """</td>
              <td class="tg-baqh">""" + scores[5] + """</td>
              <td class="tg-baqh">""" + str(count[5]) + """</td>
              <td class="tg-baqh">
              <button type="button" onclick="alert(""" + cluster[5] + """)">Show Cluster</button>
              </td>
            </tr>
            <tr>
              <td class="tg-baqh">""" + words[6] + """</td>
              <td class="tg-baqh">""" + scores[6] + """</td>
              <td class="tg-baqh">""" + str(count[6]) + """</td>
              <td class="tg-baqh">
              <button type="button" onclick="alert(""" + cluster[6] + """)">Show Cluster</button>
              </td>
            </tr>
            <tr>
              <td class="tg-baqh">""" + words[7] + """</td>
              <td class="tg-baqh">""" + scores[7] + """</td>
              <td class="tg-baqh">""" + str(count[7]) + """</td>
              <td class="tg-baqh">
              <button type="button" onclick="alert(""" + cluster[7] + """)">Show Cluster</button>
              </td>
            </tr>
            <tr>
              <td class="tg-baqh">""" + words[8] + """</td>
              <td class="tg-baqh">""" + scores[8] + """</td>
              <td class="tg-baqh">""" + str(count[8]) + """</td>
              <td class="tg-baqh">
              <button type="button" onclick="alert(""" + cluster[8] + """)">Show Cluster</button>
              </td>
            </tr>
            <tr>
              <td class="tg-baqh">""" + words[9] + """</td>
              <td class="tg-baqh">""" + scores[9] + """</td>
              <td class="tg-baqh">""" + str(count[9]) + """</td>
              <td class="tg-baqh">
              <button type="button" onclick="alert(""" + cluster[9] + """)">Show Cluster</button>
              </td>
            </tr>
          </table>

      </div>

      <!-- ################################################################################################################# -->
        <script src="https://d3js.org/d3.v4.js"></script>
        <!-- Load d3-cloud -->
        <script src="https://cdn.jsdelivr.net/gh/holtzy/D3-graph-gallery@master/LIB/d3.layout.cloud.js"></script>
        <script>
            var words = [ """ + "\"" + str_words + "\"" + """];
            var newWords = [];
            newWords.push(words[0].split(","));

            console.log(newWords);


            var margin = {top: 10, right: 10, bottom: 10, left: 10},
            width = 450 - margin.left - margin.right,
            height = 450 - margin.top - margin.bottom;

            // append the svg object to the body of the page
            var svg = d3.select("#title").append("svg")
            .attr("width", width + margin.left + margin.right)
            .attr("height", height + margin.top + margin.bottom)
            .append("g")
            .attr("transform",
            "translate(" + margin.left + "," + margin.top + ")");

            // Constructs a new cloud layout instance. It run an algorithm to find the position of words that suits your requirements
            var layout = d3.layout.cloud()
            .size([width, height])
            .words(newWords[0].map(function(d) { return {text: d}; }))
            .padding(10)
            .fontSize(10)
            .on("end", draw);
            layout.start();

            // This function takes the output of 'layout' above and draw the words
            // Better not to touch it. To change parameters, play with the 'layout' variable above
            function draw(words) {
            svg
            .append("g")
            .attr("transform", "translate(" + layout.size()[0] / 2 + "," + layout.size()[1] / 2 + ")")
            .selectAll("text")
            .data(words)
            .enter().append("text")
            .style("font-size", function(d) { return d.size + "px"; })
            .attr("text-anchor", "middle")
            .attr("transform", function(d) {
            return "translate(" + [d.x, d.y] + ")rotate(" + d.rotate + ")";
            })
            .text(function(d) { return d.text; });
            }



        </script>
    </body>
    """

    return message
