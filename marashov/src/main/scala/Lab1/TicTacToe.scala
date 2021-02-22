package Lab1

import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.udf
import vegas.sparkExt.VegasSpark
import vegas.{Area, Nom, Vegas}

import java.io.{File, PrintWriter}

class TicTacToe {
  def execute(): Unit = {
    val resultsPath = "src/main/resources/lab1/tic_tac_toe/results.txt"
    val dataPath = "src/main/resources/lab1/tic_tac_toe/tic_tac_toe.txt"
    val plotPath = "src/main/resources/lab1/tic_tac_toe/plot.html"

    val spark = SparkSession
      .builder()
      .appName("ML_2021")
      .master("local")
      .getOrCreate()

    import spark.implicits._

    val sparkContext = spark.sparkContext
    sparkContext.setLogLevel("ERROR")

    val resultsFile = new File(resultsPath)
    if (!resultsFile.exists()) {

      val dataFrame = spark.read
        .text(dataPath)

      dataFrame.printSchema()
      dataFrame.show()

      val newDF = dataFrame.map(row => {
        val cols = row.getString(0).split(',')
        val cells = cols.slice(0, 9).map {
          case "x" => 2.0
          case "b" => 1.0
          case "o" => 0.0
        }
        val label = cols(9) match {
          case "positive" => 1.0
          case "negative" => 0.0
        }
        (cells, label)
      })

      val newNames = Seq("featuresArray", "label")
      val dfRenamed = newDF.toDF(newNames: _*)
      val convertUDF = udf((array: Seq[Double]) => {
        Vectors.dense(array.toArray)
      })
      val convertedDF = dfRenamed.select(convertUDF($"featuresArray").as("features"), $"label")

      convertedDF.printSchema()
      convertedDF.show(10, truncate = false)

      val writer = new PrintWriter(resultsFile)

      for (i <- 1 to 99 by 1) {
        val testingPart = 0.01 * i
        val trainingPart = 1.0 - testingPart
        println(testingPart, ' ', trainingPart)
        val Array(training, test) = convertedDF.randomSplit(Array(trainingPart, testingPart))
        val nb = new NaiveBayes()
        val model = nb.fit(training)
        val predicted = model.transform(test)
        predicted.show()
//        val errorsTableDF = predicted.groupBy($"label", $"prediction").count()
        val correctPredictions = predicted.filter("label = prediction").count().toDouble
        val totalPredictions = predicted.count().toDouble
        writer.println(testingPart, trainingPart, correctPredictions / totalPredictions)
      }
      writer.close()
    }

    val rowPlotDataFrame = spark.read.text(resultsPath)

    val plotDataFrame = rowPlotDataFrame.map(row => {
      val cols = row.getString(0).drop(1).dropRight(1).split(',')
      (cols(1).toDouble, cols(2).toDouble)
    })
    val namedPlotDF = plotDataFrame
      .withColumnRenamed("_1", "training_part")
      .withColumnRenamed("_2", "accuracy")

    namedPlotDF.printSchema()
    namedPlotDF.show()

    val plotFile = new File(plotPath)
    if (plotFile.exists()) {
      plotFile.delete()
    }
    val writer = new PrintWriter(plotFile)

    writer.println(
      Vegas("description")
        .withDataFrame(namedPlotDF)
        .mark(Area)
        .encodeX("training_part", Nom)
        .encodeY("accuracy")
        .html.pageHTML()
    )

    writer.close()
  }
}
