package Lab1

import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.udf
import vegas.sparkExt.VegasSpark
import vegas.{Area, Nom, Vegas}

import java.io.{File, PrintWriter}

class Spam {
  def execute(): Unit = {
    val resultsPath = "src/main/resources/lab1/spam/results.txt"
    val dataPath = "src/main/resources/lab1/spam/spam.csv"
    val plotPath = "src/main/resources/lab1/spam/plot.html"

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
        .option("header", "true")
        .option("escape", "\"")
        .csv(dataPath)
        .drop($"_c0")

      dataFrame.printSchema()
      dataFrame.show()

      val featureCols = dataFrame.columns.dropRight(1)

      println(featureCols.mkString("Array(", ", ", ")"))
      var castedDF = dataFrame
      for (featureCol <- featureCols) {
        castedDF = castedDF.withColumn(featureCol, dataFrame(featureCol).cast("Double"))
      }

      castedDF.printSchema()
      castedDF.show(10, truncate = false)

      val vectorAssembler = new VectorAssembler()
        .setInputCols(featureCols)
        .setOutputCol("features")

      val featureDF = vectorAssembler
        .transform(castedDF)

      featureDF.printSchema()
      featureDF.show()

      val labelUDF = udf((labelStr: String) => {
        labelStr match {
          case "spam" => 1.0
          case "nonspam" => 0.0
        }
      })

      val labeledDF = featureDF.select($"features", labelUDF($"type").as("label"))
      labeledDF.printSchema()
      labeledDF.show(10, truncate = false)

      val writer = new PrintWriter(resultsFile)

      for (i <- 1 to 99 by 1) {
        val testingPart = 0.01 * i
        val trainingPart = 1.0 - testingPart
        println(testingPart, ' ', trainingPart)
        val Array(training, test) = labeledDF.randomSplit(Array(trainingPart, testingPart))
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
