package Lab1

import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.udf
import vegas.sparkExt.VegasSpark
import vegas.{Area, Layer, Line, Nom, Vegas}

import java.io.{File, PrintWriter}

class NormalDistribution {
  def execute(): Unit = {
    val randGuessPath = "src/main/resources/lab1/normal_distribution/random_guess.txt"
    val resultsPath = "src/main/resources/lab1/normal_distribution/results.txt"
    val dataPath = "src/main/resources/lab1/normal_distribution/points.csv"
    val plotPathROC = "src/main/resources/lab1/normal_distribution/plotROC.html"
    val plotPathPR = "src/main/resources/lab1/normal_distribution/plotPR.html"

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

      dataFrame.printSchema()
      dataFrame.show()

      val castedDF = dataFrame
        .withColumn("x1", dataFrame.col("x1").cast("Double"))
        .withColumn("x2", dataFrame.col("x2").cast("Double"))
        .withColumn("class", dataFrame.col("class").cast("Int"))

      val vectorAssembler = new VectorAssembler()
        .setInputCols(Array("x1", "x2"))
        .setOutputCol("features")

      val featuresDF = vectorAssembler.transform(castedDF)

      val labelUDF = udf((labelClass: Int) => {
        labelClass match {
          case -1 => 0.0
          case 1 => 1.0
        }
      })

      val labeledDF = featuresDF.select($"features", labelUDF($"class").as("label"))

      labeledDF.printSchema()
      labeledDF.show(10, truncate = false)
      val Array(training, test) = labeledDF.randomSplit(Array(0.75, 0.25))

      val predictionUdfGenerator = (border: Double) => {
        udf((probabilities: Vector) => {
          if (probabilities(0) >= border) {
            0.0
          } else {
            1.0
          }
        })
      }

      val writer = new PrintWriter(resultsFile)

      for (i <- 0 to 100) {
        val border = i / 100.0
        println("Border = ", border)
        val nb = new NaiveBayes()
        val model = nb.fit(training)
        val probabilitiesDF = model.transform(test)
        val predicted = probabilitiesDF
          .withColumn(
            "prediction",
            predictionUdfGenerator
              .apply(border)(probabilitiesDF.col("probability")))
        predicted.show(20, truncate = false)

        val errorsTableDF = predicted.groupBy($"label", $"prediction").count()
        val correctPredictions = predicted.filter("label = prediction").count().toDouble
        val totalPredictions = predicted.count().toDouble

        val TP = predicted
          .filter("label = 1.0")
          .filter("prediction = 1.0")
          .count()

        val TN = predicted
          .filter("label = 0.0")
          .filter("prediction = 0.0")
          .count()

        val FP = predicted
          .filter("label = 0.0")
          .filter("prediction = 1.0")
          .count()

        val FN = predicted
          .filter("label = 1.0")
          .filter("prediction = 0.0")
          .count()

        println(TP, FP)
        println(FN, TN)

        val FPR: Double = FP.toDouble / (FP.toDouble + TN.toDouble)
        val TPR: Double = TP.toDouble / (TP.toDouble + FN.toDouble)

        val recall = TPR
        val precision = TP.toDouble / (TP.toDouble + FP.toDouble)

        println(correctPredictions / totalPredictions)
        errorsTableDF.show()

        writer.println(FPR, TPR, recall, precision)
      }
      writer.close()
    }

    val rowPlotDataFrame = spark.read.text(resultsPath)
    val rowRandomGuessPlotDataFrame = spark.read.text(randGuessPath)

    val plotDataFrame = rowPlotDataFrame.map(row => {
      val cols = row.getString(0).drop(1).dropRight(1).split(',')
      (cols(0).toDouble, cols(1).toDouble, cols(2).toDouble, cols(3).toDouble)
    })
    val randomGuessPlotDataFrame = rowRandomGuessPlotDataFrame.map(row => {
      val cols = row.getString(0).drop(1).dropRight(1).split(',')
      (cols(0).toDouble, cols(1).toDouble, cols(2).toDouble, cols(3).toDouble)
    })

    val namedPlotDF = plotDataFrame
      .withColumnRenamed("_1", "FPR")
      .withColumnRenamed("_2", "TPR")
      .withColumnRenamed("_3", "recall")
      .withColumnRenamed("_4", "precision")

    val namedRandGuessDF = randomGuessPlotDataFrame
      .withColumnRenamed("_1", "x1")
      .withColumnRenamed("_2", "y1")
      .withColumnRenamed("_3", "x2")
      .withColumnRenamed("_4", "y2")

    namedPlotDF.printSchema()
    namedPlotDF.show()

    namedRandGuessDF.printSchema()
    namedRandGuessDF.show()

    val plotFile = new File(plotPathROC)
    if (plotFile.exists()) {
      plotFile.delete()
    }
    val writerROC = new PrintWriter(plotFile)

    val l = Layer("description")
      .withDataFrame(namedPlotDF)
      .mark(Line)
      .encodeX("FPR")
      .encodeY("TPR")

    val l2 =
      Layer("description")
        .withDataFrame(namedRandGuessDF)
        .mark(Line)
        .encodeX("x1")
        .encodeY("y1")
        .encodeColor(value = "#f00")

    writerROC.println(
      Vegas
        .layered()
        .withLayers(l, l2)
        .html
        .pageHTML()
    )

    writerROC.close()

    val plotFilePR = new File(plotPathPR)
    if (plotFilePR.exists()) {
      plotFilePR.delete()
    }
    val writerPR = new PrintWriter(plotFilePR)

    val l3 = Layer("description")
      .withDataFrame(namedPlotDF)
      .mark(Line)
      .encodeX("recall")
      .encodeY("precision")

    val l4 =
      Layer("description")
        .withDataFrame(namedRandGuessDF)
        .mark(Line)
        .encodeX("x2")
        .encodeY("y2")
        .encodeColor(value = "#f00")

    writerPR.println(
      Vegas
        .layered()
        .withLayers(l3, l4)
        .html
        .pageHTML()
    )

    writerPR.close()
  }
}
