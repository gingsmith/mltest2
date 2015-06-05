package distopt

import org.apache.spark.{SparkContext, SparkConf}
import org.apache.spark.SparkContext._
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.regression.LassoWithSGD
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.ml.regression._
import org.apache.spark.ml.classification._
import org.apache.spark.sql.{SQLContext, DataFrame}
//import org.apache.spark.sql.SQLContext.implicits._

object ml_driver {

  def main(args: Array[String]) {

    val options =  args.map { arg =>
      arg.dropWhile(_ == '-').split('=') match {
        case Array(opt, v) => (opt -> v)
        case Array(opt) => (opt -> "true")
        case _ => throw new IllegalArgumentException("Invalid argument: " + arg)
      }
    }.toMap

    // read in inputs
    val master = options.getOrElse("master", "local[4]")
    val trainFile = options.getOrElse("trainFile", "")
    val numFeatures = options.getOrElse("numFeatures", "0").toInt
    val numSplits = options.getOrElse("numSplits","1").toInt
    val chkptDir = options.getOrElse("chkptDir","");
    var chkptIter = options.getOrElse("chkptIter","100").toInt
    val testFile = options.getOrElse("testFile", "")
    val justCoCoA = options.getOrElse("justCoCoA", "true").toBoolean // set to false to compare different methods
    
    // algorithm-specific inputs
    val lambda = options.getOrElse("lambda", "0.01").toDouble // regularization parameter
    val numRounds = options.getOrElse("numRounds", "200").toInt // number of outer iterations, called T in the paper
    val localIterFrac = options.getOrElse("localIterFrac","1.0").toDouble; // fraction of local points to be processed per round, H = localIterFrac * n
    val beta = options.getOrElse("beta","1.0").toDouble;  // scaling parameter when combining the updates of the workers (1=averaging)
    val debugIter = options.getOrElse("debugIter","10").toInt // set to -1 to turn off debugging output
    val seed = options.getOrElse("seed","0").toInt // set seed for debug purposes
    val linReg = options.getOrElse("linReg","true").toBoolean

    // print out inputs
    println("master:       " + master);          println("trainFile:    " + trainFile);
    println("numFeatures:  " + numFeatures);     println("numSplits:    " + numSplits);
    println("chkptDir:     " + chkptDir);        println("chkptIter     " + chkptIter);       
    println("testfile:     " + testFile);        println("justCoCoA     " + justCoCoA);       
    println("lambda:       " + lambda);          println("numRounds:    " + numRounds);       
    println("localIterFrac:" + localIterFrac);   println("beta          " + beta);     
    println("debugIter     " + debugIter);       println("seed          " + seed);
    println("linReg:       " + linReg)  

    // start spark context
    val conf = new SparkConf().setMaster(master)
      .setAppName("demoCoCoA")
      .setJars(SparkContext.jarOfObject(this).toSeq)
    val sc = new SparkContext(conf)
    if (chkptDir != "") {
      sc.setCheckpointDir(chkptDir)
    } else {
      chkptIter = numRounds + 1
    }

    val sqlContext = new SQLContext(sc)
    import sqlContext.implicits._

    // Load training data in LIBSVM format.
    val data = MLUtils.loadLibSVMFile(sc, trainFile, -1, numSplits).cache()
    val parts = data.partitions.size    // number of partitions of the data, K in the paper
    val n = data.count()
    println("\nRunning MLlib MB-SGD on "+n+" data examples, distributed over "+parts+" workers")

    if(linReg){
        // need newest version of spark to do this
        val lr = new LinearRegression()
        lr.setMaxIter(numRounds).setRegParam(lambda).setElasticNetParam(1.0).setTol(1e-10)
        // Print out the parameters, documentation, and any default values.
        println("Linear Regression parameters:\n" + lr.explainParams() + "\n")
        val model = lr.fit(data.toDF())
        ///println("Model was fit using parameters: " + model.parent.extractParamMap())
    } else{
        val lr = new LogisticRegression()
        lr.setMaxIter(numRounds).setRegParam(lambda).setElasticNetParam(1.0).setTol(1e-10).setFitIntercept(false)
        // Print out the parameters, documentation, and any default values.
        println("Logistic Regression parameters:\n" + lr.explainParams() + "\n")
        val model = lr.fit(data.toDF())
        //println("Model was fit using parameters: " + model.parent.extractParamMap())
    }


    sc.stop()
   }
}
