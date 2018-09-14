// Databricks notebook source


// COMMAND ----------

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.functions._
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.classification.{LogisticRegressionModel, LogisticRegressionWithLBFGS}
import org.apache.spark.sql.functions.udf

 // /FileStore/tables/2013_07___Citi_Bike_trip_data-11bea.csv

// COMMAND ----------

val sc = SparkContext

// COMMAND ----------

val sqlContext = org.apache.spark.sql.SQLContext

// COMMAND ----------

val Bike = spark.read.format("com.databricks.spark.csv")
           .option("header", "true")
           .option("inferSchema", "true")
           .option("delimiter", ",")
           .load("/FileStore/tables/2013_07___Citi_Bike_trip_data-11bea.csv")

// COMMAND ----------

Bike.printSchema()

// COMMAND ----------

Bike.show(10)

// COMMAND ----------

val BikeDF = Bike.toDF("trip_duration", "start_time", "stop_time", "start_station_id", "start_station_name", "start_station_latitude", "start_station_longitude", "end_station_id", "end_station_name", "end_station_latitude", "end_station_longitude", "bike_id", "user_type", "birth_year", "gender")

//OLD syntex//
//BikeDF.registerTempTable("bike")//

//New syntex//
BikeDF.createOrReplaceTempView("Bike_table")
spark.sql("select * from Bike_table").show(10,false)

// COMMAND ----------

// Which Route Citi Bike Ride the most?//

println("-> Route Citi Bikers ride the most:")
spark.sql("SELECT start_station_id,start_station_name,end_station_id,end_station_name, COUNT(1) AS cnt FROM bike GROUP BY 
start_station_id,start_station_name,end_station_id,end_station_name ORDER BY cnt DESC LIMIT 1").take(1).foreach(println)


// COMMAND ----------

// Find the biggest trip and It's duration?//
println("-> Biggest trip and its duration:")
def Distancal(lat1:Double, lon1:Double, lat2:Double, lon2:Double):Int = 
{
  val AVERAGE_RADIUS_OF_EARTH_KM = 6371 
  val latDistance = Math.toRadians(lat1 - lat2)
  val lngDistance = Math.toRadians(lon1 - lon2) 
  val sinLat = Math.sin(latDistance / 2) 
  val sinLng = Math.sin(lngDistance / 2)
  val a = sinLat * sinLat + (Math.cos(Math.toRadians(lat1)) * Math.cos(Math.toRadians(lat2)) * sinLng * sinLng) 
  val c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a))(AVERAGE_RADIUS_OF_EARTH_KM * c).toString
} 
spark.udf.register("Distancal", Distancal_)
spark.sql("SELECT start_station_id,start_station_name,end_station_id,end_station_name, trip_duration
,Distancal(start_station_latitude,start_station_longitude,end_station_latitude,end_station_longitude) 
AS dist FROM bike ORDER BY trip_duration DESC, dist DESC LIMIT 1").take(1).foreach(println)





// COMMAND ----------



// COMMAND ----------

// When Do they go?//
println("-> Hours of a day they Ride mostly:")
spark.sql("SELECT date_format(start_time,'H:m'), COUNT(1) as cnt FROM bike GROUP BY date_format(start_time,'H:m') ORDER BY cnt DESC LIMIT 10")
.take(10).foreach(println)

// COMMAND ----------

// How far do they go?//
println("-> Most far rides:")
spark.sql("SELECT start_station_id,start_station_name,end_station_id,end_station_name, trip_duration, 
Distancal(start_station_latitude,start_station_longitude,end_station_latitude,end_station_longitude) AS dist FROM bike ORDER BY dist DESC LIMIT 1")
.take(1).foreach(println)

// COMMAND ----------

//Which station are most popular//
println("-> most popular stations:")
var popularStationsDF = spark.sql("SELECT start_station_id as station_id,start_station_name as station_name, 
COUNT(1) as cnt FROM bike GROUP BY start_station_id,start_station_name UNION ALL SELECT end_station_id as station_id,
nd_station_name as station_name, COUNT(1) as cnt FROM bike GROUP BY end_station_id,end_station_name")
//New syntex//
popularStationsDF.createOrReplaceTempView("popularStations_table")
spark.sql("select * from popularStations_table").show(10,false)

// COMMAND ----------

//which days of the week are most rides taken ON?//
println("-> Day of the week most rides taken on:")
spark.sql("SELECT date_format(start_time,'E') AS Day,COUNT(1) AS cnt FROM bike GROUP BY date_format(start_time,'E') 
ORDER BY cnt DESC LIMIT 1").take(1).foreach(println)


// COMMAND ----------

// predicting the gender based on patterns in the trip//
val BikeRDD = BikeDF.rdd
val features = BikeRDD.map { 
  Bike =>val gender = if (Bike.getInt(14) == 1) 0.0 else if (Bike.getInt(14) == 2) 1.0 else  2.0
  val trip_duration = Bike.getInt(0).toDouble
  val start_time = Bike.getTimestamp(1).getTime.toDouble
  val stop_time = Bike.getTimestamp(2).getTime.toDouble
  val start_station_id = Bike.getInt(3).toDouble
  val start_station_latitude = Bike.getDouble(5)
  val start_station_longitude = Bike.getDouble(6)
  val end_station_id = Bike.getInt(7).toDouble
  val end_station_latitude = Bike.getDouble(9)
  val end_station_longitude = Bike.getDouble(10)
  val user_type = if (Bike.getString(12) == "Customer") 1.0 else 2.0
  Array(gender, trip_duration, start_time, stop_time, start_station_id, 
        start_station_latitude, start_station_longitude, end_station_id,      
        end_station_latitude, end_station_longitude, user_type)
    }

// COMMAND ----------

 val labeled = features.map { x => LabeledPoint(x(0), Vectors.dense(x(1), x(2), 
                  x(3), x(4), x(5), x(6), x(7), x(8), x(9), x(10)))}

// COMMAND ----------

// Split data into training and test
val training = labeled.filter(_.label != 2).randomSplit(Array(0.40, 0.60))(1)
val test = labeled.filter(_.label != 2).randomSplit(Array(0.60, 0.40))(1)

// COMMAND ----------

val test_count = test.count

// COMMAND ----------

training.cache
training.count

// COMMAND ----------

// Run training algorithm to build the model
val model = new LogisticRegressionWithLBFGS().setNumClasses(2).run(training)

// COMMAND ----------

// Compute raw scores on the test set.
val predictionAndLabels = test.map 
{ case LabeledPoint(label, features) =>
  val prediction = model.predict(features)
   (prediction, label)
    }

// COMMAND ----------

println("Predictions:")
predictionAndLabels.take(10).foreach(println)

// COMMAND ----------

val wrong = predictionAndLabels.filter 
{
    case (label, prediction) => label != prediction
    }
    val wrong_count = wrong.count
    val accuracy = 1 - (wrong_count.toDouble / test_count)
    println(s"Accuracy model1: " + accuracy)



// COMMAND ----------


