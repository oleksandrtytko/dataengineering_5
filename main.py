from pyspark.sql import SparkSession
from pyspark.sql.types import StringType
from pyspark.sql.functions import regexp_replace, col, date_format, count, max, date_add, row_number, desc, avg, to_date, date_sub, udf, lit
from pyspark.sql.window import Window
from datetime import datetime, timedelta
from pathlib import Path
import zipfile


def map_user_type(usertype):
    if usertype == "member":
        return "Subscriber"
    else:
        return "Customer"


def rename_columns(df, m):
    for t in m:
        df = df.withColumn(t[0], df[t[1]]).drop(t[1])
    return df


def delete_columns(df, l):
    for c in l:
        df = df.drop(c)
    return df


def set_null_columns(df, l):
    for c in l:
        df = df.withColumn(c, lit(None))
    return df


def read_data(spark, file_path):
    archive = zipfile.ZipFile(file_path)
    csv_str = str(archive.read(Path(file_path).stem + '.csv'), 'utf-8')
    return spark.read.csv(spark.sparkContext.parallelize(csv_str.splitlines()), header=True)


def write_answer(task_num, df):
    path = "/app/data/report"
    df.coalesce(1).write.option("header", "true").csv(path + "/task_" + str(task_num))


def main():
    spark = SparkSession.builder.appName("Exercise6").enableHiveSupport().getOrCreate()

    df = read_data(spark, "/app/data/Divvy_Trips_2019_Q4.zip")

    df_2020 = read_data(spark, "/app/data/Divvy_Trips_2020_Q1.zip")

    df_2020 = set_null_columns(delete_columns(rename_columns(df_2020, [
        ("trip_id", "ride_id"),
        ("start_time", "started_at"),
        ("from_station_name", "start_station_name"),
        ("from_station_id", "start_station_id"),
        ("to_station_name", "end_station_name"),
        ("to_station_id", "end_station_id")
    ]), ["start_lat", "start_lng", "end_lat", "end_lng"]), ["bikeid", "gender", "birthyear"])

    map_user_f = udf(lambda x: map_user_type(x), StringType())

    df_2020 = df_2020.withColumn("usertype", map_user_f(df_2020["member_casual"])).drop("member_casual")
    df_2020 = df_2020.withColumn("tripduration", lit(None)).drop("rideable_type")

    df = df.union(df_2020)
    df = df.withColumn('tripduration', regexp_replace(col("tripduration"), ",", "").cast("double"))

    df_grouped_by_date = df.groupBy(date_format("start_time", "yyyy-MM-dd").alias("start_date"))
    write_answer(1, df_grouped_by_date.agg(avg("tripduration").alias("avg")).agg(avg("avg").alias("trip_duration_average")))#task 1

    write_answer(2, df_grouped_by_date.agg(count("*").alias("count")).orderBy("start_date"))#task 2

    df_grouped_by_date_and_station = df.groupBy(date_format("start_time", "yyyy-MM").alias("start_date_month"), "from_station_name").agg(count("*").alias("count"))
    write_answer(3, df_grouped_by_date_and_station.groupBy("start_date_month").agg(max("count").alias("count"))\
        .join(df_grouped_by_date_and_station, on=["count", "start_date_month"])\
        .drop("count")\
        .orderBy("start_date_month"))#task 3

    df_converted_date = df.withColumn("start_date", date_format("start_time", "yyyy-MM-dd")).withColumn("start_time", to_date(df["start_time"], "yyyy-MM-dd HH:mm:SS"))
    df_converted_date_grouped = df_converted_date.filter(df_converted_date["start_date"] >= (datetime.strptime(df_converted_date.selectExpr("max(start_date)").collect()[0][0], "%Y-%m-%d") - timedelta(weeks=2)).strftime('%Y-%m-%d'))\
        .groupBy("start_date", "from_station_name")\
        .agg(count("*").alias("count"))
    df_converted_date_grouped_sorted = df_converted_date_grouped.orderBy(desc("count"))
    window_spec = Window.partitionBy("start_date").orderBy(desc("count"))
    df_converted_date_grouped_sorted_ranked = df_converted_date_grouped_sorted.withColumn("rank", row_number().over(window_spec))
    write_answer(4, df_converted_date_grouped_sorted_ranked.filter(col("rank") <= 3).orderBy("start_date").drop("rank").drop("count"))#task 4

    write_answer(5, df.groupBy("gender").agg(avg("tripduration").alias("average")).filter(col("gender").isNotNull()).orderBy("average", ascending=False).drop("average").limit(1)) #5task

    df_average_tripduration_by_birthyear = df.filter(col("birthyear").isNotNull()).groupBy("birthyear").agg(avg("tripduration").alias("average"))
    write_answer(6.1, df_average_tripduration_by_birthyear.orderBy("average").limit(10).select(col("birthyear").alias("birthyear_min_duration")))
    write_answer(6.2, df_average_tripduration_by_birthyear.orderBy("average", ascending=False).limit(10).select(col("birthyear").alias("birthyear_max_duration")))#6 task


if __name__ == "__main__":
    main()
