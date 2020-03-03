import configparser
from datetime import datetime
import os
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql import Window
from pyspark.sql.functions import row_number, desc, col, when, udf
from pyspark.sql.functions import year, month, dayofmonth, hour, weekofyear, date_format, from_unixtime, dayofweek
from pyspark.sql.functions import monotonically_increasing_id
from pyspark.sql.types import StructType as R, StructField as Fld, DoubleType as Dbl, StringType as Str, IntegerType as Int, DateType as Dat, TimestampType, LongType


config = configparser.ConfigParser()
config.read('dl.cfg')

os.environ['AWS_ACCESS_KEY_ID']=config['AWS']['AWS_ACCESS_KEY_ID']
os.environ['AWS_SECRET_ACCESS_KEY']=config['AWS']['AWS_SECRET_ACCESS_KEY']


def create_spark_session():
    
    """
    Create or retrieve a Spark Session
    """
    spark = SparkSession \
        .builder \
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:2.7.0") \
        .getOrCreate()
    return spark


def process_song_data(spark, input_data, output_data):
    
    """
    Extract song data from S3 bucket.
    """
    
    # get filepath to song data file
    song_data = input_data + 'song_data/*/*/*/*.json'
    
    # read song data file
    
    songSchema = R([
        Fld("artist_id",Str()),
        Fld("artist_latitude",Dbl()),
        Fld("artist_location",Str()),
        Fld("artist_longitude",Dbl()),
        Fld("artist_name",Str()),
        Fld("duration",Dbl()),
        Fld("num_songs",Int()),
        Fld("song_id",Str()),
        Fld("title",Str()),
        Fld("year",Int()),
    ])
    
    song_df = spark.read.json(song_data, schema=songSchema)
    
    # replace 0 values in years with None
    def replace(x):
        return when(col(x) != 0, col(x)).otherwise(None)
    
    song_df = song_df.withColumn("year", replace("year"))
    
    # replace blank/empty values with Null
    def blank_as_null(x):
        return when(col(x) != "", col(x)).otherwise(None)
    
    song_df = song_df.withColumn("artist_location", blank_as_null("artist_location"))

    # extract columns to create songs table
    songs_table = song_df.select('song_id', 'artist_id', 'year', 'duration', 'title')
    
    # write songs table to parquet files partitioned by year and artist
    songs_table.write.partitionBy("year", "artist_id").parquet(output_data + "songs.parquet")

    # extract columns to create artists table
    
    artist_spec = Window.partitionBy("artist_id").orderBy(desc("year"))
    
    artist_partition = song_df.selectExpr('artist_id', 'artist_name as name', 'artist_location as location', 'artist_longitude as longitude',\
                                          'artist_latitude as latitude', 'year').withColumn("row_number", row_number().over(artist_spec))\
    .where(col('row_number') == 1)
    
    artist_table = artist_partition.selectExpr('artist_id as ID', 'name', 'location', 'longitude', 'latitude')
    
    # write artists table to parquet files
    
    artist_table.write.parquet(output_data + "artists.parquet")


def process_log_data(spark, input_data, output_data):
    
    """
    Extract events and log data from S3 bucket.
    """
    
    # get filepath to log data file
    
    log_data = "log_data/*/*/*.json"

    # read log data file
    
    log_df = spark.read.json(input_data + log_data)
    
    # filter by actions for song plays
    
    log_df = log_df.where(col('page') == 'NextSong')

    # extract columns for users table   
    
    user_spec = Window.partitionBy("userId").orderBy(desc("ts"))
    
    user_partition = log_df.selectExpr('userId', 'firstName', 'lastName', 'gender', 'level', 'ts')\
    .withColumn("row_number", row_number().over(user_spec))\
    .where(col('row_number') == 1)
    
    users_table = user_partition.selectExpr('userId', 'firstName', 'lastName', 'gender', 'level')
    
    
    # write users table to parquet files
    users_table.write.parquet(output_data + 'users.parquet')

    # create timestamp column from original timestamp column
    
    log_df = log_df.withColumn("start_time", from_unixtime(col("ts")/1000, 'yyyy-MM-dd HH:mm:ss.SS').cast("timestamp"))
    
    # extract columns to create time table
    
    time_table = log_df.selectExpr('start_time')\
    .dropDuplicates().withColumn('week', weekofyear(col('start_time'))).withColumn('year', year(col('start_time')))\
    .withColumn('month', month(col('start_time'))).withColumn('day', dayofmonth(col('start_time')))\
    .withColumn('Day_of_Week', dayofweek(col('start_time')))\
    .withColumn('Hour', hour(col('start_time')))
    
    # write time table to parquet files partitioned by year and month
    
    time_table.write.partitionBy('year', 'month').parquet(output_data + 'time.parquet')

    # create a songplay_id from the log_df
    log_df = log_df.withColumn("songplay_id", monotonically_increasing_id()) 

    # extract columns from joined song and log datasets to create songplays table 
    #JOIN is based on these columns #song == title; artist == artist_name; duration == length; start_time == start_time
    
    song_df.createOrReplaceTempView("songs_log")
    log_df.createOrReplaceTempView("evts_log")
    time_table.createOrReplaceTempView("time_log")
    
    songplays_table = spark.sql("""
    select e.songplay_id, 
    e.start_time, e.userId, e.level, 
    s.song_id, s.artist_id, e.sessionId, 
    e.location, e.userAgent, t.year, t.month 
    from songs_log s INNER JOIN evts_log e ON s.title = e.song 
    AND s.artist_name = e.artist AND s.duration = e.length 
    INNER JOIN time_log t ON t.start_time = e.start_time""")
    
    # write songplays table to parquet files partitioned by year and month
    
    songplays_table.write.partitionBy('year', 'month').parquet(output_data + 'songplays.parquet')


def main():
    spark = create_spark_session()
    input_data = "s3a://udacity-dend/"
    output_data = "s3a://sparkify-dend4/Adeniyi/"
    
    process_song_data(spark, input_data, output_data)    
    process_log_data(spark, input_data, output_data)


if __name__ == "__main__":
    main()
