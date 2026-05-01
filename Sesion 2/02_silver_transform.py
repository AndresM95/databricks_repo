from pyspark.sql import functions as F
from pyspark.sql import SparkSession

#--------------------------------------------
# configuración
#--------------------------------------------

CATALOG = "airline_mantenimiento"
BRONZE_SCHEMA = "bronze"
SILVER_SCHEMA = "silver"
VOLUME_NAME = "landing"

spark = SparkSession.builder.getOrCreate()


#-------------------------------------------
# 0. Asegurar Schema Silver
#--------------------------------------------

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SILVER_SCHEMA}")

#-------------------------------------------
# 1. Leer Bronze
#--------------------------------------------

# leemos las tablas bronze
vuelos_bz = spark.table(f"{CATALOG}.{BRONZE_SCHEMA}.vuelos")
aeronaves_bz = spark.table(f"{CATALOG}.{BRONZE_SCHEMA}.aeronaves")
aeropuertos_bz = spark.table(f"{CATALOG}.{BRONZE_SCHEMA}.aeropuertos")
mantenimientos_bz = spark.table(f"{CATALOG}.{BRONZE_SCHEMA}.mantenimientos")

#-------------------------------------------
# 2. Silver Vuelos
#--------------------------------------------

#Transformaciones silver vuelos

vuelos_slv = (
    vuelos_bz
    .select("vuelo_id", 
            "fecha",
            "origen_id", 
            "destino_id", 
            "aeronave_id", 
            "estado", 
            "duracion_min"
            )
    .withColumn("fecha", F.to_date(F.col("fecha")))  # Convertir a fecha
    .withColumn("duracion_min", F.col("duracion_min").cast("int"))  # Convertir a entero
    .dropDuplicates(["vuelo_id"])
)

aeropuertos_slv = (
    aeropuertos_bz
    .select(
        "aeropuerto_id",
        "nombre",
        "ciudad",
        "pais",
        "lat",
        "lon"
    )
    .withColumn("lat", F.col("lat").cast("double"))
    .withColumn("lon", F.col("lon").cast("double"))
    .filter(F.col("lat").isNotNull() & F.col("lon").isNotNull())
    .filter(
        (F.col("lat") >= -90) & (F.col("lat") <= 90) &
        (F.col("lon") >= -180) & (F.col("lon") <= 180)
    )
    .dropDuplicates(["aeropuerto_id"])
)

aeronaves_slv = (
    aeronaves_bz
    .select(
        "aeronave_id",
        "modelo",
        "fabricante",
        "anio_fabricacion"
    )
    .withColumn("anio_fabricacion", F.col("anio_fabricacion").cast("int"))
    .filter(F.col("anio_fabricacion").isNotNull())
    .dropDuplicates(["aeronave_id"])
)

mantenimientos_slv = (
    mantenimientos_bz
    .select(
        "mantenimiento_id",
        "aeronave_id",
        "fecha",
        "tipo",
        "costo_usd",
        "duracion_hr"
    )
    .withColumn("fecha", F.to_date(F.col("fecha")))
    .withColumn("costo_usd", F.col("costo_usd").cast("double"))
    .withColumn("duracion_hr", F.col("duracion_hr").cast("double"))
    .dropDuplicates(["mantenimiento_id"])
)

#-------------------------------------------
# 3. Validacion de calidad
#--------------------------------------------

# Validación de Calidad
print("Nulos vuelo_id:", vuelos_slv.filter(F.col("vuelo_id").isNull()).count())

print("Duplicados vuelos:", vuelos_slv.groupBy("vuelo_id").count().filter(F.col("count") > 1).count())

#-------------------------------------------
# 4. Guardar en Silver
#--------------------------------------------

# guardar tablas silver
vuelos_slv.write.format("delta")\
    .mode("overwrite")\
    .option("overwriteSchema", True)\
    .saveAsTable(f"{CATALOG}.{SILVER_SCHEMA}.vuelos")

aeropuertos_slv.write.format("delta")\
    .mode("overwrite")\
    .option("overwriteSchema", True)\
    .saveAsTable(f"{CATALOG}.{SILVER_SCHEMA}.aeropuertos")

aeronaves_slv.write.format("delta")\
    .mode("overwrite")\
    .option("overwriteSchema", True)\
    .saveAsTable(f"{CATALOG}.{SILVER_SCHEMA}.aeronaves")

mantenimientos_slv.write.format("delta")\
    .mode("overwrite")\
    .option("overwriteSchema", True)\
    .saveAsTable(f"{CATALOG}.{SILVER_SCHEMA}.mantenimientos")


print("Silver Delta completada")
print(f"Tablas creadas: {CATALOG}.{SILVER_SCHEMA}.vuelos, aeronaves, aeropuertos, mantenimientos")