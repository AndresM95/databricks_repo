from pyspark.sql import functions as F
from pyspark.sql import SparkSession

#--------------------------------------------
# configuración
#--------------------------------------------

CATALOG = "airline_mantenimiento"
SILVER_SCHEMA = "silver"
GOLD_SCHEMA = "gold"
VOLUME_NAME = "landing"

spark = SparkSession.builder.getOrCreate()

#-------------------------------------------
# 0. Asegurar Schema Gold
#--------------------------------------------

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{GOLD_SCHEMA}")

#-------------------------------------------
# 1. Leer Silver
#--------------------------------------------

# leer silver para puntualidad
df_vuelos = spark.table(f"{CATALOG}.{SILVER_SCHEMA}.vuelos")
df_aeropuertos = spark.table(f"{CATALOG}.{SILVER_SCHEMA}.aeropuertos")
df_aeronaves = spark.table(f"{CATALOG}.{SILVER_SCHEMA}.aeronaves")

#-------------------------------------------
# 2. Preparar vuelos 
#--------------------------------------------

# preparas aeropuertos de origen
df_aeropuertos_origen = df_aeropuertos.select(
    F.col("aeropuerto_id").alias("origen_id"),
    F.col("nombre").alias("origen_nombre"),
    F.col("ciudad").alias("origen_ciudad"),
    F.col("pais").alias("origen_pais")
)

#-------------------------------------------
# 3. Enriquecer vuelos 
#--------------------------------------------

df_vuelos_enriquecido = (
    df_vuelos
    .join(df_aeropuertos_origen, on="origen_id", how="left")
    .join(df_aeronaves, on="aeronave_id", how="left")
    .withColumn("anio", F.year("fecha"))
    .withColumn("mes", F.month("fecha"))
)

#-------------------------------------------
# 4. Calcular KPI' de puntualidad
#--------------------------------------------

#calcular KPI de puntualidad
df_kpi_puntualidad = (
    df_vuelos_enriquecido
    .groupBy("modelo", "fabricante", "origen_pais", "mes", "anio")
    .agg(
        F.count("vuelo_id").alias("total_vuelos"),
        F.count(F.when(F.col("estado") == "a_tiempo", True)).alias("vuelos_a_tiempo"),
        F.count(F.when(F.col("estado") == "retrasado", True)).alias("vuelos_retrasados"),
        F.count(F.when(F.col("estado") == "cancelado", True)).alias("vuelos_cancelados"),
        F.round(F.avg("duracion_min"), 2).alias("duracaion_promedio_min")
    )
.withColumn("puntualidad_pct", F.round((F.col("vuelos_a_tiempo") / F.col("total_vuelos"))*100, 2))
)

#-------------------------------------------
# 5. Guardar Gold
#--------------------------------------------

df_kpi_puntualidad .write.format("delta")\
    .mode("overwrite")\
    .option("overwriteSchema", True)\
    .saveAsTable(f"{CATALOG}.{GOLD_SCHEMA}.kpi_puntualidad_vuelos")

#-------------------------------------------
# 6. Capa Gold - KPI de costos de mantenimiento
#--------------------------------------------

df_mantenimiento = spark.table(f"{CATALOG}.{SILVER_SCHEMA}.mantenimientos")

df_mantenimientos_enriquecido = (
    df_mantenimiento
    .join(df_aeronaves, on="aeronave_id", how="left")
    .withColumn("anio", F.year("fecha"))
    .withColumn("mes", F.month("fecha"))
)

# Calcular KPI de mantenimiento
df_kpi_mantenimiento = (
    df_mantenimientos_enriquecido
    .groupBy("aeronave_id", "modelo", "fabricante", "tipo", "mes", "anio")
    .agg(
        F.count("mantenimiento_id").alias("total_mantenimientos"),
        F.round(F.avg("costo_usd"), 2).alias("costo_promedio_usd"),
        F.round(F.sum("costo_usd"), 2).alias("costo_total_usd"),
        F.round(F.avg("duracion_hr"), 2).alias("duracion_promedio_hr"),
        F.round(F.sum("duracion_hr"), 2).alias("duracion_total_hr")
    )
)

# Guardar Gold Mantenimiento
df_kpi_mantenimiento.write.format("delta")\
    .mode("overwrite")\
    .option("overwriteSchema", True)\
    .saveAsTable(f"{CATALOG}.{GOLD_SCHEMA}.kpi_costo_mantenimiento")