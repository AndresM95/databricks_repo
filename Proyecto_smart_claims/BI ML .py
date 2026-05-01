# Databricks notebook source
# DBTITLE 1,Cell 1
# ============================================================
# REGLAS DE NEGOCIO Y BI — Smart Claims
# ============================================================
 
from pyspark.sql import functions as F
from pyspark.sql import types as T
 
CATALOG       = "proyecto_smart_claims"
SILVER_SCHEMA = "silver"
GOLD_SCHEMA   = "gold"
 
# ------------------------------------------------------------
# PASO A: Cargar todas las tablas necesarias
# ------------------------------------------------------------
print("=" * 60)
print("PASO A: CARGANDO TABLAS")
print("=" * 60)
 
claims_clean    = spark.table(f"{CATALOG}.{SILVER_SCHEMA}.claims_clean")
policies_clean  = spark.table(f"{CATALOG}.{SILVER_SCHEMA}.policies_clean")
customers_clean = spark.table(f"{CATALOG}.{SILVER_SCHEMA}.customers_clean")
metadata        = spark.table(f"{CATALOG}.{SILVER_SCHEMA}.claim_images_metadata_clean")

# Check if predictions table exists
try:
    predicciones = spark.table(f"{CATALOG}.{GOLD_SCHEMA}.claim_images_predictions")
    predictions_available = True
    print(f"  ✅ claim_images_predictions: {predicciones.count():,} filas")
except Exception:
    predictions_available = False
    print(f"  ⚠️  claim_images_predictions: tabla no disponible (se continuará sin predicciones)")
 
for nombre, df_tmp in [
    ("claims_clean",                claims_clean),
    ("policies_clean",              policies_clean),
    ("customers_clean",             customers_clean),
    ("claim_images_metadata_clean", metadata),
]:
    print(f"  ✅ {nombre}: {df_tmp.count():,} filas")
 
# ------------------------------------------------------------
# PASO B: Construir tabla base integrada
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("PASO B: INTEGRANDO TABLAS")
print("=" * 60)
 
# Agregar predicción por claim_no (mejor score por reclamo)
if predictions_available:
    pred_por_claim = (
        predicciones
        .join(metadata.select("image_name", "claim_no", "chassis_no"),
              on="image_name", how="left")
        .filter(F.col("claim_no").isNotNull())
        .groupBy("claim_no", "chassis_no")
        .agg(
            F.first("damage_label",
                    ignorenulls=True).alias("damage_label"),
            F.round(F.max("damage_score"), 4).alias("damage_score"),
            F.first("confidence_level",
                    ignorenulls=True).alias("confidence_level"),
            F.sum(F.col("is_reliable").cast("int")).alias("imagenes_confiables"),
            F.count("*").alias("total_imagenes"),
        )
    )
else:
    # Create empty predictions with schema when table doesn't exist
    pred_por_claim = spark.createDataFrame(
        [],
        T.StructType([
            T.StructField("claim_no", T.StringType(), True),
            T.StructField("chassis_no", T.StringType(), True),
            T.StructField("damage_label", T.StringType(), True),
            T.StructField("damage_score", T.DoubleType(), True),
            T.StructField("confidence_level", T.StringType(), True),
            T.StructField("imagenes_confiables", T.LongType(), True),
            T.StructField("total_imagenes", T.LongType(), True),
        ])
    )
 
# Join principal: claims + policies + customers + predicción
tabla_base = (
    claims_clean
    .join(policies_clean,
          claims_clean["policy_no"] == policies_clean["POLICY_NO"],
          how="left")
    .join(customers_clean,
          policies_clean["CUST_ID"] == customers_clean["customer_id"],
          how="left")
    .join(pred_por_claim,
          on="claim_no", how="left")
    .select(
        # Reclamo
        claims_clean["claim_no"],
        claims_clean["policy_no"],
        claims_clean["claim_date"],
        claims_clean["claim_date"].alias("incident_date"),
        claims_clean["total"].alias("claim_total"),
        claims_clean["type"].alias("incident_type"),
        claims_clean["collision_type"],
        claims_clean["severity"].alias("incident_severity"),
        F.lit(None).cast("string").alias("authorities_contacted"),
        claims_clean["hour"].alias("incident_hour"),
        claims_clean["number_of_vehicles_involved"],
        claims_clean["injury"].alias("bodily_injuries"),
        claims_clean["number_of_witnesses"].alias("witnesses"),
        F.lit(None).cast("string").alias("police_report_available"),
        policies_clean["MAKE"].alias("auto_make"),
        policies_clean["MODEL"].alias("auto_model"),
        policies_clean["MODEL_YEAR"].alias("auto_year"),
        policies_clean["CHASSIS_NO"].alias("chassis_no"),
 
        # Póliza
        policies_clean["PREMIUM"].alias("premium"),
        policies_clean["POL_EFF_DATE"].alias("policy_start"),
        policies_clean["POL_EXPIRY_DATE"].alias("policy_end"),
 
        # Cliente
        customers_clean["first_name"],
        customers_clean["last_name"],
        customers_clean["date_of_birth"],
 
        # Predicción del modelo
        F.col("damage_label"),
        F.col("damage_score"),
        F.col("confidence_level"),
        F.col("imagenes_confiables"),
        F.col("total_imagenes"),
    )
)
 
print(f"✅ Tabla base integrada: {tabla_base.count():,} filas")
 
# ------------------------------------------------------------
# PASO C: Aplicar reglas de negocio
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("PASO C: APLICANDO REGLAS DE NEGOCIO")
print("=" * 60)
 
tabla_con_reglas = (
    tabla_base
 
    # ── REGLA 1: Póliza vigente al momento del incidente ─────
    .withColumn(
        "r1_poliza_vigente",
        F.when(
            F.col("incident_date").isNull() |
            F.col("policy_start").isNull() |
            F.col("policy_end").isNull(),
            F.lit(True)
        ).otherwise(
            (F.col("incident_date") >= F.col("policy_start")) &
            (F.col("incident_date") <= F.col("policy_end"))
        )
    )
 
    # ── REGLA 2: Monto del reclamo razonable ─────────────────
    .withColumn(
        "r2_monto_razonable",
        F.when(
            F.col("claim_total").isNull() | F.col("premium").isNull(),
            F.lit(True)
        ).otherwise(
            F.col("claim_total") <= (F.col("premium") * 3)
        )
    )
 
    # ── REGLA 3: Reporte policial ─────────────────────────────
    # No disponible en los datos — siempre pasa
    .withColumn("r3_reporte_policial", F.lit(True))
 
    # ── REGLA 4: Predicción confiable del modelo ──────────────
    .withColumn(
        "r4_prediccion_confiable",
        F.when(
            F.col("confidence_level").isNull(),
            F.lit(True)   # sin predicción no penaliza
        ).otherwise(
            F.col("confidence_level").isin("alta", "media")
        )
    )
 
    # ── REGLA 5: Coherencia severidad reportada vs predicha ───
    # CORRECCIÓN: manejo explícito de nulos para evitar null en ~
    .withColumn(
        "r5_coherencia_severidad",
        F.when(
            F.col("damage_label").isNull() | F.col("incident_severity").isNull(),
            F.lit(True)   # sin predicción no podemos determinar incoherencia
        ).otherwise(
            ~(
                (F.col("damage_label").isin("scratch", "dent")) &
                (F.col("incident_severity").isin("Major Damage", "Total Loss"))
            )
        )
    )
 
    # ── REGLA 6: Reclamo dentro del período razonable ─────────
    # No disponible — incident_date = claim_date, siempre pasa
    .withColumn("r6_tiempo_razonable", F.lit(True))
 
    # ── Conteo de reglas fallidas ─────────────────────────────
    # CORRECCIÓN: coalesce para tratar nulos como 0 y evitar null en la suma
    .withColumn(
        "reglas_fallidas",
        F.coalesce((F.col("r1_poliza_vigente")       == False).cast("int"), F.lit(0)) +
        F.coalesce((F.col("r2_monto_razonable")      == False).cast("int"), F.lit(0)) +
        F.coalesce((F.col("r3_reporte_policial")     == False).cast("int"), F.lit(0)) +
        F.coalesce((F.col("r4_prediccion_confiable") == False).cast("int"), F.lit(0)) +
        F.coalesce((F.col("r5_coherencia_severidad") == False).cast("int"), F.lit(0)) +
        F.coalesce((F.col("r6_tiempo_razonable")     == False).cast("int"), F.lit(0))
    )
 
    # ── DECISIÓN FINAL ────────────────────────────────────────
    .withColumn(
        "decision_final",
        F.when(F.col("reglas_fallidas") == 0, "LIBERADO")
         .when(F.col("reglas_fallidas") == 1, "REVISION_MENOR")
         .otherwise("INVESTIGAR")
    )
 
    # ── Motivos de revisión ───────────────────────────────────
    .withColumn(
        "motivos_revision",
        F.concat_ws(" | ",
            F.when(F.col("r1_poliza_vigente")       == False, F.lit("Póliza no vigente")),
            F.when(F.col("r2_monto_razonable")      == False, F.lit("Monto excesivo")),
            F.when(F.col("r3_reporte_policial")     == False, F.lit("Sin reporte policial")),
            F.when(F.col("r4_prediccion_confiable") == False, F.lit("Predicción no confiable")),
            F.when(F.col("r5_coherencia_severidad") == False, F.lit("Severidad incoherente")),
            F.when(F.col("r6_tiempo_razonable")     == False, F.lit("Tiempo excedido")),
        )
    )
)
 
print("✅ Reglas de negocio aplicadas:")
print("   R1: Póliza vigente al momento del incidente")
print("   R2: Monto del reclamo <= 3x el premium")
print("   R3: Reporte policial (no disponible en datos — siempre pasa)")
print("   R4: Predicción del modelo confiable")
print("   R5: Coherencia entre severidad reportada y predicha")
print("   R6: Reclamo dentro de 90 días (no disponible — siempre pasa)")
 
print("\nDistribución de decisiones:")
tabla_con_reglas.groupBy("decision_final") \
    .count().orderBy("decision_final").show()
 
# ------------------------------------------------------------
# PASO D: Guardar tabla final en Gold
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("PASO D: GUARDANDO TABLA FINAL EN GOLD")
print("=" * 60)
 
TABLA_FINAL = f"{CATALOG}.{GOLD_SCHEMA}.claim_insights_final"
 
(
    tabla_con_reglas
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(TABLA_FINAL)
)
 
total_final = tabla_con_reglas.count()
print(f"✅ {TABLA_FINAL}")
print(f"   {total_final:,} reclamos enriquecidos guardados.")

# COMMAND ----------

# ============================================================
# BI — Visualizaciones Smart Claims
# ============================================================
 
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import numpy as np
import warnings
warnings.filterwarnings("ignore")
 
# Paleta corporativa completa
COLORS = {
    "azul":      "#1F4E79",
    "azul_med":  "#2E75B6",
    "azul_cla":  "#9DC3E6",
    "verde":     "#375623",
    "verde_med": "#70AD47",
    "amarillo":  "#F4B942",
    "rojo":      "#C00000",
    "gris":      "#595959",
    "gris_cla":  "#D9D9D9",
}
 
# Cargar tabla final
df = spark.table(TABLA_FINAL).toPandas()
print(f"✅ Datos cargados: {len(df):,} reclamos")
print(f"   Columnas: {list(df.columns)}")
 

# COMMAND ----------

 
# ============================================================
# BLOQUE A — VISTA DE NEGOCIO
# ============================================================
 
fig = plt.figure(figsize=(20, 16))
fig.patch.set_facecolor("white")
 
gs = gridspec.GridSpec(2, 3, figure=fig,
                       hspace=0.45, wspace=0.35)
 
fig.suptitle(
    "SMART CLAIMS — BLOQUE A: VISTA DE NEGOCIO",
    fontsize=18, fontweight="bold",
    color=COLORS["azul"], y=0.98
)
 
# ── A1: Reclamos por severidad ──────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
 
sev_order  = ["Minor Damage", "Major Damage", "Total Loss", "Trivial Damage"]
sev_colors = [COLORS["amarillo"], COLORS["azul_med"],
              COLORS["rojo"],     COLORS["verde_med"]]
 
sev_counts = (df["incident_severity"]
              .value_counts()
              .reindex(sev_order, fill_value=0))
 
bars = ax1.bar(range(len(sev_counts)), sev_counts.values,
               color=sev_colors, edgecolor="white",
               linewidth=1.5, width=0.6)
 
for bar, val in zip(bars, sev_counts.values):
    ax1.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 0.3,
             f"{val:,}", ha="center", va="bottom",
             fontsize=10, fontweight="bold",
             color=COLORS["gris"])
 
ax1.set_xticks(range(len(sev_counts)))
ax1.set_xticklabels(
    [s.replace(" ", "\n") for s in sev_counts.index],
    fontsize=9
)
ax1.set_title("Reclamos por Severidad",
              fontsize=12, fontweight="bold",
              color=COLORS["azul"], pad=10)
ax1.set_ylabel("Cantidad de reclamos", fontsize=9)
ax1.spines[["top", "right"]].set_visible(False)
ax1.set_facecolor("#FAFAFA")
 
# ── A2: Pérdida total por tipo de incidente ─────────────────
ax2 = fig.add_subplot(gs[0, 1])
 
perdida = (df.groupby("incident_type")["claim_total"]
             .sum()
             .sort_values(ascending=True))
 
bars2 = ax2.barh(range(len(perdida)), perdida.values,
                 color=COLORS["azul_med"],
                 edgecolor="white", linewidth=1.2)
 
for bar, val in zip(bars2, perdida.values):
    ax2.text(val + perdida.max() * 0.01,
             bar.get_y() + bar.get_height() / 2,
             f"${val/1e6:.1f}M" if val >= 1e6 else f"${val:,.0f}",
             va="center", fontsize=9,
             color=COLORS["gris"])
 
ax2.set_yticks(range(len(perdida)))
ax2.set_yticklabels(perdida.index, fontsize=9)
ax2.set_title("Pérdida Total por\nTipo de Incidente",
              fontsize=12, fontweight="bold",
              color=COLORS["azul"], pad=10)
ax2.set_xlabel("Total reclamado ($)", fontsize=9)
ax2.spines[["top", "right"]].set_visible(False)
ax2.set_facecolor("#FAFAFA")
 
# ── A3: Frecuencia por hora del incidente ───────────────────
ax3 = fig.add_subplot(gs[0, 2])
 
if "incident_hour" in df.columns:
    hora_counts = (df["incident_hour"]
                   .dropna()
                   .astype(int)
                   .value_counts()
                   .sort_index())
 
    hora_colors = [
        COLORS["azul"] if h < 6 or h >= 20
        else COLORS["amarillo"] if 6 <= h < 12
        else COLORS["azul_med"] if 12 <= h < 18
        else COLORS["verde_med"]
        for h in hora_counts.index
    ]
 
    ax3.bar(hora_counts.index, hora_counts.values,
            color=hora_colors, edgecolor="white",
            linewidth=0.8, width=0.85)
 
    ax3.axvspan(-0.5,  5.5, alpha=0.06, color=COLORS["azul"],    label="Madrugada")
    ax3.axvspan( 5.5, 11.5, alpha=0.06, color=COLORS["amarillo"], label="Mañana")
    ax3.axvspan(11.5, 17.5, alpha=0.06, color=COLORS["azul_med"], label="Tarde")
    ax3.axvspan(17.5, 23.5, alpha=0.06, color=COLORS["verde_med"],label="Noche")
 
    ax3.set_title("Frecuencia de Reclamos\npor Hora del Incidente",
                  fontsize=12, fontweight="bold",
                  color=COLORS["azul"], pad=10)
    ax3.set_xlabel("Hora del día", fontsize=9)
    ax3.set_ylabel("Cantidad", fontsize=9)
    ax3.set_xticks(range(0, 24, 3))
    ax3.legend(fontsize=7, loc="upper left")
    ax3.spines[["top", "right"]].set_visible(False)
    ax3.set_facecolor("#FAFAFA")
 
# ── A4: Top 5 marcas con mayor pérdida ──────────────────────
ax4 = fig.add_subplot(gs[1, 0])
 
top_marcas = (df.groupby("auto_make")["claim_total"]
                .sum()
                .sort_values(ascending=False)
                .head(5))
 
wedges, texts, autotexts = ax4.pie(
    top_marcas.values,
    labels=top_marcas.index,
    autopct="%1.1f%%",
    colors=[COLORS["azul"], COLORS["azul_med"], COLORS["azul_cla"],
            COLORS["amarillo"], COLORS["gris_cla"]],
    startangle=90,
    wedgeprops=dict(edgecolor="white", linewidth=2)
)
 
for at in autotexts:
    at.set_fontsize(9)
    at.set_fontweight("bold")
 
ax4.set_title("Top 5 Marcas por\nPérdida Total",
              fontsize=12, fontweight="bold",
              color=COLORS["azul"], pad=10)
 
# ── A5: Distribución del monto de reclamo ───────────────────
ax5 = fig.add_subplot(gs[1, 1])
 
claim_data = df["claim_total"].dropna()
ax5.hist(claim_data, bins=20,
         color=COLORS["azul_med"],
         edgecolor="white", linewidth=0.8)
 
ax5.axvline(claim_data.mean(), color=COLORS["rojo"],
            linewidth=2, linestyle="--",
            label=f"Media: ${claim_data.mean():,.0f}")
ax5.axvline(claim_data.median(), color=COLORS["verde_med"],
            linewidth=2, linestyle="-.",
            label=f"Mediana: ${claim_data.median():,.0f}")
 
ax5.set_title("Distribución del\nMonto de Reclamo",
              fontsize=12, fontweight="bold",
              color=COLORS["azul"], pad=10)
ax5.set_xlabel("Monto ($)", fontsize=9)
ax5.set_ylabel("Frecuencia", fontsize=9)
ax5.legend(fontsize=8)
ax5.spines[["top", "right"]].set_visible(False)
ax5.set_facecolor("#FAFAFA")
 
# ── A6: Reclamos por mes ────────────────────────────────────
ax6 = fig.add_subplot(gs[1, 2])
 
if "claim_date" in df.columns:
    df["claim_month"] = pd.to_datetime(
        df["claim_date"], errors="coerce"
    ).dt.to_period("M").astype(str)
 
    mes_counts = (df["claim_month"]
                  .dropna()
                  .value_counts()
                  .sort_index())
 
    ax6.plot(range(len(mes_counts)), mes_counts.values,
             color=COLORS["azul"], linewidth=2.5,
             marker="o", markersize=6,
             markerfacecolor=COLORS["amarillo"],
             markeredgecolor=COLORS["azul"])
    ax6.fill_between(range(len(mes_counts)),
                     mes_counts.values,
                     alpha=0.15, color=COLORS["azul_med"])
 
    ax6.set_xticks(range(len(mes_counts)))
    ax6.set_xticklabels(mes_counts.index,
                        rotation=45, ha="right", fontsize=7)
    ax6.set_title("Evolución Mensual\nde Reclamos",
                  fontsize=12, fontweight="bold",
                  color=COLORS["azul"], pad=10)
    ax6.set_ylabel("Cantidad", fontsize=9)
    ax6.spines[["top", "right"]].set_visible(False)
    ax6.set_facecolor("#FAFAFA")
 
plt.savefig("/tmp/bi_bloque_a.png",
            dpi=150, bbox_inches="tight",
            facecolor="white")
plt.show()
print("✅ Bloque A guardado.")
 

# COMMAND ----------

# ============================================================
# BLOQUE B — VISTA DE PREDICCIÓN
# ============================================================
 
fig = plt.figure(figsize=(20, 14))
fig.patch.set_facecolor("white")
gs  = gridspec.GridSpec(2, 3, figure=fig,
                        hspace=0.45, wspace=0.35)
 
fig.suptitle(
    "SMART CLAIMS — BLOQUE B: VISTA DE PREDICCIÓN",
    fontsize=18, fontweight="bold",
    color=COLORS["azul"], y=0.98
)
 
# ── B1: Distribución de predicciones del modelo ─────────────
ax1 = fig.add_subplot(gs[0, 0])
 
pred_counts = (df["damage_label"]
               .fillna("Sin predicción")
               .value_counts())
 
bar_colors = [COLORS["azul"], COLORS["azul_med"], COLORS["azul_cla"],
              COLORS["amarillo"], COLORS["gris"]][:len(pred_counts)]
 
bars = ax1.bar(range(len(pred_counts)), pred_counts.values,
               color=bar_colors, edgecolor="white",
               linewidth=1.5, width=0.6)
 
for bar, val in zip(bars, pred_counts.values):
    ax1.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 0.2,
             f"{val:,}", ha="center", va="bottom",
             fontsize=10, fontweight="bold",
             color=COLORS["gris"])
 
ax1.set_xticks(range(len(pred_counts)))
ax1.set_xticklabels(pred_counts.index, rotation=30,
                    ha="right", fontsize=9)
ax1.set_title("Distribución de Predicciones\ndel Modelo",
              fontsize=12, fontweight="bold",
              color=COLORS["azul"], pad=10)
ax1.set_ylabel("Cantidad de reclamos", fontsize=9)
ax1.spines[["top", "right"]].set_visible(False)
ax1.set_facecolor("#FAFAFA")
 
# ── B2: Distribución por nivel de confianza ─────────────────
ax2 = fig.add_subplot(gs[0, 1])
 
conf_map    = {"alta": COLORS["verde_med"],
               "media": COLORS["amarillo"],
               "baja": COLORS["rojo"]}
conf_counts = (df["confidence_level"]
               .fillna("sin datos")
               .value_counts())
conf_colors = [conf_map.get(c, COLORS["gris"])
               for c in conf_counts.index]
 
wedges, texts, autotexts = ax2.pie(
    conf_counts.values,
    labels=conf_counts.index,
    autopct="%1.1f%%",
    colors=conf_colors,
    startangle=90,
    wedgeprops=dict(edgecolor="white", linewidth=2)
)
 
for at in autotexts:
    at.set_fontsize(10)
    at.set_fontweight("bold")
 
ax2.set_title("Nivel de Confianza\nde las Predicciones",
              fontsize=12, fontweight="bold",
              color=COLORS["azul"], pad=10)
 
# ── B3: Coherencia severidad reportada vs predicha ──────────
ax3 = fig.add_subplot(gs[0, 2])
 
if "r5_coherencia_severidad" in df.columns:
    coherencia = df.copy()
    coherencia["coherente"] = np.where(
        coherencia["r5_coherencia_severidad"].fillna(True),
        "Coherente", "Incoherente"
    )
    coh_counts = coherencia["coherente"].value_counts()
    coh_colors = [COLORS["verde_med"] if c == "Coherente"
                  else COLORS["rojo"]
                  for c in coh_counts.index]
 
    bars3 = ax3.bar(coh_counts.index, coh_counts.values,
                    color=coh_colors, edgecolor="white",
                    linewidth=1.5, width=0.4)
 
    for bar, val in zip(bars3, coh_counts.values):
        pct = val / len(coherencia) * 100
        ax3.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.3,
                 f"{val:,}\n({pct:.1f}%)",
                 ha="center", va="bottom",
                 fontsize=10, fontweight="bold",
                 color=COLORS["gris"])
 
    ax3.set_title("Coherencia: Severidad\nReportada vs Predicha",
                  fontsize=12, fontweight="bold",
                  color=COLORS["azul"], pad=10)
    ax3.set_ylabel("Cantidad de reclamos", fontsize=9)
    ax3.spines[["top", "right"]].set_visible(False)
    ax3.set_facecolor("#FAFAFA")
 
# ── B4: Score del modelo por tipo de daño ───────────────────
ax4 = fig.add_subplot(gs[1, 0])
 
score_por_label = (df[df["damage_label"].notna()]
                   .groupby("damage_label")["damage_score"]
                   .agg(["mean", "std"])
                   .sort_values("mean", ascending=True))
 
if len(score_por_label) > 0:
    y_pos = range(len(score_por_label))
    ax4.barh(y_pos, score_por_label["mean"],
             xerr=score_por_label["std"].fillna(0),
             color=COLORS["azul_med"],
             edgecolor="white", linewidth=1.2,
             capsize=4, ecolor=COLORS["gris"])
 
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels(score_por_label.index, fontsize=9)
    ax4.axvline(0.5, color=COLORS["rojo"],
                linewidth=1.5, linestyle="--",
                label="Umbral confiable (0.5)")
    ax4.axvline(0.8, color=COLORS["verde_med"],
                linewidth=1.5, linestyle="--",
                label="Umbral alto (0.8)")
 
    ax4.set_title("Score Promedio\npor Tipo de Daño",
                  fontsize=12, fontweight="bold",
                  color=COLORS["azul"], pad=10)
    ax4.set_xlabel("Score de confianza", fontsize=9)
    ax4.set_xlim(0, 1)
    ax4.legend(fontsize=8)
    ax4.spines[["top", "right"]].set_visible(False)
    ax4.set_facecolor("#FAFAFA")
 
# ── B5: Score del modelo vs monto del reclamo ───────────────
ax5 = fig.add_subplot(gs[1, 1])
 
df_scatter = df[df["damage_label"].notna() &
                df["claim_total"].notna() &
                df["damage_score"].notna()].copy()
 
if len(df_scatter) > 0:
    scatter_colors = {
        label: color for label, color in zip(
            df_scatter["damage_label"].unique(),
            [COLORS["azul"], COLORS["azul_med"],
             COLORS["amarillo"], COLORS["rojo"],
             COLORS["verde_med"]]
        )
    }
 
    for label, group in df_scatter.groupby("damage_label"):
        ax5.scatter(
            group["damage_score"],
            group["claim_total"],
            c=scatter_colors.get(label, COLORS["gris"]),
            label=label, alpha=0.7,
            s=60, edgecolors="white", linewidth=0.5
        )
 
    ax5.axvline(0.5, color=COLORS["rojo"],
                linewidth=1.5, linestyle="--", alpha=0.7)
    ax5.set_title("Score del Modelo vs\nMonto del Reclamo",
                  fontsize=12, fontweight="bold",
                  color=COLORS["azul"], pad=10)
    ax5.set_xlabel("Score de confianza del modelo", fontsize=9)
    ax5.set_ylabel("Monto del reclamo ($)", fontsize=9)
    ax5.legend(fontsize=8, loc="upper left")
    ax5.spines[["top", "right"]].set_visible(False)
    ax5.set_facecolor("#FAFAFA")
 
# ── B6: Cobertura de predicciones ───────────────────────────
ax6 = fig.add_subplot(gs[1, 2])
 
con_pred   = df["damage_label"].notna().sum()
sin_pred   = df["damage_label"].isna().sum()
confiables = (df["confidence_level"].isin(["alta", "media"])).sum()
no_conf    = len(df) - confiables
 
categorias = ["Con\npredicción", "Sin\npredicción",
              "Predicción\nconfiable", "No\nconfiable"]
valores    = [con_pred, sin_pred, confiables, no_conf]
colores    = [COLORS["verde_med"], COLORS["gris_cla"],
              COLORS["azul"],     COLORS["amarillo"]]
 
bars6 = ax6.bar(categorias, valores, color=colores,
                edgecolor="white", linewidth=1.5, width=0.5)
 
for bar, val in zip(bars6, valores):
    pct = val / len(df) * 100
    ax6.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 0.3,
             f"{val:,}\n({pct:.0f}%)",
             ha="center", va="bottom",
             fontsize=9, fontweight="bold",
             color=COLORS["gris"])
 
ax6.set_title("Cobertura de\nPredicciones",
              fontsize=12, fontweight="bold",
              color=COLORS["azul"], pad=10)
ax6.set_ylabel("Cantidad de reclamos", fontsize=9)
ax6.spines[["top", "right"]].set_visible(False)
ax6.set_facecolor("#FAFAFA")
 
plt.savefig("/tmp/bi_bloque_b.png",
            dpi=150, bbox_inches="tight",
            facecolor="white")
plt.show()
print("✅ Bloque B guardado.")
 

# COMMAND ----------

# ============================================================
# BLOQUE C — VISTA DE DECISIÓN FINAL
# ============================================================
 
fig = plt.figure(figsize=(20, 16))
fig.patch.set_facecolor("white")
gs  = gridspec.GridSpec(2, 3, figure=fig,
                        hspace=0.50, wspace=0.38)
 
fig.suptitle(
    "SMART CLAIMS — BLOQUE C: VISTA DE DECISIÓN FINAL",
    fontsize=18, fontweight="bold",
    color=COLORS["azul"], y=0.98
)
 
decision_colors = {
    "LIBERADO":       COLORS["verde_med"],
    "REVISION_MENOR": COLORS["amarillo"],
    "INVESTIGAR":     COLORS["rojo"],
}
 
# ── C1: Distribución de Decisiones Finales ──────────────────
ax1 = fig.add_subplot(gs[0, 0])
 
dec_counts = (df["decision_final"]
              .fillna("SIN DECISIÓN")
              .value_counts())
dc_colors  = [decision_colors.get(d, COLORS["gris"])
              for d in dec_counts.index]
 
wedges, texts, autotexts = ax1.pie(
    dec_counts.values,
    labels=dec_counts.index,
    autopct="%1.1f%%",
    colors=dc_colors,
    startangle=90,
    wedgeprops=dict(edgecolor="white", linewidth=2.5),
    pctdistance=0.75
)
 
for at in autotexts:
    at.set_fontsize(10)
    at.set_fontweight("bold")
    at.set_color("white")
 
ax1.set_title("Distribución de\nDecisiones Finales",
              fontsize=12, fontweight="bold",
              color=COLORS["azul"], pad=10)
 
# ── C2: Predicción del Modelo vs Decisión Final ─────────────
ax2 = fig.add_subplot(gs[0, 1])
 
if "damage_label" in df.columns:
    df_c2  = df[df["damage_label"].notna()]
    cross  = (df_c2
              .groupby(["damage_label", "decision_final"])
              .size()
              .unstack(fill_value=0))
 
    x          = range(len(cross))
    width      = 0.25
    decisiones = ["LIBERADO", "REVISION_MENOR", "INVESTIGAR"]
 
    for i, decision in enumerate(decisiones):
        if decision in cross.columns:
            ax2.bar(
                [xi + i * width for xi in x],
                cross[decision].values,
                width=width,
                color=decision_colors[decision],
                edgecolor="white",
                linewidth=1,
                label=decision
            )
 
    ax2.set_xticks([xi + width for xi in x])
    ax2.set_xticklabels(cross.index, rotation=30,
                        ha="right", fontsize=9)
    ax2.set_title("Predicción del Modelo\nvs Decisión Final",
                  fontsize=12, fontweight="bold",
                  color=COLORS["azul"], pad=10)
    ax2.set_ylabel("Cantidad de reclamos", fontsize=9)
    ax2.legend(fontsize=8)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.set_facecolor("#FAFAFA")
 
# ── C3: Motivos de Revisión ──────────────────────────────────
ax3 = fig.add_subplot(gs[0, 2])
 
reglas = {
    "Póliza no vigente":     (~df["r1_poliza_vigente"].fillna(True)).sum(),
    "Monto excesivo":        (~df["r2_monto_razonable"].fillna(True)).sum(),
    "Sin reporte policial":  (~df["r3_reporte_policial"].fillna(True)).sum(),
    "Pred. no confiable":    (~df["r4_prediccion_confiable"].fillna(True)).sum(),
    "Severidad incoherente": (~df["r5_coherencia_severidad"].fillna(True)).sum(),
    "Tiempo excedido":       (~df["r6_tiempo_razonable"].fillna(True)).sum(),
}
 
reglas_s = dict(sorted(reglas.items(),
                        key=lambda x: x[1],
                        reverse=True))
 
max_val = max(reglas_s.values()) if max(reglas_s.values()) > 0 else 1
 
bars3 = ax3.barh(list(reglas_s.keys()),
                 list(reglas_s.values()),
                 color=COLORS["rojo"],
                 edgecolor="white",
                 linewidth=1.2)
 
for bar, val in zip(bars3, reglas_s.values()):
    # CORRECCIÓN: offset proporcional al máximo para evitar salirse del eje
    ax3.text(val + max_val * 0.02,
             bar.get_y() + bar.get_height() / 2,
             f"{val:,}",
             va="center", fontsize=9,
             fontweight="bold",
             color=COLORS["gris"])
 
ax3.set_title("Motivos de Revisión\n(reglas fallidas)",
              fontsize=12, fontweight="bold",
              color=COLORS["azul"], pad=10)
ax3.set_xlabel("Cantidad de reclamos afectados", fontsize=9)
ax3.spines[["top", "right"]].set_visible(False)
ax3.set_facecolor("#FAFAFA")
 
# ── C4: Monto Promedio por Decisión ─────────────────────────
ax4 = fig.add_subplot(gs[1, 0])
 
monto_dec  = (df.groupby("decision_final")["claim_total"]
                .agg(["mean", "sum"])
                .reindex(["LIBERADO", "REVISION_MENOR", "INVESTIGAR"]))
dc4_colors = [decision_colors.get(d, COLORS["gris"])
              for d in monto_dec.index]
 
bars4 = ax4.bar(monto_dec.index,
                monto_dec["mean"].fillna(0),
                color=dc4_colors,
                edgecolor="white",
                linewidth=1.5, width=0.5)
 
# CORRECCIÓN: fillna(0) antes de max() para evitar NaN
media_max = monto_dec["mean"].fillna(0).max()
for bar, val in zip(bars4, monto_dec["mean"].fillna(0)):
    ax4.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + media_max * 0.01,
             f"${val:,.0f}",
             ha="center", va="bottom",
             fontsize=9, fontweight="bold",
             color=COLORS["gris"])
 
ax4.set_title("Monto Promedio\npor Decisión Final",
              fontsize=12, fontweight="bold",
              color=COLORS["azul"], pad=10)
ax4.set_ylabel("Monto promedio ($)", fontsize=9)
ax4.spines[["top", "right"]].set_visible(False)
ax4.set_facecolor("#FAFAFA")
 
# ── C5: Reglas Fallidas por Severidad ───────────────────────
ax5 = fig.add_subplot(gs[1, 1])
 
if "reglas_fallidas" in df.columns and "incident_severity" in df.columns:
    reg_sev = (df.groupby("incident_severity")["reglas_fallidas"]
                 .mean()
                 .sort_values(ascending=True))
 
    bars5 = ax5.barh(reg_sev.index, reg_sev.values,
                     color=COLORS["azul_med"],
                     edgecolor="white", linewidth=1.2)
 
    max_reg = reg_sev.max() if reg_sev.max() > 0 else 1
    for bar, val in zip(bars5, reg_sev.values):
        ax5.text(val + max_reg * 0.02,
                 bar.get_y() + bar.get_height() / 2,
                 f"{val:.2f}",
                 va="center", fontsize=9,
                 color=COLORS["gris"])
 
    ax5.axvline(1, color=COLORS["rojo"],
                linewidth=1.5, linestyle="--",
                label="1 regla fallida")
    ax5.set_title("Promedio de Reglas Fallidas\npor Severidad",
                  fontsize=12, fontweight="bold",
                  color=COLORS["azul"], pad=10)
    ax5.set_xlabel("Promedio de reglas fallidas", fontsize=9)
    ax5.legend(fontsize=8)
    ax5.spines[["top", "right"]].set_visible(False)
    ax5.set_facecolor("#FAFAFA")
 
# ── C6: Resumen Ejecutivo KPIs ───────────────────────────────
ax6 = fig.add_subplot(gs[1, 2])
ax6.axis("off")
 
total      = len(df)
liberados  = (df["decision_final"] == "LIBERADO").sum()
investigar = (df["decision_final"] == "INVESTIGAR").sum()
revision   = (df["decision_final"] == "REVISION_MENOR").sum()
pred_ok    = df["damage_label"].notna().sum()
score_prom = df["damage_score"].mean()
monto_total  = df["claim_total"].sum()
monto_riesgo = df[df["decision_final"] == "INVESTIGAR"]["claim_total"].sum()
 
fmt_total  = f"${monto_total/1e6:.2f}M"  if monto_total  >= 1e6 else f"${monto_total:,.0f}"
fmt_riesgo = f"${monto_riesgo/1e6:.2f}M" if monto_riesgo >= 1e6 else f"${monto_riesgo:,.0f}"
 
kpis = [
    ("Total reclamos",         f"{total:,}",                                    COLORS["azul"]),
    ("Liberados",              f"{liberados:,} ({liberados/total:.0%})",         COLORS["verde_med"]),
    ("Revisión menor",         f"{revision:,} ({revision/total:.0%})",           COLORS["amarillo"]),
    ("A investigar",           f"{investigar:,} ({investigar/total:.0%})",       COLORS["rojo"]),
    ("Con predicción",         f"{pred_ok:,} ({pred_ok/total:.0%})",             COLORS["azul_med"]),
    ("Score promedio modelo",  f"{score_prom:.3f}" if not pd.isna(score_prom) else "N/A", COLORS["azul"]),
    ("Monto total reclamado",  fmt_total,                                        COLORS["gris"]),
    ("Monto en investigación", fmt_riesgo,                                       COLORS["rojo"]),
]
 
ax6.text(0.5, 0.98, "RESUMEN EJECUTIVO",
         ha="center", va="top",
         fontsize=13, fontweight="bold",
         color=COLORS["azul"],
         transform=ax6.transAxes)
 
for i, (kpi, valor, color) in enumerate(kpis):
    y = 0.85 - i * 0.10
    ax6.add_patch(mpatches.FancyBboxPatch(
        (0.02, y - 0.03), 0.96, 0.08,
        boxstyle="round,pad=0.01",
        facecolor=color, alpha=0.12,
        edgecolor=color, linewidth=1,
        transform=ax6.transAxes
    ))
    ax6.text(0.06, y + 0.01, kpi,
             fontsize=9, color=COLORS["gris"],
             transform=ax6.transAxes)
    ax6.text(0.94, y + 0.01, valor,
             fontsize=10, fontweight="bold",
             color=color, ha="right",
             transform=ax6.transAxes)
 
plt.savefig("/tmp/bi_bloque_c.png",
            dpi=150, bbox_inches="tight",
            facecolor="white")
plt.show()
print("✅ Bloque C guardado.")
 
print("\n" + "=" * 60)
print("BI COMPLETO — SMART CLAIMS")
print("=" * 60)
print("""
  BLOQUE A — Vista de negocio
    → Reclamos por severidad
    → Pérdida por tipo de incidente
    → Frecuencia por hora del incidente
    → Top marcas por pérdida
    → Distribución de montos
    → Evolución mensual
 
  BLOQUE B — Vista de predicción
    → Distribución de predicciones del modelo
    → Nivel de confianza
    → Coherencia severidad reportada vs predicha
    → Score por tipo de daño
    → Score vs monto del reclamo
    → Cobertura de predicciones
 
  BLOQUE C — Vista de decisión final
    → Distribución de decisiones
    → Predicción vs decisión final
    → Motivos de revisión por regla
    → Monto promedio por decisión
    → Reglas fallidas por severidad
    → Resumen ejecutivo con KPIs
 
✅ Pipeline completo: dato → predicción → regla → decisión
""")
