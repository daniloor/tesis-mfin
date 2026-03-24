# Analisis de Volatilidad GGAL - Tesis MFin

Este repositorio contiene los scripts y datos necesarios para procesar y analizar la informacion de opciones y acciones de GGAL (Grupo Financiero Galicia).

## Estructura del Repositorio

```
repositorio-tesis-mfin/
├── data/                                   # Datos de entrada
│   ├── $GGAL - Info Opex YYYY-MM.xlsb      # 29 archivos Excel (2021-04 a 2025-12)
│   ├── GGAL.BA_max_1d_datos.csv            # Datos historicos de Yahoo Finance
│   └── download_data_yfinance.py           # Script para descargar datos de Yahoo Finance
├── intraday/                               # Datos intradiarios de TradingView
│   ├── BCBA_DLY_GGAL, 10 (1).csv           # Datos cada 10 minutos
│   └── BCBA_DLY_GGAL, 30 (4).csv           # Datos cada 30 minutos
├── process_data/                           # Procesamiento de datos
│   ├── process_all_ggal_data.py            # Script principal de procesamiento
│   ├── dividends_complete.csv              # Lista completa de dividendos y cupones
│   ├── data.dat                            # [GENERADO] Datos consolidados del stock
│   └── options_data_YYYY_MM.dat            # [GENERADO] 29 archivos de opciones
├── garch/                                  # Analisis GARCH
│   ├── GARCH_ANALYSIS.py                   # Script de estimacion ARCH/GARCH
│   ├── garch_results.txt                   # [GENERADO] Resultados numericos
│   └── results/                            # [GENERADO] Graficos y CSVs
├── har/                                    # Analisis HAR-RV
│   ├── HAR_RV.py                           # Script completo multi-horizonte con IV
│   └── results/                            # [GENERADO] Tablas y graficos
├── ml/                                     # Analisis Machine Learning
│   ├── ML_ANALYSIS.py                      # Script de modelos ML
│   └── results/                            # [GENERADO] Tablas comparativas
└── backtesting/                            # Backtesting de estrategias
    ├── RV_VS_IV_ANALYSIS.py                # Analisis RV vs IV
    ├── GAMMA_SCALPING_BACKTEST.py          # Backtesting gamma scalping
    └── results/                            # [GENERADO] Graficos y CSVs
```

## Rangos de Fechas de los Datos

### Datos Principales

| Archivo | Inicio | Fin | Frecuencia | Fuente |
|---------|--------|-----|------------|--------|
| data.dat | 2021-02-19 | 2025-12-18 | Diario | Archivos OPEX |
| GGAL.BA_max_1d_datos.csv | 2000-07-26 | 2026-02-12 | Diario | Yahoo Finance |
| BCBA_DLY_GGAL, 10 (1).csv | 2021-07-19 14:00 | 2026-01-23 19:50 | 10 minutos | TradingView |
| BCBA_DLY_GGAL, 30 (4).csv | 2012-01-03 14:00 | 2026-01-23 19:30 | 30 minutos | TradingView |

### Datos de Opciones por Periodo OPEX

| Archivo | Inicio | Fin |
|---------|--------|-----|
| options_data_2021_04.dat | 2021-02-19 | 2021-04-15 |
| options_data_2021_06.dat | 2021-04-16 | 2021-06-17 |
| options_data_2021_08.dat | 2021-06-18 | 2021-08-19 |
| options_data_2021_10.dat | 2021-08-20 | 2021-10-14 |
| options_data_2021_12.dat | 2021-10-15 | 2021-12-16 |
| options_data_2022_02.dat | 2021-12-17 | 2022-02-17 |
| options_data_2022_04.dat | 2022-02-18 | 2022-04-12 |
| options_data_2022_06.dat | 2022-04-13 | 2022-06-15 |
| options_data_2022_08.dat | 2022-06-16 | 2022-08-18 |
| options_data_2022_10.dat | 2022-08-19 | 2022-10-20 |
| options_data_2022_12.dat | 2022-10-21 | 2022-12-15 |
| options_data_2023_02.dat | 2022-12-16 | 2023-02-16 |
| options_data_2023_04.dat | 2023-02-17 | 2023-04-20 |
| options_data_2023_06.dat | 2023-04-21 | 2023-06-15 |
| options_data_2023_08.dat | 2023-06-16 | 2023-08-17 |
| options_data_2023_10.dat | 2023-08-18 | 2023-10-19 |
| options_data_2023_12.dat | 2023-10-20 | 2023-12-14 |
| options_data_2024_02.dat | 2023-12-15 | 2024-02-15 |
| options_data_2024_04.dat | 2024-02-16 | 2024-04-18 |
| options_data_2024_06.dat | 2024-04-19 | 2024-06-18 |
| options_data_2024_08.dat | 2024-06-19 | 2024-08-15 |
| options_data_2024_10.dat | 2024-08-16 | 2024-10-17 |
| options_data_2024_12.dat | 2024-10-18 | 2024-12-19 |
| options_data_2025_02.dat | 2024-12-20 | 2025-02-20 |
| options_data_2025_04.dat | 2025-02-21 | 2025-04-15 |
| options_data_2025_06.dat | 2025-04-16 | 2025-06-18 |
| options_data_2025_08.dat | 2025-06-19 | 2025-08-13 |
| options_data_2025_10.dat | 2025-08-14 | 2025-10-16 |
| options_data_2025_12.dat | 2025-10-17 | 2025-12-18 |

## Requisitos

```bash
pip install pandas numpy pyxlsb yfinance arch scipy statsmodels matplotlib seaborn scikit-learn xgboost lightgbm catboost
```

---

## 1. Procesamiento de Datos (`process_data/`)

### Descripcion

El script `process_all_ggal_data.py` procesa los archivos Excel de OPEX para generar datos consolidados del stock y opciones de GGAL.

### Ejecucion

```bash
cd process_data
python process_all_ggal_data.py
```

### Que hace el script

1. Lee los 29 archivos Excel de la carpeta `data/`
2. Extrae datos del stock y opciones de cada periodo
3. Detecta dividendos a partir de cambios en los strikes de opciones
4. Aplica ajustes de precio estilo Yahoo Finance (multiplicativo)
5. Genera `data.dat` y 29 archivos `options_data_YYYY_MM.dat`

### Archivos de Salida

**`data.dat`** - Datos consolidados del stock (1181 registros)

| Columna | Descripcion |
|---------|-------------|
| Date | Fecha |
| Last_Price | Precio de cierre |
| Open, High, Low | Precio apertura, maximo, minimo |
| Simple_Return | Retorno simple diario |
| Log_Return | Retorno logaritmico diario |
| IV_Call_Avg, IV_Put_Avg | Volatilidad implicita promedio |
| Adj_Price | Precio ajustado por dividendos |

**`options_data_YYYY_MM.dat`** - 29 archivos con datos de opciones individuales

| Columna | Descripcion |
|---------|-------------|
| Ticker | Simbolo de la opcion |
| Type | "Call" o "Put" |
| Strike | Precio de ejercicio |
| IV | Volatilidad implicita |
| Delta, Gamma, Theta, Vega, Rho | Griegas |

---

## 2. Analisis GARCH (`garch/`)

### Descripcion

El script `GARCH_ANALYSIS.py` estima modelos ARCH(1) y GARCH(1,1) para pronosticar la volatilidad de los retornos de GGAL. Compara el desempeño usando retornos simples vs logaritmicos.

### Ejecucion

```bash
cd garch
python GARCH_ANALYSIS.py
```

### Que hace el script

1. Carga los datos de `process_data/data.dat`
2. Calcula estadisticas descriptivas de retornos simples y logaritmicos
3. Realiza test de efectos ARCH (Ljung-Box sobre retornos al cuadrado)
4. Divide datos en entrenamiento (80%) y prueba (20%)
5. Estima 4 modelos: ARCH(1) y GARCH(1,1) para cada tipo de retorno
6. Realiza pronosticos rolling window fuera de muestra
7. Genera graficos de diagnostico y tablas comparativas

### Dataset Utilizado

Los modelos GARCH utilizan **unicamente datos diarios** de `data.dat`, sin requerir datos intradiarios ni volatilidad implicita.

| Metrica | Valor |
|---------|-------|
| Archivo fuente | `process_data/data.dat` |
| Columna de precios | **Adj_Price** (ajustado por dividendos) |
| Observaciones originales | 1,181 dias |
| Observaciones utilizadas | 1,180 dias |
| Periodo | 2021-02-22 a 2025-12-18 |
| Conjunto de entrenamiento | 944 dias (80%) |
| Conjunto de test | 236 dias (20%) |

**Importante:** Los retornos se calculan sobre el **precio ajustado por dividendos (Adj_Price)** para capturar el rendimiento total del inversor. Esto evita la distorsion artificial que ocurre en los dias ex-dividendo cuando se usan precios sin ajustar (Last_Price), ya que el precio cae por el monto del dividendo pero el retorno real del inversor es cero o positivo.

**Nota:** Solo se pierde 1 observacion (el primer dia, 2021-02-19) porque no hay precio anterior para calcular el retorno. A diferencia de los modelos HAR y ML que requieren datos intradiarios, los modelos GARCH trabajan exclusivamente con retornos diarios.

### Resultados Principales

#### Comparacion de Modelos

| Modelo | Tipo Retorno | AIC | BIC | Persistencia | Vol. Anualizada |
|--------|--------------|-----|-----|--------------|-----------------|
| ARCH(1) | Simple | 4923.93 | 4933.63 | 0.184 | 52.98% |
| ARCH(1) | Logaritmico | 4906.09 | 4915.79 | 0.176 | 52.38% |
| GARCH(1,1) | Simple | 4896.32 | 4910.87 | 0.939 | 53.29% |
| **GARCH(1,1)** | **Logaritmico** | **4877.88** | **4892.43** | **0.946** | **52.54%** |

#### Mejor Modelo: GARCH(1,1) con Retornos Logaritmicos

El modelo GARCH(1,1) con retornos logaritmicos es el mejor segun los criterios AIC y BIC.

**Parametros estimados:**
- omega: 0.5867
- alpha (efecto ARCH): 0.0732
- beta (efecto GARCH): 0.8732
- Persistencia (alpha + beta): 0.946
- Half-life: 12.6 dias
- Volatilidad anualizada: 52.54%

**Interpretacion:**
- La alta persistencia (0.946) indica que los shocks de volatilidad tardan aproximadamente 12.6 dias en reducirse a la mitad
- El efecto ARCH (0.073) es relativamente bajo, indicando que shocks individuales tienen impacto moderado
- El efecto GARCH (0.873) es alto, mostrando que la volatilidad pasada es muy predictiva de la volatilidad futura

#### Por que Retornos Logaritmicos son Preferibles

1. **Mejor ajuste estadistico**: AIC 18.44 puntos menor que retornos simples
2. **Distribucion mas simetrica**: Sesgo de 0.26 vs 0.60 en retornos simples
3. **Aditividad temporal**: Los retornos log son aditivos en el tiempo
4. **Estandar academico**: Preferidos en la literatura financiera

#### Pronostico Fuera de Muestra

- Periodo de prueba: 2025-01-02 a 2025-12-18 (236 dias)
- Correlacion pronostico/realizado (Log): 0.0956
- MSE (Log): 2961.75 vs 3622.00 (Simple)

### Archivos Generados

```
garch/
├── garch_results.txt                       # Reporte completo de resultados
└── results/
    ├── 01_price_returns_timeseries.png     # Series de precio y retornos
    ├── 02_return_distribution.png          # Distribucion de retornos
    ├── 03_autocorrelation_analysis.png     # ACF/PACF de retornos al cuadrado
    ├── 04_fitted_conditional_volatility.png # Volatilidad condicional ajustada
    ├── 05_model_comparison.png             # Comparacion AIC/BIC
    ├── 06_forecast_vs_realized.png         # Pronostico vs realizado
    ├── 07_model_diagnostics.png            # Diagnosticos de residuos
    ├── model_comparison.csv                # Tabla comparativa de modelos
    └── parameters_summary.csv              # Parametros estimados
```

---

## 3. Analisis HAR-RV (`har/`)

### Descripcion

El script `HAR_RV.py` implementa modelos HAR (Heterogeneous AutoRegressive) siguiendo la metodologia de Kumar (2010), extendidos con volatilidad implicita (IV) y multiples horizontes de pronostico.

**Referencia:** Kumar, M. (2010). "Improving the accuracy: volatility modeling and forecasting using high-frequency data and the variational component." *Journal of Industrial Engineering and Management*, 3(1), 199-220.

### Ejecucion

```bash
cd har
python HAR_RV.py
```

### Que hace el script

1. Carga datos intradiarios de 10 y 30 minutos desde `intraday/`
2. Carga datos de volatilidad implicita (IV) desde `process_data/data.dat`
3. Calcula medidas de volatilidad realizada (RV, RBV, TRBV)
4. Descompone la volatilidad en componentes continuos (C) y de salto (J)
5. Estima 10 modelos HAR (5 Kumar originales + 5 con IV)
6. Evalua en 3 horizontes de pronostico (1d, 5d, 22d)
7. Genera tablas y graficos comparativos

### Especificaciones del Analisis

- **10 modelos**: 5 Kumar originales + 5 con volatilidad implicita (IV)
- **2 frecuencias de datos**: 10 minutos y 30 minutos
- **3 horizontes de pronostico**: 1 dia, 5 dias, 22 dias
- **Total**: 60 combinaciones modelo-frecuencia-horizonte
- **Split train/test**: 80%/20%

### Datasets Utilizados

**Importante:** Para garantizar una comparacion justa entre modelos, TODOS los modelos HAR (con y sin IV) se evaluan sobre el **mismo dataset**, resultado de un inner join entre datos intradiarios y datos de volatilidad implicita.

#### Tamanos de Datasets Originales

| Dataset | Dias Originales | Periodo |
|---------|-----------------|---------|
| Intradiario 10 min | 1,105 dias | 2021-07-19 a 2026-01-23 |
| Intradiario 30 min | 3,421 dias | 2012-01-03 a 2026-01-23 |
| data.dat (con IV) | 1,181 dias | 2021-02-19 a 2025-12-18 |

#### Datasets Despues del Inner Join con IV

| Frecuencia | Dias Despues de Merge | Train (80%) | Test (20%) |
|------------|----------------------|-------------|------------|
| 10 minutos | 1,082 dias | 829 dias | 208 dias |
| 30 minutos | 1,181 dias | 908 dias | 228 dias |

**Notas:**
- Los datos de 30 minutos tienen historia desde 2012, pero solo se usan los dias que coinciden con datos de IV (desde febrero 2021)
- Los datos de 10 minutos comienzan en julio 2021, casi al mismo tiempo que los datos de IV
- Esta metodologia permite comparar directamente modelos HAR-RV vs HAR-RV-IV sobre exactamente las mismas observaciones

### Medidas de Volatilidad

| Medida | Descripcion |
|--------|-------------|
| RV | Volatilidad realizada estandar (suma de retornos al cuadrado) |
| RBV | Varianza bipotencia realizada (robusta a saltos) |
| TRBV | Varianza bipotencia con umbral |
| C | Componente continuo (= RBV) |
| J | Componente de salto (= max(RV - RBV, 0)) |

### Modelos Estimados

| Modelo | Regresores | Descripcion |
|--------|------------|-------------|
| HAR-RV | RV_d, RV_w, RV_m | Modelo base con RV diario, semanal, mensual |
| HAR-RBV | RBV_d, RBV_w, RBV_m | Usando varianza bipotencia (robusta a saltos) |
| HAR-TRBV | TRBV_d, TRBV_w, TRBV_m | Usando varianza con umbral |
| HAR-CJ-RBV | C_d, C_w, C_m, J_d, J_w, J_m | Descomposicion en saltos y continuo |
| HAR-TCJ-RBV | TC_d, TC_w, TC_m, TJ_d, TJ_w, TJ_m | Con umbral |
| LOG-HAR-* | ln(.) de los anteriores | Versiones logaritmicas |

### Resultados Principales

#### Mejores Modelos por Horizonte (10 minutos)

| Horizonte | Mejor Modelo (solo RV) | MSE (RV) | Mejor Modelo (con IV) | MSE (IV) |
|-----------|------------------------|----------|----------------------|----------|
| 1 dia | HAR-RBV | 1.02E-04 | **HAR-RBV-IV** | **9.97E-05** |
| 5 dias | HAR-RBV | 4.42E-05 | **HAR-RBV-IV** | **4.20E-05** |
| 22 dias | HAR-RV | 2.29E-05 | **HAR-RV-IV** | **2.12E-05** |

#### Mejores Modelos por Horizonte (30 minutos)

| Horizonte | Mejor Modelo (solo RV) | MSE (RV) | Mejor Modelo (con IV) | MSE (IV) |
|-----------|------------------------|----------|----------------------|----------|
| 1 dia | HAR-CJ-RBV | 1.36E-04 | **HAR-CJ-RBV-IV** | **1.32E-04** |
| 5 dias | HAR-CJ-RBV | 4.30E-05 | **HAR-RV-IV-Full** | **3.99E-05** |
| 22 dias | HAR-RV | 1.84E-05 | **HAR-RV-IV** | **1.75E-05** |

#### Modelos Evaluados

**5 Modelos Kumar (2010) originales:**
| Modelo | Regresores | Descripcion |
|--------|------------|-------------|
| HAR-RV | RV_d, RV_w, RV_m | Modelo base |
| HAR-RBV | RBV_d, RBV_w, RBV_m | Varianza bipotencia (robusta a saltos) |
| HAR-TRBV | TRBV_d, TRBV_w, TRBV_m | Varianza con umbral |
| HAR-CJ-RBV | C_d, C_w, C_m, J_d, J_w | Descomposicion continuo/salto |
| HAR-TCJ-RBV | TC_d, TC_w, TC_m, TJ_d, TJ_w | Con umbral |

**5 Modelos con Volatilidad Implicita (extension):**
| Modelo | Descripcion |
|--------|-------------|
| HAR-RV-IV | HAR-RV + IV diario |
| HAR-RBV-IV | HAR-RBV + IV diario |
| HAR-RV-IV-Full | HAR-RV + IV diario, semanal, mensual |
| HAR-RBV-IV-Full | HAR-RBV + IV diario, semanal, mensual |
| HAR-CJ-RBV-IV | HAR-CJ-RBV + IV diario |

#### Interpretacion

- **HAR-RBV es el mejor modelo basado solo en RV**, confirmando los hallazgos de Kumar (2010)
- **La volatilidad implicita (IV) mejora los pronosticos** en todos los horizontes y frecuencias
- **HAR-RBV-IV domina para horizontes cortos y medios** (1d, 5d) con datos de 10 minutos
- **HAR-RV-IV domina para horizonte largo** (22d) en ambas frecuencias
- **La frecuencia de muestreo importa**: 10 minutos supera a 30 minutos en R² (0.32 vs 0.22)

### Archivos Generados

```
har/
├── HAR_RV.py                               # Script principal
└── results/
    ├── kumar_models_comprehensive.csv      # Resultados completos (60 combinaciones)
    ├── kumar_best_by_horizon.csv           # Mejor modelo por horizonte
    ├── kumar_mse_by_horizon.png            # Comparacion MSE por horizonte
    └── kumar_r2_by_horizon.png             # Comparacion R2 por horizonte
```

---

## 4. Analisis Machine Learning (`ml/`)

### Descripcion

El script `ML_ANALYSIS.py` extiende la metodologia HAR-RV incorporando modelos de machine learning con features avanzados (momentum, volatilidad de la volatilidad) y tecnicas de regularizacion.

### Ejecucion

```bash
cd ml
python ML_ANALYSIS.py
```

### Que hace el script

1. Carga datos intradiarios de 10 minutos y datos de opciones (IV)
2. Calcula features HAR (RV_d, RV_w, RV_m, RBV_d, IV_d)
3. Agrega features de momentum (RV_mom_1d, RV_trend, VRP)
4. Agrega features de volatilidad de la volatilidad (RV_vol_5d, RV_range_5d)
5. Entrena multiples modelos ML con regularizacion
6. Compara con baselines HAR-RBV y HAR-RBV-IV
7. Genera tablas comparativas por horizonte de pronostico

### Dataset Utilizado

El script ML utiliza el mismo inner join que HAR entre datos de 10 minutos y volatilidad implicita:

| Metrica | Valor |
|---------|-------|
| Dias totales (despues de merge) | 1,081 dias |
| Dias limpios (despues de feature engineering) | 993 dias |
| Periodo | Jul 2021 - Dic 2025 |
| Conjunto de entrenamiento | 794 dias (80%) |
| Conjunto de test | 199 dias (20%) |

**Nota:** El dataset es ligeramente menor que HAR porque el feature engineering requiere ventanas de 22 dias y elimina observaciones con valores faltantes.

### Modelos Evaluados

| Modelo | Tipo | Configuracion |
|--------|------|---------------|
| Ridge | Lineal | alpha=15 (regularizacion L2) |
| BayesianRidge | Lineal | Regularizacion automatica via priors |
| RandomForest | Ensemble | Arboles poco profundos, bagging |
| XGBoost | Gradient Boosting | max_depth=3-4, reg_lambda=10-15 |
| LightGBM | Gradient Boosting | max_depth=3, learning_rate=0.03 |
| CatBoost | Gradient Boosting | Regularizado |

### Features Utilizados

| Categoria | Features | Descripcion |
|-----------|----------|-------------|
| HAR Core | RV_d, RV_w, RV_m, RBV_d | Volatilidad realizada diaria, semanal, mensual |
| IV | IV_d, IV_w | Volatilidad implicita |
| Momentum | RV_mom_1d, RV_trend, VRP | Cambios en volatilidad, prima de riesgo de varianza |
| Vol-of-Vol | RV_vol_5d, RV_range_5d, RV_cv_5d, RV_pctrank | Volatilidad de la volatilidad |

### Resultados Principales

#### Comparacion ML vs HAR (Fuera de Muestra)

| Horizonte | Mejor ML | MSE (ML) | Mejor HAR | MSE (HAR) | Mejora ML |
|-----------|----------|----------|-----------|-----------|-----------|
| 1 dia | Ridge | 9.65E-05 | HAR-RBV-IV | 9.77E-05 | **+1.2%** |
| 5 dias | LightGBM | 4.18E-05 | HAR-RBV-IV | 4.33E-05 | **+3.4%** |
| 22 dias | BayesRidge | 2.16E-05 | HAR-RV-IV | 2.21E-05 | **+1.9%** |

**Conclusion:** Los modelos ML superan a los baselines HAR-IV en los 3 horizontes, con mejoras entre 1.2% y 3.4%.

#### Ranking por Horizonte

**1 dia:**
| Rank | Modelo | Test MSE | Tipo |
|------|--------|----------|------|
| 1 | Ridge | 9.65E-05 | ML |
| 2 | BayesRidge | 9.72E-05 | ML |
| 3 | HAR-RBV-IV | 9.77E-05 | Baseline |

**5 dias:**
| Rank | Modelo | Test MSE | Tipo |
|------|--------|----------|------|
| 1 | LightGBM | 4.18E-05 | ML |
| 2 | RandomForest | 4.20E-05 | ML |
| 3 | XGBoost | 4.24E-05 | ML |
| 4 | HAR-RBV-IV | 4.33E-05 | Baseline |

**22 dias:**
| Rank | Modelo | Test MSE | Tipo |
|------|--------|----------|------|
| 1 | BayesRidge | 2.16E-05 | ML |
| 2 | Ridge | 2.17E-05 | ML |
| 3 | HAR-RV-IV | 2.21E-05 | Baseline |

### Interpretacion

- **Horizonte 1 dia:** Modelos lineales (Ridge) dominan porque las relaciones son aproximadamente lineales
- **Horizonte 5 dias:** Modelos de arboles (LightGBM) capturan patrones no lineales
- **Horizonte 22 dias:** La volatilidad revierte a la media, favoreciendo modelos lineales (BayesRidge)
- **Features de momentum** (VRP, RV_trend) aportan valor incremental sobre HAR puro
- **Las mejoras son modestas** (1-3%) pero consistentes en todos los horizontes

### Importancia de Features (XGBoost)

| Rank | Feature | Importancia (1d) | Categoria |
|------|---------|------------------|-----------|
| 1 | RV_m | 0.285 | HAR Core |
| 2 | RV_w | 0.264 | HAR Core |
| 3 | IV_d | 0.115 | IV |
| 4 | RBV_d | 0.066 | HAR Core |
| 5 | RV_d | 0.062 | HAR Core |
| 6 | RV_cv_5d | 0.036 | Vol-of-Vol |
| 7 | RV_pctrank | 0.034 | Vol-of-Vol |

**Interpretacion:** Los features HAR de largo plazo (RV_m, RV_w) dominan la prediccion con mas del 50% de la importancia combinada, seguidos por la volatilidad implicita (IV_d).

### Archivos Generados

```
ml/
├── ML_ANALYSIS.py                          # Script principal
└── results/
    ├── final_model_comparison.csv          # Comparacion detallada de todos los modelos
    ├── summary.csv                         # Resumen ejecutivo ML vs HAR
    └── feature_importance.csv              # Importancia de features por horizonte
```

---

## 5. Backtesting de Estrategias (`backtesting/`)

### Descripcion

Esta carpeta contiene dos scripts complementarios:

1. **RV_VS_IV_ANALYSIS.py**: Analiza la relacion entre Volatilidad Realizada (RV) e Implicita (IV) para evaluar la Prima de Riesgo de Varianza (VRP)
2. **GAMMA_SCALPING_BACKTEST.py**: Backtesting de estrategias de gamma scalping usando senales del modelo HAR-RBV-IV

### Ejecucion

```bash
cd backtesting
python RV_VS_IV_ANALYSIS.py
python GAMMA_SCALPING_BACKTEST.py
```

### Script 1: RV_VS_IV_ANALYSIS.py

Analiza la relacion entre volatilidad realizada (calculada con datos intradiarios) y volatilidad implicita (de opciones).

**Que hace:**
1. Calcula RV diaria desde datos de 10 minutos
2. Extrae IV promedio de calls y puts
3. Calcula la Prima de Riesgo de Varianza (VRP = IV - RV)
4. Genera estadisticas descriptivas y graficos

**Resultados Principales:**

| Metrica | Periodo Completo | Periodo Test |
|---------|------------------|--------------|
| Dias de trading | 1,076 | 197 |
| Dias IV > RV | 802 (74.5%) | 156 (79.2%) |
| Correlacion (RV, IV) | 0.407 | 0.588 |
| Ratio IV/RV promedio | 1.32 | 1.34 |
| VRP promedio diario | 0.54% | 0.55% |

**Interpretacion:**
- Las opciones de GGAL estan **sistematicamente sobrepriceadas**: IV > RV en ~75-79% de los dias
- Existe una prima de riesgo de varianza positiva y persistente (~0.55% diario)
- Esto favorece estrategias de **venta de volatilidad** (short straddles/strangles)

### Script 2: GAMMA_SCALPING_BACKTEST.py

Backtesting de estrategias de gamma scalping con delta hedging diario.

**Estrategias evaluadas:**
- **CALL**: Opciones call ATM + delta hedge
- **PUT**: Opciones put ATM + delta hedge
- **STRADDLE**: Call + Put ATM (mismo strike) + delta hedge
- **STRANGLE**: Call + Put OTM (5% fuera del dinero) + delta hedge
- **LONG_STOCK**: Comprar y mantener (baseline)
- **SHORT_STOCK**: Vender y mantener (baseline)

**Configuracion:**
- Periodo test: Febrero 2025 - Diciembre 2025 (~10 meses)
- Delta hedge diario
- Roll cuando gamma < 50% del ATM gamma
- Minimo 7 dias hasta vencimiento

**Modelo de Senales:** HAR-RBV-IV (features: RBV_d, RBV_w, RBV_m, IV_d)

**Resultados Principales:**

| Estrategia | Trades | Retorno |
|------------|--------|---------|
| **STRANGLE** | 20 | **+18.4%** |
| SHORT_STOCK | 1 | +15.5% |
| STRADDLE | 13 | +13.7% |
| PUT | 8 | +10.6% |
| CALL | 9 | +5.7% |
| LONG_STOCK | 1 | -15.7% |

**Resultados por Ciclo OPEX:**

| OPEX | CALL | PUT | STRADDLE | STRANGLE |
|------|------|-----|----------|----------|
| 2025-04 | -0.4% | -1.9% | -0.6% | -1.6% |
| 2025-06 | +1.5% | +1.8% | +0.6% | +2.9% |
| 2025-08 | +2.2% | +2.6% | +4.9% | +5.9% |
| 2025-10 | -4.8% | -5.5% | -9.8% | -4.2% |
| 2025-12 | +7.5% | +13.6% | +18.9% | +16.3% |

**Interpretacion:**
- **STRANGLE es la mejor estrategia** (+18.4%), superando incluso a SHORT_STOCK
- Todas las estrategias de opciones fueron rentables
- Octubre 2025 fue el periodo mas dificil (rally explosivo de GGAL)
- Diciembre 2025 fue el mejor periodo para todas las estrategias
- En un contexto de tasas altas (~40% anual), LONG_STOCK pierde dinero (-15.7%)
- El modelo HAR-RBV-IV utilizado para generar senales incorpora volatilidad implicita, mejorando la precision de las predicciones respecto a HAR-RBV puro

### Archivos Generados

```
backtesting/
├── RV_VS_IV_ANALYSIS.py                    # Analisis RV vs IV
├── GAMMA_SCALPING_BACKTEST.py              # Backtesting gamma scalping
└── results/
    ├── 01_rv_iv_timeseries.png             # Series temporales RV vs IV
    ├── 02_distribution_comparison.png      # Comparacion de distribuciones
    ├── 03_rolling_statistics.png           # Estadisticas moviles
    ├── 04_summary_dashboard.png            # Dashboard resumen
    ├── statistics_comparison.csv           # Estadisticas RV vs IV
    ├── daily_rv_iv_data.csv                # Datos diarios
    ├── gamma_scalp_summary.csv             # Resumen de estrategias
    ├── gamma_scalp_*_trades.csv            # Detalle de trades por estrategia
    └── gamma_scalp_*_hedge_log.csv         # Log de hedging diario
```

---

## Metodologia de Ajuste de Dividendos

El procesamiento utiliza el metodo de ajuste multiplicativo (estilo Yahoo Finance):

1. **Deteccion**: Se detectan dividendos comparando cambios uniformes en los strikes de opciones
2. **Factor de ajuste**: `factor = 1 - (dividendo / precio_ex_date)`
3. **Aplicacion**: Precios anteriores se multiplican por el factor acumulado

---

## Notas

- Los archivos generados ya estan incluidos en el repositorio
- Para regenerar datos, ejecute los scripts en orden: primero `process_data/`, luego `garch/`, luego `har/`, luego `ml/`
