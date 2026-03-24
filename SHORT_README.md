# Implementacion Computacional

## Repositorio

El codigo fuente y los datos utilizados en este trabajo se encuentran disponibles en:

**https://github.com/daniloor/tesis-mfin**

El repositorio contiene todos los scripts de Python necesarios para replicar los resultados, junto con los datos de entrada y los resultados generados.

## Estructura del Repositorio

```
repositorio-tesis-mfin/
├── data/                      # Datos de entrada (29 archivos Excel OPEX + Yahoo Finance)
├── intraday/                  # Datos intradiarios de TradingView
├── process_data/              # Procesamiento y consolidacion de datos
├── garch/                     # Modelos ARCH y GARCH
├── har/                       # Modelos HAR-RV (Kumar 2010) con extension IV
├── ml/                        # Modelos de Machine Learning
├── backtesting/               # Backtesting de estrategias de opciones
└── README.md                  # Documentacion completa
```

## Datos

### Fuentes de Datos

| Tipo | Archivo | Periodo | Fuente |
|------|---------|---------|--------|
| Precios diarios | `GGAL.BA_max_1d_datos.csv` | 2000-07-26 a 2026-02-12 | Yahoo Finance |
| Intradiario 10 min | `BCBA_DLY_GGAL, 10 (1).csv` | 2021-07-19 a 2026-01-23 | TradingView |
| Intradiario 30 min | `BCBA_DLY_GGAL, 30 (4).csv` | 2012-01-03 a 2026-01-23 | TradingView |
| Opciones con IV | 29 archivos `.xlsb` | 2021-02-19 a 2025-12-18 | Sabrofrehley (BYMA) |

### Procesamiento de Datos

El script `process_data/process_all_ggal_data.py` consolida los datos de opciones y aplica ajustes por dividendos utilizando el metodo multiplicativo (estilo Yahoo Finance).

**Librerias utilizadas:** `pandas`, `numpy`, `pyxlsb`

**Archivos generados:**
- `data.dat`: Serie diaria consolidada con precios, retornos y volatilidad implicita promedio
- `options_data_YYYY_MM.dat`: 29 archivos con datos de opciones individuales por periodo OPEX

## Implementacion de Modelos

### 1. Modelos GARCH (`garch/GARCH_ANALYSIS.py`)

**Modelos implementados:**
- ARCH(1)
- GARCH(1,1)

**Especificacion:**
```
GARCH(1,1): σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}
```

**Libreria:** `arch` (Kevin Sheppard)

**Configuracion:**
- Distribucion de errores: Normal
- Metodo de estimacion: Maxima verosimilitud
- Horizonte de pronostico: Rolling window fuera de muestra

### 2. Modelos HAR-RV (`har/HAR_RV.py`)

**Modelos implementados (siguiendo Kumar, 2010):**

| Modelo | Regresores |
|--------|------------|
| HAR-RV | RV_d, RV_w, RV_m |
| HAR-RBV | RBV_d, RBV_w, RBV_m |
| HAR-TRBV | TRBV_d, TRBV_w, TRBV_m |
| HAR-CJ-RBV | C_d, C_w, C_m, J_d, J_w |
| HAR-RV-IV | RV_d, RV_w, RV_m, IV_d |
| HAR-RBV-IV | RBV_d, RBV_w, RBV_m, IV_d |

**Medidas de volatilidad realizada:**

```python
# Volatilidad Realizada (RV)
RV_t = sqrt(sum(r²_{t,j}))

# Varianza Bipotencia Realizada (RBV) - Kumar Ec. 8
mu_1 = sqrt(2/pi)
RBV_t = mu_1^{-2} * sum(|r_{t,j}| * |r_{t,j-1}|)

# Componentes Continuo y Salto - Kumar Ec. 12-13
C_t = RBV_t
J_t = max(RV_t - RBV_t, 0)
```

**Libreria:** `scikit-learn` (LinearRegression para OLS)

**Configuracion:**
- Frecuencias de muestreo: 10 minutos, 30 minutos
- Horizontes de pronostico: 1, 5, 22 dias
- Split train/test: 80%/20%

### 3. Modelos de Machine Learning (`ml/ML_ANALYSIS.py`)

**Modelos implementados:**

| Modelo | Libreria | Hiperparametros principales |
|--------|----------|----------------------------|
| Ridge | `scikit-learn` | alpha=15 |
| BayesianRidge | `scikit-learn` | Regularizacion automatica |
| Random Forest | `scikit-learn` | max_depth=3-4, n_estimators=100 |
| XGBoost | `xgboost` | max_depth=3-4, reg_lambda=10-15 |
| LightGBM | `lightgbm` | max_depth=3, learning_rate=0.03 |
| CatBoost | `catboost` | Regularizado |

**Features utilizados:**

```python
# HAR Core
RV_d, RV_w, RV_m      # Volatilidad realizada diaria, semanal, mensual
RBV_d                  # Varianza bipotencia diaria

# Volatilidad Implicita
IV_d, IV_w            # IV diaria y semanal

# Momentum
RV_mom_1d             # Cambio en RV vs dia anterior
RV_trend              # Tendencia de RV
VRP                   # Prima de riesgo de varianza (IV - RV)

# Volatilidad de la Volatilidad
RV_vol_5d             # Desviacion estandar de RV (5 dias)
RV_range_5d           # Rango de RV (5 dias)
RV_cv_5d              # Coeficiente de variacion
```

### 4. Backtesting de Estrategias (`backtesting/GAMMA_SCALPING_BACKTEST.py`)

**Estrategias implementadas:**
- CALL: Venta de call ATM + delta hedge
- PUT: Venta de put ATM + delta hedge
- STRADDLE: Venta de call + put ATM (mismo strike)
- STRANGLE: Venta de call + put OTM (5% fuera del dinero)

**Modelo de senales:** HAR-RBV-IV

```python
# Features del modelo de senales
features = ['RBV_d', 'RBV_w', 'RBV_m', 'IV_d']

# Target: log(RBV) promedio de los proximos 5 dias
y_5d = log(RBV).rolling(5).mean().shift(-5)
```

**Funciones de Black-Scholes:**

```python
def bs_delta_call(S, K, T, r, sigma):
    d1 = (log(S/K) + (r + 0.5*sigma²)*T) / (sigma*sqrt(T))
    return norm.cdf(d1)

def bs_gamma(S, K, T, r, sigma):
    d1 = (log(S/K) + (r + 0.5*sigma²)*T) / (sigma*sqrt(T))
    return norm.pdf(d1) / (S * sigma * sqrt(T))
```

**Librerias:** `scipy.stats` (norm), `numpy`, `pandas`

**Parametros de configuracion:**

| Parametro | Valor |
|-----------|-------|
| Comision opciones | 0.20% |
| Comision acciones | 0.0605% |
| Umbral gamma para roll | 50% |
| Dias minimos al vencimiento | 7 |
| Offset OTM (strangles) | 5% |

## Entorno de Desarrollo

**Lenguaje:** Python 3.10+

**Librerias principales:**

```
pandas>=2.0.0          # Manipulacion de datos
numpy>=1.24.0          # Computacion numerica
scipy>=1.10.0          # Funciones estadisticas
statsmodels>=0.14.0    # Modelos econometricos
arch>=5.3.0            # Modelos GARCH
scikit-learn>=1.3.0    # Machine Learning
xgboost>=1.7.0         # Gradient Boosting
lightgbm>=4.0.0        # Gradient Boosting
catboost>=1.2.0        # Gradient Boosting
matplotlib>=3.7.0      # Visualizacion
seaborn>=0.12.0        # Visualizacion estadistica
pyxlsb>=1.0.10         # Lectura de archivos Excel binarios
yfinance>=0.2.0        # Descarga de datos de Yahoo Finance
```

**Instalacion de dependencias:**

```bash
pip install pandas numpy scipy statsmodels arch scikit-learn xgboost lightgbm catboost matplotlib seaborn pyxlsb yfinance
```

## Ejecucion

Los scripts se ejecutan en el siguiente orden:

```bash
# 1. Procesamiento de datos
cd process_data && python process_all_ggal_data.py

# 2. Analisis GARCH
cd garch && python GARCH_ANALYSIS.py

# 3. Modelos HAR-RV
cd har && python HAR_RV.py

# 4. Modelos Machine Learning
cd ml && python ML_ANALYSIS.py

# 5. Backtesting
cd backtesting && python GAMMA_SCALPING_BACKTEST.py
```

Todos los resultados (tablas CSV y graficos PNG) se generan automaticamente en subcarpetas `results/` dentro de cada directorio.
