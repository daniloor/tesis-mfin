# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 15:47:04 2024

@author: Joel Ivan Cittar
"""

import yfinance as yf
import datetime

def period_to_days(period_str):
    if period_str == 'max':
        return float('inf')
    elif period_str == 'ytd':
        today = datetime.date.today()
        start_of_year = datetime.date(today.year, 1, 1)
        delta = today - start_of_year
        return delta.days
    else:
        num = ''.join(filter(str.isdigit, period_str))
        unit = ''.join(filter(str.isalpha, period_str))
        if not num.isdigit():
            return None
        num = int(num)
        if unit == 'd':
            return num
        elif unit == 'mo':
            return num * 30
        elif unit == 'y':
            return num * 365
        else:
            return None

def descargar_datos_accion():
    ticker = input("Ingrese el símbolo del ticket que desea descargar (por ejemplo, TSLA, MSFT): ")
    formato = input("Ingrese el formato de archivo para guardar los datos ('csv' o 'excel'): ").lower()
    
    # Mostrar las opciones de periodo disponibles
    print("\nOpciones de periodo disponibles para 'period':")
    print("1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max")
    
    periodo = input("Ingrese el periodo de datos que desea descargar (por ejemplo, '1mo' para un mes): ").lower()
    periodo_dias = period_to_days(periodo)
    if periodo_dias is None:
        print("Periodo no válido. Por favor, ingrese un periodo válido.")
        return
    
    # Definir los intervalos y sus máximos días
    intervalos = {
        '1m': {'Barsize': '1 Minuto', 'Max Days': 7},
        '2m': {'Barsize': '2 Minutos', 'Max Days': 60},
        '5m': {'Barsize': '5 Minutos', 'Max Days': 60},
        '15m': {'Barsize': '15 Minutos', 'Max Days': 60},
        '30m': {'Barsize': '30 Minutos', 'Max Days': 60},
        '60m': {'Barsize': '1 Hora', 'Max Days': 730},
        '90m': {'Barsize': '90 Minutos', 'Max Days': 60},
        '1h': {'Barsize': '1 Hora', 'Max Days': 730},
        '1d': {'Barsize': '1 Día', 'Max Days': float('inf')},
        '5d': {'Barsize': '5 Días', 'Max Days': float('inf')},
        '1wk': {'Barsize': '1 Semana', 'Max Days': float('inf')},
        '1mo': {'Barsize': '1 Mes', 'Max Days': float('inf')},
        '3mo': {'Barsize': '3 Meses', 'Max Days': float('inf')}
    }
    
    # Filtrar intervalos válidos según el periodo
    intervalos_validos = []
    for intervalo, info in intervalos.items():
        max_dias = info['Max Days']
        if max_dias >= periodo_dias:
            intervalos_validos.append(intervalo)
    
    if not intervalos_validos:
        print("No hay intervalos disponibles para el periodo seleccionado.")
        return
    
    # Mostrar los intervalos válidos
    print("\nOpciones de intervalo disponibles para el periodo seleccionado:")
    for intervalo in intervalos_validos:
        info = intervalos[intervalo]
        print(f"{intervalo}: {info['Barsize']}")
    
    intervalo = input("Ingrese el intervalo de datos que desea (por ejemplo, '1d' para diario): ").lower()
    
    if intervalo not in intervalos_validos:
        print("Intervalo no válido para el periodo seleccionado.")
        return
    
    # Descargar datos históricos de la acción
    datos = yf.download(ticker, period=periodo, interval=intervalo)
    
    # Verificar si se obtuvieron datos
    if datos.empty:
        print("No se obtuvieron datos. Verifique el símbolo del ticket y los parámetros de periodo e intervalo.")
        return
    
    # Guardar datos en el formato seleccionado
    if formato == 'csv':
        nombre_archivo = f"{ticker}_{periodo}_{intervalo}_datos.csv"
        datos.to_csv(nombre_archivo)
        print(f"Datos guardados en {nombre_archivo}")
    elif formato == 'excel':
        nombre_archivo = f"{ticker}_{periodo}_{intervalo}_datos.xlsx"
        datos.to_excel(nombre_archivo)
        print(f"Datos guardados en {nombre_archivo}")
    else:
        print("Formato no reconocido. Por favor, elija 'csv' o 'excel'.")
    
if __name__ == "__main__":
    descargar_datos_accion()
