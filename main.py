import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from cargar_y_preprocesar import (
    cargar_y_preprocesar_v2, 
    clasificar_inmuebles_por_zona, 
    analizar_distribución_expensas,
    validar_datos
)

def entrenar_modelo(X, y, X_raw):
    """
    Entrenar modelo de predicción de precios con pipeline y validación cruzada
    """
    # División de datos
    X_train, X_test, y_train, y_test, X_raw_train, X_raw_test = train_test_split(
        X, y, X_raw, test_size=0.2, random_state=42
    )
    
    # Crear pipeline con escalado y modelo
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestRegressor(
            n_estimators=100,
            max_depth=8,
            min_samples_split=10,
            min_samples_leaf=4,
            random_state=42,
            n_jobs=-1
        ))
    ])
    
    # Entrenar pipeline
    pipeline.fit(X_train, y_train)
    
    # Predicciones
    y_pred = pipeline.predict(X_test)
    
    # Métricas de evaluación
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print("\n📊 Métricas de Evaluación:")
    print(f"MAE: ${mae:,.2f}")
    print(f"RMSE: ${rmse:,.2f}")
    print(f"R²: {r2:.2%}")
    
    return pipeline, X_test, y_test, y_pred, X_raw_test

def detectar_oportunidades(X_test, y_test, y_pred, X_raw_test, umbral_descuento=-5, 
                         precio_max=350000, min_superficie=35, max_antiguedad=35):
    """
    Detectar oportunidades de inversión con criterios más flexibles
    """
    resultados = X_raw_test.copy()
    resultados['Precio_Real'] = y_test
    resultados['Precio_Predicho'] = y_pred
    resultados['Diferencia_Porcentual'] = ((y_test - y_pred) / y_test * 100).round(2)
    
    # Filtrar oportunidades
    oportunidades = resultados[
        (resultados['Diferencia_Porcentual'] < umbral_descuento) &
        (resultados['Precio_Real'] < precio_max) &
        (resultados['Superficie Construida'] >= min_superficie)
    ]
    
    if len(oportunidades) > 0:
        print("\n🏠 OPORTUNIDADES DE INVERSIÓN DETECTADAS 🏠")
        print(f"\nSe encontraron {len(oportunidades)} oportunidades")
        print("\nTop 5 Oportunidades:")
        cols_display = ['Zona', 'Tipo de Propiedad', 'Superficie Construida', 
                       'Precio_Real', 'Precio_Predicho', 'Diferencia_Porcentual']
        print(oportunidades[cols_display].head().to_string())
    else:
        print("\n⚠️ No se encontraron oportunidades con los criterios actuales.")
        print("Sugerencia: Ajustar los criterios de búsqueda:")
        print(f"- Umbral de descuento actual: {umbral_descuento}%")
        print(f"- Precio máximo actual: ${precio_max:,.2f}")
        print(f"- Superficie mínima actual: {min_superficie} m²")
    
    return oportunidades

def generar_informe_mercado(dataset):
    """
    Generar informe comprehensivo del mercado inmobiliario
    """
    print("\n📍 ANÁLISIS DE MERCADO 📍")
    
    # Estadísticas generales
    print("\nEstadísticas Generales:")
    print(f"Total de propiedades: {len(dataset)}")
    print(f"Precio promedio: ${dataset['Precio'].mean():,.2f}")
    print(f"Superficie promedio: {dataset['Superficie Construida'].mean():.2f} m²")
    
    # Análisis por tipo de propiedad
    print("\nPrecios por Tipo de Propiedad:")
    tipo_stats = dataset.groupby('Tipo de Propiedad')['Precio'].agg(['count', 'mean', 'median'])
    print(tipo_stats.round(2))
    
    # Análisis por zona
    print("\nPrecios por Zona:")
    zona_stats = dataset.groupby('Zona')['Precio'].agg(['count', 'mean', 'median'])
    print(zona_stats.round(2))
    
    return tipo_stats, zona_stats

def cargar_y_preprocesar_v2(ruta_dataset):
    """
    Función de preprocesamiento mejorada
    """
    try:
        # Cargar el dataset
        dataset = pd.read_csv(ruta_dataset, encoding='latin-1')
        
        # Validar y limpiar datos
        dataset_limpio = validar_datos(dataset)
        
        # Guardar una copia del dataset original limpio
        X_raw = dataset_limpio.copy()
        
        # Preparar características para el modelo
        features_numericas = ['Superficie Total', 'Superficie Construida', 
                            'Número de Dormitorios', 'Número de Baños']
        
        # Codificar variables categóricas
        X = pd.get_dummies(dataset_limpio[['Tipo de Propiedad', 'Zona'] + features_numericas])
        y = dataset_limpio['Precio']
        
        print(f"Registros procesados exitosamente: {len(X)}")
        print(f"Rango de precios: ${y.min():,.2f} - ${y.max():,.2f}")
        print(f"Precio promedio: ${y.mean():,.2f}")
        
        return X, y, X_raw
        
    except Exception as e:
        print(f"Error en el preprocesamiento: {str(e)}")
        raise

def main():
    # Cargar y preprocesar datos
    X, y, X_raw = cargar_y_preprocesar_v2("dataset_inmuebles.csv")
    
    # Entrenar modelo
    modelo, X_test, y_test, y_pred, X_raw_test = entrenar_modelo(X, y, X_raw)
    
    # Detectar oportunidades
    oportunidades = detectar_oportunidades(
        X_test, 
        y_test, 
        y_pred,
        X_raw_test,
        umbral_descuento=-5,
        precio_max=350000,
        min_superficie=35
    )
    
    # Generar informe de mercado
    tipo_stats, zona_stats = generar_informe_mercado(X_raw)

if __name__ == "__main__":
    main()