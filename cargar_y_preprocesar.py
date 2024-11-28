import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

def limpiar_precio(precio):
    """
    Limpia y convierte valores de precio a float
    Maneja casos con símbolos de moneda y comas
    """
    if pd.isna(precio):
        return np.nan
    if isinstance(precio, str):
        # Eliminar símbolos de moneda, espacios y comas
        precio = precio.replace('$', '').replace(',', '').strip()
        try:
            return float(precio)
        except ValueError:
            return np.nan
    return float(precio) if not pd.isna(precio) else np.nan

def validar_datos(dataset):
    """
    Valida y limpia datos inválidos o extremos
    """
    # Copiar dataset para no modificar el original
    df = dataset.copy()
    
    # Convertir superficies a numérico
    df['Superficie Total'] = pd.to_numeric(df['Superficie Total'], errors='coerce')
    df['Superficie Construida'] = pd.to_numeric(df['Superficie Construida'], errors='coerce')
    
    # Eliminar filas donde superficie total o construida es 0 o NaN
    df = df[df['Superficie Total'].notna() & (df['Superficie Total'] > 0)]
    df = df[df['Superficie Construida'].notna() & (df['Superficie Construida'] > 0)]
    
    # Validar que superficie construida no sea mayor que superficie total
    df = df[df['Superficie Construida'] <= df['Superficie Total']]
    
    # Convertir y validar precio
    df['Precio'] = df['Precio'].apply(limpiar_precio)
    df = df[df['Precio'].notna() & (df['Precio'] > 0)]
    
    # Manejar expensas
    df['Expensas'] = df['Expensas'].apply(limpiar_precio)
    df['Expensas'] = df['Expensas'].fillna(0)  # Convertir NaN a 0 para expensas
    
    # Validar dormitorios y baños
    df['Número de Dormitorios'] = pd.to_numeric(df['Dormitorios'], errors='coerce')
    df['Número de Baños'] = pd.to_numeric(df['Numero de Banos'], errors='coerce')
    
    # Rellenar valores faltantes en dormitorios y baños con la mediana
    df['Número de Dormitorios'] = df['Número de Dormitorios'].fillna(df['Número de Dormitorios'].median())
    df['Número de Baños'] = df['Número de Baños'].fillna(df['Número de Baños'].median())
    
    # Eliminar valores extremos (outliers) usando IQR
    for columna in ['Precio', 'Superficie Total', 'Superficie Construida']:
        Q1 = df[columna].quantile(0.25)
        Q3 = df[columna].quantile(0.75)
        IQR = Q3 - Q1
        df = df[~((df[columna] < (Q1 - 1.5 * IQR)) | (df[columna] > (Q3 + 1.5 * IQR)))]
    
    print(f"Registros válidos después de la limpieza: {len(df)}")
    return df

def clasificar_inmuebles_por_zona(dataset):
    """
    Función para clasificar inmuebles por zona con manejo de errores mejorado
    """
    try:
        # Asegurarse de que los precios estén limpios
        dataset_limpio = dataset.copy()
        dataset_limpio['Precio'] = dataset_limpio['Precio'].apply(limpiar_precio)
        dataset_limpio['Superficie Construida'] = pd.to_numeric(dataset_limpio['Superficie Construida'], errors='coerce')
        
        # Eliminar filas con valores nulos en las columnas necesarias
        dataset_limpio = dataset_limpio.dropna(subset=['Zona', 'Precio', 'Superficie Construida'])
        
        # Agrupar por zona y calcular estadísticas
        zonas = dataset_limpio.groupby('Zona').agg({
            'Precio': ['mean', 'median', 'count'],
            'Superficie Construida': ['mean', 'median']
        }).round(2)
        
        # Renombrar columnas para mejor claridad
        zonas.columns = [
            'Precio_Medio', 
            'Precio_Mediana', 
            'Número_Inmuebles',
            'Superficie_Media', 
            'Superficie_Mediana'
        ]
        
        # Ordenar por número de inmuebles
        return zonas.reset_index().sort_values('Número_Inmuebles', ascending=False)
    
    except Exception as e:
        print(f"Error en clasificación por zona: {str(e)}")
        return pd.DataFrame()  # Retornar DataFrame vacío en caso de error

def analizar_distribución_expensas(dataset):
    """
    Función para analizar la distribución de expensas con manejo de errores mejorado
    """
    try:
        # Crear copia del dataset
        df = dataset.copy()
        
        # Limpiar precios de expensas
        df['Expensas_Limpio'] = df['Expensas'].apply(limpiar_precio)
        df['Expensas_Limpio'] = df['Expensas_Limpio'].fillna(0)
        
        # Calcular estadísticas
        total_inmuebles = len(df)
        expensas_nulas = df[df['Expensas_Limpio'] == 0].shape[0]
        expensas_no_nulas = df[df['Expensas_Limpio'] > 0]
        
        stats_expensas = {
            'Porcentaje_Expensas_Nulas': round((expensas_nulas / total_inmuebles) * 100, 2),
            'Promedio_Expensas': round(expensas_no_nulas['Expensas_Limpio'].mean(), 2),
            'Mediana_Expensas': round(expensas_no_nulas['Expensas_Limpio'].median(), 2),
            'Expensas_Máximas': round(df['Expensas_Limpio'].max(), 2),
            'Expensas_Mínimas': round(df[df['Expensas_Limpio'] > 0]['Expensas_Limpio'].min(), 2),
            'Total_Propiedades': total_inmuebles,
            'Propiedades_Con_Expensas': len(expensas_no_nulas)
        }
        
        return stats_expensas
    
    except Exception as e:
        print(f"Error en análisis de expensas: {str(e)}")
        return {}  # Retornar diccionario vacío en caso de error

def cargar_y_preprocesar_v2(ruta_dataset):
    """
    Función principal mejorada de preprocesamiento de datos de inmuebles
    """
    try:
        # Cargar el dataset
        dataset = pd.read_csv(ruta_dataset, encoding='latin-1')
        
        # Validar y limpiar datos
        dataset = validar_datos(dataset)
        
        # Columnas numéricas a procesar
        columnas_numericas = [
            'Superficie Total',
            'Superficie Construida',
            'Precio',
            'Expensas',
            'Número de Dormitorios',
            'Número de Baños'
        ]
        
        # Columnas categóricas
        categorical_features = ['Tipo de Propiedad', 'Zona']
        
        # Codificar variables categóricas
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        cat_cols_exists = [col for col in categorical_features if col in dataset.columns]
        
        if cat_cols_exists:
            encoded_features = encoder.fit_transform(dataset[cat_cols_exists])
            encoded_df = pd.DataFrame(
                encoded_features,
                columns=encoder.get_feature_names_out(cat_cols_exists)
            )
            dataset = pd.concat([dataset, encoded_df], axis=1)
            columnas_numericas.extend(encoded_df.columns)
        
        # Características adicionales
        dataset['Precio_por_m2'] = dataset['Precio'] / dataset['Superficie Construida']
        columnas_numericas.append('Precio_por_m2')
        
        # Procesar fecha de publicación
        dataset['Fecha_Publicacion'] = pd.to_datetime(
            dataset['Fecha de Publicacion'], 
            dayfirst=True, 
            errors='coerce'
        )
        dataset['Antiguedad'] = (
            pd.Timestamp.now() - dataset['Fecha_Publicacion']
        ).dt.days / 365.25
        columnas_numericas.append('Antiguedad')
        
        # Preparar X e y
        X = dataset[columnas_numericas]
        y = dataset['Precio']
        
        # Eliminar filas donde y es NaN
        mask_valid = y.notna()
        X = X[mask_valid]
        y = y[mask_valid]
        
        # Imputar valores faltantes con la mediana solo en X
        imputer = SimpleImputer(strategy='median')
        X_imputed = pd.DataFrame(
            imputer.fit_transform(X),
            columns=X.columns
        )
        
        print(f"Registros procesados exitosamente: {len(X_imputed)}")
        print(f"Rango de precios: ${y.min():,.2f} - ${y.max():,.2f}")
        print(f"Precio promedio: ${y.mean():,.2f}")
        
        return X_imputed, y
        
    except Exception as e:
        print(f"Error en el preprocesamiento: {str(e)}")
        raise