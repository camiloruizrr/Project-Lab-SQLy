import os
import weaviate
from langchain_community.embeddings import HuggingFaceEmbeddings
# No usaremos Weaviate de langchain, pero la mantenemos por si el usuario quiere usarla después
# from langchain_community.vectorstores import Weaviate 
# Importaciones de Weaviate V4 eliminadas:
# from weaviate.classes.config import Property, DataType, Configure
# from weaviate.classes.data import DataObject

# --- CONFIGURACIÓN ---
# Asegúrate de que esta ruta sea correcta
DICCIONARIOS_DIR = r"C:\Users\camil\Downloads\AgenteSQL\Diccionarios"
WEAVIATE_HOST = "localhost"
WEAVIATE_HTTP_PORT = "8080"
WEAVIATE_CLASS_NAME = "SQLContexto"
# En V3, el puerto gRPC no se usa en la conexión del cliente
# WEAVIATE_GRPC_PORT = "50051" 
# --- FIN CONFIGURACIÓN ---

# 1. Cargar Embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

docs = []
metadatas = []

# 2. Leer Documentos
print("Leyendo archivos de contexto...")
try:
    for file in os.listdir(DICCIONARIOS_DIR):
        if file.endswith(".txt"):
            file_path = os.path.join(DICCIONARIOS_DIR, file)
            with open(file_path, "r", encoding="utf-8") as f:
                # Usamos el archivo completo como un solo "chunk" o documento.
                contenido = f.read()
                docs.append(contenido) 
                metadatas.append({"nombre_diccionario": file})
    print(f"Archivos de contexto leídos: {len(docs)}")
    if not docs:
        print(f"Advertencia: No se encontraron archivos '.txt' en la carpeta: {DICCIONARIOS_DIR}")
        exit()
except FileNotFoundError:
    print(f"Error: El directorio '{DICCIONARIOS_DIR}' no se encontró. Verifica la ruta.")
    exit()


# 3. CONEXIÓN A WEAVIATE (MODIFICADO para V3)
print(f"Conectándose a Weaviate en: http://{WEAVIATE_HOST}:{WEAVIATE_HTTP_PORT}")
client = None
try:
    # Usamos el constructor weaviate.Client para V3
    client = weaviate.Client(
        url=f"http://{WEAVIATE_HOST}:{WEAVIATE_HTTP_PORT}",
    )
    
    # En V3, podemos verificar la salud de la API
    if not client.is_live() or not client.is_ready():
        raise Exception("El cliente de Weaviate no está conectado o no está listo.")
    print("Conexión con Weaviate exitosa.")
except Exception as e:
    print(f"Error al conectar con Weaviate. Asegúrate de que el servidor esté corriendo en {WEAVIATE_HOST}:{WEAVIATE_HTTP_PORT}.")
    print(f"Error detallado: {e}")
    # En V3, no hay un método .close() explícito para el cliente HTTP, pero cerramos si existe
    # if client: client.close()
    exit()

# 4. GESTIÓN Y CREACIÓN EXPLÍCITA DE LA CLASE (MODIFICADO para V3)
print(f"Gestionando la clase '{WEAVIATE_CLASS_NAME}'...")

# Definición del esquema (clase) como un diccionario para V3
schema_class = {
    "class": WEAVIATE_CLASS_NAME,
    "properties": [
        # Propiedad requerida para el contenido del texto
        {"name": "text", "dataType": ["text"]},
        # Propiedad para los metadatos
        {"name": "nombre_diccionario", "dataType": ["text"]},
    ],
    # IMPORTANTE: Desactivamos la vectorización interna de Weaviate 
    # Usando vectorizer: 'none' y moduleConfig para evitar que el vectorizer por defecto actúe.
    "vectorizer": "none",
    "moduleConfig": {
        "text2vec-contextionary": { # Si este es el módulo por defecto que usas
            "skip": True 
        },
        # Asegúrate de que cualquier otro vectorizer por defecto también esté omitido si es necesario
    }
}

# Opcional: Eliminar la clase si ya existe para una carga limpia
if client.schema.exists(WEAVIATE_CLASS_NAME):
    print(f"Eliminando la clase existente: {WEAVIATE_CLASS_NAME}")
    client.schema.delete_class(WEAVIATE_CLASS_NAME)


try:
    client.schema.create_class(schema_class)
    print(f"Clase '{WEAVIATE_CLASS_NAME}' creada exitosamente con vectorizador 'none'.")

except Exception as e:
    print(f"Error al crear la clase en Weaviate: {e}")
    exit()

# 5. INYECCIÓN DE DATOS MANUALMENTE (Vectorización + batch V3)
print("Vectorizando e inyectando datos en Weaviate manualmente. Esto puede tomar un momento...")

# 5a. Vectorizar todos los textos
print(f"Generando embeddings para {len(docs)} documentos...")
try:
    vectors = embeddings.embed_documents(docs)
    print(f"Embeddings generados para {len(vectors)} documentos.")
except Exception as e:
    print(f"Error al generar embeddings: {e}")
    exit()

# 5b. Iniciar el modo batch y añadir los objetos (MODIFICADO para V3)
try:
    # ----------------------------------------------------
    # Configuración Mínima del Batch
    # Inicializamos el batch
    client.batch.configure(batch_size=100) 
    # ----------------------------------------------------
    
    print(f"Inyectando {len(docs)} objetos...")
    for text, metadata, vector in zip(docs, metadatas, vectors):
        properties = {
            "text": text,
            "nombre_diccionario": metadata["nombre_diccionario"]
        }
        
        # Añadir a la cola del lote
        client.batch.add_data_object(
            data_object=properties,
            class_name=WEAVIATE_CLASS_NAME,
            vector=vector
        )
    
    # ----------------------------------------------------
    # ¡LA LÍNEA CLAVE! Esto fuerza el envío de los objetos restantes al servidor.
    insert_results = client.batch.flush()
    # ----------------------------------------------------

    print(f"\nVectorstore de Weaviate generado y cargado en la clase: {WEAVIATE_CLASS_NAME}.")
    
    # ----------------------------------------------------
    # MODIFICACIÓN: Verificar si insert_results no es None antes de iterar
    if insert_results is not None:
        # Opcional: Verificar si hubo errores en el lote
        # Nota: La estructura para verificar errores en V3 es un poco compleja, esta es una aproximación:
        error_found = False
        for res in insert_results:
            if 'result' in res and 'errors' in res['result'] and res['result']['errors']:
                error_found = True
                print(f"Error en documento: {res.get('id', 'N/A')}. Detalles: {res['result']['errors']}")

        if error_found:
            print("Advertencia: Se encontraron errores durante la inyección de datos. Revisa los detalles de arriba.")
    # ----------------------------------------------------
    
    # 6. VERIFICACIÓN DEL CONTEO (V3)
    aggregate_result = client.query.aggregate(WEAVIATE_CLASS_NAME).with_meta_count().do()
    count = aggregate_result['data']['Aggregate'][WEAVIATE_CLASS_NAME][0]['meta']['count']
    print(f"Total de documentos cargados: {count}")

except Exception as e:
    print(f"Error durante la inyección de datos con batch: {e}")

# 7. Cierra la conexión (No es necesario en V3 para cliente HTTP, pero lo mantenemos como buenas prácticas si se usa en otros contextos)
print("Proceso finalizado.")