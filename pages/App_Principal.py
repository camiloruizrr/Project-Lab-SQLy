# pages/Aplicacion_Principal.py
# ================================================
# Interfaz visual para el Agente SQL usando Streamlit,
# asegurada con autenticación.
# ================================================

import streamlit as st
import streamlit_authenticator as stauth
import pandas as pd
from sqlalchemy import create_engine
from groq import Groq
from langchain_community.utilities import SQLDatabase 
import re
import warnings
from io import BytesIO
from langchain_community.vectorstores import FAISS
import weaviate
from langchain_community.vectorstores import Weaviate
from langchain_community.embeddings import HuggingFaceEmbeddings
from difflib import get_close_matches
import json
import os 
from importlib.metadata import version, PackageNotFoundError
import sys
import importlib.metadata

from dotenv import load_dotenv


apiKey = os.environ.get( "apiKey")

warnings.filterwarnings("ignore")

# --- 1. CONFIGURACIÓN DE SEGURIDAD (Necesaria para Logout) ---

# ✅ NUEVA CONFIGURACIÓN: EXPANDIR BARRA LATERAL PARA MOSTRAR LOGOUT
st.set_page_config(
    page_title="Agente SQL",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded" 
)

# Usamos la misma configuración de cookie que Home.py
config_data = {
    'credentials': {
        'usernames': {
            'camilo': {
                'email': 'camiloruiz2576@gmail.com',
                'name': 'Camilo Ruiz',
                'password': '$2b$12$Auuz.HMO6p9DS2eU7rJBCOepDFbW7UJ9X5TQytWU4efkSHbheeVo6'
            },
            'rbriggs': {
                'email': 'rbriggs@example.com',
                'name': 'Rebecca Briggs',
                'password': '$2b$12$4O.j.M6l.M3.V1x.i.Y7s7w.P8i5m.N.M4g5f5t.X3v.V4r4e8y.w9a'
            }
        }
    },
    'cookie': {
        'expiry_days': 30,
        # CLAVE CORREGIDA para coincidir con Home.py
        'key': 'sql_agent_cookie_key_v2', 
        'name': 'sql_agent_cookie'
    },
    'preauthorized': {
        'emails': ['admin@example.com']
    }
}

credentials = config_data['credentials']
cookie_name = config_data['cookie']['name']
cookie_key = config_data['cookie']['key']
cookie_expiry_days = config_data['cookie']['expiry_days']
preauthorized_emails = config_data['preauthorized']['emails']

authenticator = stauth.Authenticate(
    credentials,
    cookie_name,
    cookie_key,
    cookie_expiry_days,
    preauthorized=preauthorized_emails
)


# --- 2. CONTROL DE SEGURIDAD CRÍTICO ---
if not st.session_state.get("authentication_status"):
    # Si el usuario no está autenticado, redirigir a la página de inicio
    st.error("You do not have permission to access this page. Please log in.")
    st.switch_page("Home.py") 
    st.stop()

# --- INICIALIZACIÓN DE LA MEMORIA DEL CHAT ---
if "messages" not in st.session_state:
    st.session_state.messages = [] # [{role: "user", content: "..."}]
if "sql_history" not in st.session_state:
    st.session_state.sql_history = {} # {user_message_id: sql_text}    
    
# --- 3. BARRA LATERAL Y LOGOUT ---

# Título y Bienvenida
#st.title("Bienvenido al 🤖 Agente SQL")
#st.caption(f"Hola, {st.session_state['name']} (Usuario: {st.session_state['username']})")

with st.sidebar:
    st.subheader("Session Control")
    
    # Muestra el nombre del usuario y su ID de sesión
    st.info(f"Usser: **{st.session_state['name']}**\n\nID: **{st.session_state['username']}**")
    
    # --- Solución Robusta para el KeyError ---
    logout_button = False
    try:
        # Intentamos ejecutar el logout.
        logout_button = authenticator.logout('Log Out', 'sidebar')
        
    except KeyError as e:
        # Si el KeyError es la cookie, forzamos el indicador de botón presionado.
        if 'sql_agent_cookie' in str(e):
            logout_button = True 
        else:
            # Si es otro KeyError, lo relanzamos
            raise e
            
    except Exception as e:
        # Captura cualquier otro error inesperado durante el logout
        st.error(f"Error inesperado al cerrar sesión: {e}")
        st.stop()
        
    # --- Lógica de Redirección (SOLO con st.switch_page) ---
    if logout_button:
        authentication_status = None
        st.session_state["authentication_status"] = None
        st.session_state["name"] = None
        st.session_state["username"] = None        
        st.toast("🔒 Session closed successfully.", icon="👋")
        
        # 🚨 LIMPIEZA ADICIONAL: Limpiamos explícitamente el estado del chat 
        # para garantizar que no se muestre ningún contenido al recargar.
        if "messages" in st.session_state:
             del st.session_state["messages"]
        if "sql_history" in st.session_state:
             del st.session_state["sql_history"]

        # 🚨 SOLUCIÓN FINAL: Usamos SOLO st.switch_page, el mecanismo nativo
        # de Streamlit para garantizar la navegación a la página de inicio.
        authentication_status = None
        st.session_state["authentication_status"] = None
        st.session_state["name"] = None
        st.session_state["username"] = None
        st.switch_page("Home.py")
        
        # Detenemos la ejecución.
        st.stop()
        
    st.divider()
#    st.markdown("Otros controles de aplicación aquí...")

# with st.sidebar:
#     st.subheader("Session Control")
#     # Mostrar el nombre de usuario
#     st.info(f"Active Session: **{st.session_state['username']}**")
    
#     # 🚨 CORRECCIÓN CLAVE: Usamos st.button() simple para disparar la acción
#     # y llamamos a authenticator.logout() sin ubicación ('main'/'sidebar')
#     # para forzar la eliminación de la cookie inmediatamente.
#     if st.session_state.get("authentication_status"):
#         # st.button() garantiza que la acción ocurre en el primer re-run después del click
#         if st.button('Log Out', key='manual_logout_button'):
#             # 1. Llamar al método logout para borrar la cookie.
#             # No se requiere ubicación ('main' o 'sidebar') si ya estamos en un contenedor (sidebar)
#             # o si usamos st.button() para forzar la acción. 
#             # Si esto falla, Streamlit lo maneja mejor que si falla el componente interno.
            
#             try:
#                 # Intentamos usar el logout de Streamlit Authenticator para que borre la cookie
#                 authenticator.logout('Log Out', 'sidebar') 
#             except KeyError:
#                 # 💥 IGNORAMOS el KeyError: Si la clave no está, significa que ya está borrada.
#                 pass
#             except Exception as e:
#                 st.error(f"Unexpected error while trying to log out.: {e}")
            
#             # 2. Borrar las variables de estado. ESTO ES LO CRÍTICO.
#             st.session_state["authentication_status"] = None
#             st.session_state["name"] = None
#             st.session_state["username"] = None
#             st.toast("🔒 Session closed successfully.", icon="👋")
            
#             # 3. Redirigir.
#             st.switch_page("Home.py")
 
# ----------------------------------------------------

# ---------- CONFIG ----------
GROQ_API_KEY = apiKey 
MODEL_NAME = "llama-3.1-8b-instant"
# DICT_PATH no se usa, lo mantengo comentado
# DICT_PATH = r"C:\Users\camil\Downloads\AgenteSQL\Diccionario_dbSales.txt" 
DB_URI = "mssql+pyodbc://sa:TuContraseñaSegura123@localhost:1433/salesdb?driver=ODBC+Driver+17+for+SQL+Server"
#DB_URI = "mssql+pyodbc://sa:TuContraseñaSegura123@sqlt2022:1433/salesdb?driver=ODBC+Driver+17+for+SQL+Server"


# --- WEAVIATE CONFIG ---
WEAVIATE_HOST = os.environ.get("WEAVIATE_HOST", "localhost")
WEAVIATE_HTTP_PORT = os.environ.get("WEAVIATE_HTTP_PORT", "8080")
WEAVIATE_URL = f"http://{WEAVIATE_HOST}:{WEAVIATE_HTTP_PORT}"
WEAVIATE_CLASS_NAME = os.environ.get("WEAVIATE_CLASS_NAME", "SQLContexto")
# --- FIN WEAVIATE CONFIG ---


def debug_weaviate_client_version():
    """Imprime información de la versión y la ubicación del módulo weaviate."""
    st.info("--- DIAGNÓSTICO DEL CLIENTE WEAVIATE ---")
    try:
        # Intenta obtener la versión instalada a través de metadatos
        weaviate_version = version("weaviate-client")
        st.info(f"Versión de 'weaviate-client' (metadata): **{weaviate_version}**")
    except PackageNotFoundError:
        st.error("ERROR: No se pudo encontrar la versión de 'weaviate-client' en el entorno de ejecución.")
        weaviate_version = "Desconocida"

    try:
        # Muestra la ruta del archivo que Python realmente está importando
        weaviate_module_path = sys.modules['weaviate'].__file__
        st.info(f"Ruta del módulo importado: `{weaviate_module_path}`")
    except KeyError:
        st.error("ERROR: El módulo 'weaviate' no parece haber sido cargado.")

    st.info("-------------------------------------------")
    return weaviate_version


# ---------- Cargar diccionario (FAISS Vectorstore) ----------

VECTOR_DB_PATH = r"C:\Users\camil\Downloads\AgenteSQL\vectorstore"


# Cachear la carga de embeddings para mejorar el rendimiento en Streamlit
# @st.cache_resource
# def load_vectorstore_components():
#     """Carga los embeddings y el vectorstore una sola vez."""
#     embeddings = HuggingFaceEmbeddings(
#         model_name="sentence-transformers/all-MiniLM-L6-v2",
#         model_kwargs={"device": "cpu"}
#     )
#     vectorstore = FAISS.load_local(
#         VECTOR_DB_PATH,
#         embeddings,
#         allow_dangerous_deserialization=True
#     )
#     # cambio de k para asegurar que consulte todos los documentos
#     retriever = vectorstore.as_retriever(search_kwargs={"k": 5}) 
#     return retriever

# retriever = load_vectorstore_components()

# ---------- Cargar diccionario (AHORA WEAVIATE) ----------

# Cachear la conexión a Weaviate y la inicialización del Retriever
@st.cache_resource
@st.cache_resource
def load_vectorstore_components():
    """
    Conecta a Weaviate y carga el modelo de embeddings (Lógica PURA).
    Retorna el cliente de Weaviate, el modelo de embeddings, y un estado/mensaje.
    """
    
    # --- Cargar embeddings ---
    try:
        import torch
        # Definimos el dispositivo de forma explícita para evitar el error 'meta tensor'
        DEVICE = torch.device("cpu") 
    except (ImportError, AttributeError):
        DEVICE = "cpu"
    
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": DEVICE} 
        )
    except Exception as e:
        # Devolver error si los embeddings fallan
        return None, None, 500, f"Error loading embeddings: {e}"

    try:
        # 1. Inicializar cliente Weaviate
        #client = weaviate.Client("http://localhost:8080")
        #client = weaviate.Client("http://serene_liskov:8080")
        client = weaviate.Client(WEAVIATE_URL)

        # 2. Verificamos conexión
        if not client.is_live():
            # Devolver error si la conexión falla
            return None, None, 400, "Weaviate is not responding. Check the Docker container."

        # Devolvemos el cliente, el modelo de embeddings, y un estado de éxito
        return client, embeddings, 200, "Connection to Weaviate successful."

    except Exception as e:
        # Devolver error si hay una excepción general de conexión
        return None, None, 500, f"Error connecting to Weaviate: {e}"

# --- Manejo del cliente y feedback (FUERA DEL CACHE) ---

# Obtenemos el cliente, el modelo de embeddings, el estado y el mensaje.
weaviate_client, embeddings_model, status_code, status_message = load_vectorstore_components()

# Manejo del feedback basado en el resultado:
if status_code != 200:
    st.error(f"❌ Error loading context ({status_code}): {status_message}")
    st.warning(
        f"Make sure Weaviate is running on {WEAVIATE_URL} and that the class '{WEAVIATE_CLASS_NAME}' exist."
    )
    st.stop()
else:
    # Solo mostramos el éxito si la conexión fue buena
    st.success(status_message)

if weaviate_client is None:
    st.warning("Failed to load context (Weaviate). Please verify that Docker/Weaviate is running.")
    st.stop()

# ---------- Conectar a DB ----------
# Cachear la conexión DB
@st.cache_resource
def get_db_connection(db_uri):
    """Crea y retorna la conexión a la DB."""
    try:
        db = SQLDatabase.from_uri(db_uri)
        return db
    except Exception as e:
        raise RuntimeError(f"Error connecting to the database: {e}")

try:
    db = get_db_connection(DB_URI)
except RuntimeError as e:
    st.error(str(e))
    st.stop()


# ---------- Cliente Groq ----------
client = Groq(api_key=GROQ_API_KEY)

# ---------- Prompt base ----------
def construir_prompt_con_contexto(pregunta):
    """
    Realiza una búsqueda RAG manual usando el cliente Weaviate v3.x 
    y construye el prompt con el contexto.
    """
    
    # 1. CRÍTICO: Vectorizar la pregunta del usuario usando el modelo de embeddings
    try:
        # Los modelos HuggingFaceEmbeddings devuelven una lista de vectores (aunque solo es uno)
        query_vector = embeddings_model.embed_query(pregunta)
    except Exception as e:
        st.error(f"Error generating the query vector with HuggingFaceEmbeddings: {e}")
        return (
             "INTERNAL RAG ERROR: Failed to vectorize the question.", 
             []
         )
    
    # 2. Parámetros de búsqueda nearVector (sintaxis Weaviate v3)
    # Ahora usamos near_vector en lugar de near_text
    near_vector_filter = {
         "vector": query_vector, # Pasamos el vector que generamos
    }
    
    resultados = []
    
    try:
        # 3. Ejecutar la consulta directamente con el cliente Weaviate
        response = weaviate_client.query.get(
            class_name=WEAVIATE_CLASS_NAME, 
            # Asumimos que el contenido de la tabla/columna está en la propiedad 'text'
            properties=["text", "nombre_diccionario"] 
        ).with_near_vector(
            near_vector_filter # Usamos near_vector con el vector generado
        ).with_limit(
            5 # k=5 como en el retriever anterior
        ).do()
        
        # 4. Procesar los resultados (misma lógica)
        if 'data' in response and 'Get' in response['data'] and WEAVIATE_CLASS_NAME in response['data']['Get']:
            for obj in response['data']['Get'][WEAVIATE_CLASS_NAME]:
                if 'text' in obj:
                    # Creamos un objeto simple que se comporta como Document de LangChain
                    doc = type('Document', (object,), {
                        'page_content': obj['text'], 
                        'metadata': {"source": obj.get('nombre_diccionario', 'desconocido')}
                    })()
                    resultados.append(doc)

    except Exception as e:
        st.error(f"Weaviate error while searching for context (V3 nearVector query): {e}")
        return (
            "INTERNAL RAG ERROR: Failed to search for context in Weaviate.", 
            []
        )

    # 5. Construir Prompt (misma lógica)
    if not resultados:
        return (
            "THERE IS NO INFORMATION IN THE DICTIONARY FOR THIS QUERY", 
            []
        )
    
    # Concatenamos el contenido de los documentos relevantes
    contexto = "\n---\n".join([r.page_content for r in resultados])

    # ✅ Recortamos contexto para evitar que el prompt se vuelva enorme
    contexto_recortado = contexto[:17400] # ajustable

    prompt = f"""
Eres un experto en SQL Server.

Usa el siguiente contexto para entender la estructura de la base de datos:

{contexto_recortado}


MANDATORY RULES:

Do not make up table or column names.

If one table contains only IDs and another contains detailed information (such as names), you must use a JOIN between them.

Use only the names EXACTLY as they appear in the context.

The correct database names you must use from the dictionaries are:

"salesdb": "sales of products information" 

"BaseballData": "baseball information and statistics"

"EmployeeCaseStudy": "employee information"

Always use the format database.schema.table

Remember to refering the table to avoid ambiguity

Do not generate UPDATE, INSERT, or DELETE statements.

Only use table names that you find in the context; do not invent tables that do not exist there.


### Ejemplo de formato correcto:
SELECT TOP 10 col1, col2
FROM MiBase.dbo.MiTabla;

### Pregunta del usuario:
{pregunta}

### Devuelve solo la consulta SQL:
"""
    # Devolvemos el prompt y los documentos emulados para el post-procesamiento
    return prompt, resultados

# --- FUNCIONES DE CORRECCIÓN/POST-PROCESAMIENTO ---

def parse_context_tables_from_docs(docs):
    """
    docs: lista de Documentos u objetos con .page_content
    Retorna mapping {table_simple_lower: full_name} y table_columns {table_simple: [cols]}
    """
    mapping = {}
    table_columns = {}

    combined = ""
    # aceptar lista de strings o lista de objetos con 'page_content'
    for d in docs:
        contenido = d.page_content if hasattr(d, "page_content") else (d if isinstance(d, str) else "")
        combined += "\n" + contenido

        # intentamos parsear JSON por documento
        try:
            data = json.loads(contenido)
            dbname = data.get("database_name") or data.get("Database") or data.get("database") or ""
            schema = data.get("Schema") or data.get("schema") or "dbo"
            tables = data.get("tables") or data.get("Tables") or []
            for t in tables:
                if isinstance(t, dict) and "table_name" in t:
                    tname = t["table_name"]
                    table_simple = tname.lower()
                    full = f"{dbname}.{schema}.{tname}" if dbname else f"{schema}.{tname}"
                    mapping[table_simple] = full
                    cols = [c["column_name"] for c in t.get("columns", []) if isinstance(c, dict) and "column_name" in c]
                    table_columns[table_simple] = cols
                elif isinstance(t, dict):
                    # formato alterno: { "People": { "PlayerID": {...}, ... } }
                    for k, v in t.items():
                        tname = k
                        table_simple = tname.lower()
                        full = f"{dbname}.{schema}.{tname}" if dbname else f"{schema}.{tname}"
                        mapping[table_simple] = full
                        cols = []
                        if isinstance(v, dict):
                            # v puede tener column definitions
                            for ck in v.keys():
                                cols.append(ck)
                        table_columns[table_simple] = cols
        except Exception:
            # no JSON — lo manejamos abajo con heurísticas
            pass

    # heurística: buscar ocurrencias tipo "table_name": "X"
    try:
        tables = re.findall(r'"table_name"\s*:\s*"([^"]+)"', combined)
        for tn in tables:
            key = tn.lower()
            if key not in mapping:
                # intentar extraer columns block
                cols_block = re.search(rf'"table_name"\s*:\s*"{re.escape(tn)}"\s*,\s*"columns"\s*:\s*\[([^\]]+)\]', combined, re.DOTALL)
                cols = []
                if cols_block:
                    cols = re.findall(r'"column_name"\s*:\s*"([^"]+)"', cols_block.group(1))
                # Si no conocemos dbname, dejamos schema.table
                dbname_match = re.search(r'"database_name"\s*:\s*"([^"]+)"', combined)
                db = dbname_match.group(1) if dbname_match else ""
                schema_match = re.search(r'"Schema"\s*:\s*"([^"]+)"', combined)
                sc = schema_match.group(1) if schema_match else "dbo"
                full = f"{db}.{sc}.{tn}" if db else f"{sc}.{tn}"
                mapping[key] = full
                table_columns[key] = cols
    except Exception:
        pass

    return mapping, table_columns


def find_best_table_for_token(token, mapping):
    """
    token: nombre simple de tabla (People, players, master, etc)
    mapping: dict table_simple -> full_name
    Retorna la key exacta o la mejor coincidencia fuzzy.
    """
    token_clean = re.sub(r'[\[\]`"]', '', token).split('.')[-1].lower()
    if token_clean in mapping:
        return token_clean
    # intentar singular/plural
    if token_clean.endswith('s') and token_clean[:-1] in mapping:
        return token_clean[:-1]
    if (token_clean + 's') in mapping:
        return token_clean + 's'
    # fuzzy match
    matches = get_close_matches(token_clean, list(mapping.keys()), n=1, cutoff=0.7)
    if matches:
        return matches[0]
    return None


def qualify_table_names_improved(sql_text, docs):
    """
    Asegura que los nombres de tablas estén en formato Database.Schema.Table usando el contexto RAG.
    docs: lista de Document (lo que devuelve retriever.invoke)
    """
    mapping, _ = parse_context_tables_from_docs(docs)

    def find_full_name(token):
        # si ya viene con puntos retorna tal cual
        t_clean = re.sub(r'[\[\]`"]', '', token.strip())
        if '.' in t_clean:
            return t_clean
        best = find_best_table_for_token(t_clean, mapping)
        return mapping.get(best) if best else None

    def replace_match(m):
        keyword = m.group(1)
        table_part = m.group(2)
        parts = table_part.split()
        table_name = parts[0]
        alias = " " + " ".join(parts[1:]) if len(parts) > 1 else ""
        full = find_full_name(table_name)
        return f"{keyword} {full}{alias}" if full else m.group(0)

    pattern = re.compile(r'\b(FROM|JOIN)\s+([A-Za-z0-9_\.\[\]`"]+(?:\s+(?:AS\s+)?[A-Za-z0-9_]+)?)', flags=re.IGNORECASE)
    return pattern.sub(replace_match, sql_text)


def get_table_columns_from_docs(docs):
    _, table_columns = parse_context_tables_from_docs(docs)
    return table_columns


def normalize_aliases(sql_text):
    # Corrige el error "AS AS"
    sql_text = re.sub(r'\bAS\s+AS\b', 'AS', sql_text, flags=re.IGNORECASE)
    sql_text = re.sub(r'\s+', ' ', sql_text).strip()
    return sql_text


def fix_common_column_names_improved(sql_text, docs, pregunta_text=""):
    import re
    table_columns = get_table_columns_from_docs(docs)

    # construir alias_map igual que antes pero permitir full table names
    alias_map = {}
    for m in re.finditer(r'\b(FROM|JOIN)\s+([A-Za-z0-9_\.\[\]`"]+)(?:\s+(?:AS\s+)?([A-Za-z0-9_]+))?', sql_text, flags=re.IGNORECASE):
        table_full = m.group(2)
        alias = m.group(3)
        # decidir table_simple: el último token del full
        table_simple = re.sub(r'[\[\]`"]', '', table_full).split('.')[-1].lower()
        if alias:
            alias_map[alias.lower()] = table_simple # Usar alias en minúsculas

    def replace(match):
        alias, col = match.group(1), match.group(2)
        alias_lower = alias.lower()
        col_lower = col.lower()
        if alias_lower in alias_map:
            table_simple = alias_map[alias_lower]
            cols = table_columns.get(table_simple, [])
            cols_lower = [c.lower() for c in cols]
            if col_lower in cols_lower:
                return f"{alias}.{cols[cols_lower.index(col_lower)]}"
            # Si 'name' y existen nameFirst/nameLast devolver ambos si la pregunta pide "nombres"
            if col_lower == "name":
                if ("namefirst" in cols_lower and "namelast" in cols_lower) or ("firstname" in cols_lower and "lastname" in cols_lower):
                    # preferir nameFirst/nameLast
                    candidates = []
                    # Lógica para encontrar el mejor nombre/apellido
                    for pref in ["nameFirst", "firstName", "namefirst", "firstname"]:
                        if pref.lower() in cols_lower:
                            candidates.append(cols[cols_lower.index(pref.lower())])
                            break
                    for pref in ["nameLast", "lastName", "namelast", "lastname"]:
                        if pref.lower() in cols_lower:
                            candidates.append(cols[cols_lower.index(pref.lower())])
                            break
                    if candidates:
                        # Si encontramos ambos, devolvemos ambos, si no, solo el que tengamos
                        return f"{alias}.{candidates[0]}, {alias}.{candidates[1]}" if len(candidates) > 1 else f"{alias}.{candidates[0]}"
            # si no encontramos, intentar fuzzy en cols
            matches = get_close_matches(col_lower, cols_lower, n=1, cutoff=0.7)
            if matches:
                real = cols[cols_lower.index(matches[0])]
                return f"{alias}.{real}"
        return match.group(0)

    # El patrón busca "alias.columna"
    return re.sub(r'\b([A-Za-z0-9_]+)\.([A-Za-z0-9_]+)\b', replace, sql_text)


def postprocess_sql(sql_text, docs, pregunta_text=""): 
    """Aplica post-procesamiento al SQL generado. Ahora recibe los docs."""
    sql_text = normalize_aliases(sql_text)
    # Pasamos los docs (lista de resultados de retriever)
    sql_text = fix_common_column_names_improved(sql_text, docs, pregunta_text) 
    return sql_text


# ---------- Generar SQL ------------

def generar_respuesta_con_groq(messages):
    """
    Genera la respuesta del Agente Groq (puede ser SQL o texto).
    messages: el historial completo de st.session_state.messages
    """
    
    # El LLM solo debe ver los mensajes de texto, no los DataFrames.
    # El último mensaje del usuario es la pregunta actual o el error/corrección.
    pregunta_actual = messages[-1]["content"]
    
    # 1) Construimos prompt con contexto RAG
    # Usamos la última pregunta del usuario para el RAG
    prompt_rag, docs = construir_prompt_con_contexto(pregunta_actual)
    
    # 🧪 DEPURA AQUÍ: Muestra el prompt RAG en la barra lateral
    with st.sidebar:
        st.markdown("---")
        st.markdown(f"**Documents found:** {len(docs)}")
        st.markdown("**Base RAG prompt:**")
        # El prompt RAG es el "system" prompt de la llamada a Groq
        st.code(prompt_rag, language='markdown') 
            
    # Creamos la lista de mensajes para Groq, incluyendo el SYSTEM PROMPT (RAG)
    groq_messages = [{"role": "system", "content": prompt_rag}]

    text_messages = [m for m in messages if isinstance(m["content"], str)]

    MAX_HISTORY_PAIRS = 5

    text_only_messages = [m for m in st.session_state.messages if isinstance(m["content"], str)]

    history_slice = text_only_messages[-(2 * MAX_HISTORY_PAIRS):]
    
    # Añadimos el historial de chat (excluyendo el SYSTEM prompt que ya lo pusimos)
    # y excluyendo cualquier mensaje que no sea texto (como DataFrames)
    for message in history_slice:
            # Groq no acepta el rol 'data', solo 'user' y 'assistant'
            role = "user" if message["role"] == "user" else "assistant"
            groq_messages.append({"role": role, "content": message["content"]})
        
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=groq_messages, # Usamos el historial completo
            temperature=0.0
        )

        sql_text = response.choices[0].message.content.strip()
        
        # --- Post-procesamiento para extraer y limpiar SQL ---
        sql_text_clean = sql_text.replace("```sql", "").replace("```", "").strip()
        
        # Intentamos extraer el SQL si viene envuelto
        match = re.search(r'\bSELECT\b', sql_text_clean, re.IGNORECASE)
        if match:
            sql_final = sql_text_clean[match.start():]
        else:
            sql_final = sql_text_clean # Podría ser un mensaje de texto
            
        sql_final = " ".join(sql_final.split()).rstrip(";")
        
        # Ajuste SQL Server vs LIMIT
        match_limit = re.search(r"LIMIT\s+(\d+)", sql_final, re.IGNORECASE)
        if match_limit:
            n = match_limit.group(1)
            sql_final = re.sub(r"SELECT", f"SELECT TOP {n}", sql_final, count=1, flags=re.IGNORECASE) 
            sql_final = re.sub(r"LIMIT\s+\d+", "", sql_final, flags=re.IGNORECASE)

        # Si el LLM no generó SQL (suele ser la primera línea de la respuesta)
        # y no contiene palabras clave de SQL, lo tratamos como un mensaje de texto
        if not sql_final.upper().startswith("SELECT"):
             # Si no comienza con SELECT, devolvemos solo el texto sin post-procesar
             return sql_text, docs 
             
        # Reemplazo automático de tablas → Database.Schema.Table
        sql_final = qualify_table_names_improved(sql_final, docs) 

        # Post-procesado inteligente (corrige AS AS y name -> nameFirst)
        sql_final = postprocess_sql(sql_final, docs, pregunta_actual) 

        return sql_final, docs # Devuelve el SQL y los documentos (para posible depuración)

    except Exception as e:
        return f"Error calling the Groq model or during post-processing: {e}", None

# ----------- Ejecutar SQL ----------------

def ejecutar_sql(sql):
    """Crea y retorna la conexión a la DB, y ejecuta el SQL. 
    Retorna (DataFrame, None) o (None, ErrorMessage).
    """
    try:
        # Reutilizamos el motor si es posible, pero creamos uno nuevo por seguridad.
        engine = create_engine(DB_URI) 
        df = pd.read_sql_query(sql, engine)
        return df, None # Éxito
    except Exception as e:
        # Devuelve el mensaje de error de la DB
        return None, f"SQL Server error: {e}"

# ---------- Interfaz Streamlit ----------
st.title("🤖 SQLy Agent")
st.caption(f"Model: {MODEL_NAME} | RAG: Weaviate")

# 1. Mostrar historial de mensajes
for message in st.session_state.messages:
    # Mostramos los mensajes de texto normalmente
    if isinstance(message["content"], str):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    # Si el contenido es un DataFrame (resultado exitoso), lo mostramos como tabla
    elif isinstance(message["content"], pd.DataFrame):
        with st.chat_message("data"):
             st.subheader("✅ SQL Query Result")
             st.dataframe(message["content"])
             # Aquí podrías añadir los botones de descarga si quieres
             
# 2. Capturar nueva entrada del usuario
if prompt := st.chat_input("Write your query in natural language..."):
    # Añadir pregunta del usuario al historial
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Mostrar el mensaje del usuario inmediatamente
    with st.chat_message("user"):
        st.markdown(prompt)

    # 3. Generar respuesta
    with st.chat_message("assistant"):
        with st.spinner("Generating SQL query..."):
            # Llama a la función con el historial completo
            agente_response, docs = generar_respuesta_con_groq(st.session_state.messages)
            
            # 4. Determinar si es SQL o una respuesta de texto
            if agente_response.upper().startswith("SELECT"):
                # --- ES SQL ---
                
                # Almacenar el SQL generado para la ejecución
                st.session_state.current_sql = agente_response
                st.session_state.current_docs = docs
                
                st.markdown("**Generated SQL:**")
                st.code(agente_response, language="sql")
                
                # 5. Intentar ejecutar el SQL
                with st.spinner("Executing SQL query on the database..."):
                    df_resultado, error_msg = ejecutar_sql(agente_response)
                
                if df_resultado is not None:
                    # --- ÉXITO EN LA EJECUCIÓN ---
                    st.success("✅ Query executed successfully.")
                    
                    # Añadir el DataFrame al historial (para que se muestre en el siguiente rerun)
                    st.session_state.messages.append({"role": "data", "content": df_resultado})
                    
                    # Añadir un mensaje de confirmación del asistente
                    st.session_state.messages.append({"role": "assistant", "content": "All set! Here are the results."})
                    st.rerun() # Forzar rerun para mostrar el DataFrame y el mensaje

                else:
                    # --- ERROR EN LA EJECUCIÓN ---
                    error_message_to_llm = f"⚠️ EXECUTION ERROR: The generated SQL query failed with the following database error: {error_msg}. Please correct the query based on this error and your RAG context, and return the **new SQL query only**."
                    
                    st.error("❌ Execution error. The agent will attempt to correct it based on this feedback.")
                    st.code(error_msg, language="text") # Muestra el error de la DB
                    
                    # Añadir el error al historial como mensaje del usuario (para alimentar al LLM)
                    st.session_state.messages.append({"role": "user", "content": error_message_to_llm})
                    
                    # Llamar al reran, la nueva consulta del usuario (el error) será procesada
                    st.rerun() 
                    
            else:
                # --- ES RESPUESTA DE TEXTO (No SQL) ---
                st.markdown(agente_response)
                st.session_state.messages.append({"role": "assistant", "content": agente_response})