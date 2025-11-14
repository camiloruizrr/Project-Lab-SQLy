# Home.py
import streamlit as st
import streamlit_authenticator as stauth

# Configurar la página (opcional, pero buena práctica)
st.set_page_config(
    page_title="Agente SQL - Login",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- 1. CONFIGURACIÓN DE USUARIOS ---
config_data = {
    'credentials': {
        'usernames': {
            'camilo': {
                'email': 'camiloruiz2576@gmail.com',
                'name': 'Camilo Ruiz',
                # Hash para 'admin123'
                'password': '$2b$12$Auuz.HMO6p9DS2eU7rJBCOepDFbW7UJ9X5TQytWU4efkSHbheeVo6'
            },
            'rbriggs': {
                'email': 'rbriggs@example.com',
                'name': 'Rebecca Briggs',
                # Hash para 'test_password_2'
                'password': '$2b$12$4O.j.M6l.M3.V1x.i.Y7s7w.P8i5m.N.M4g5f5t.X3v.V4r4e8y.w9a'
            }
        }
    },
    'cookie': {
        'expiry_days': 30,
        # Clave actualizada para invalidar cookies viejas
        'key': 'sql_agent_cookie_key_v2', 
        'name': 'sql_agent_cookie'
    },
    'preauthorized': {
        'emails': ['admin@example.com']
    }
}

# --- 2. INICIALIZAR AUTENTICADOR ---
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

# --- 3. LÓGICA DE LOGIN ---

st.title("🤖 SQLy Agent - Login")

# Ejecuta el login.
name, authentication_status, username = authenticator.login(
    form_name='Login', 
    location='main'
)

# -------------------------------------------------------------
# Lógica de Redirección (st.switch_page)
# -------------------------------------------------------------

if authentication_status:
    # --- AUTENTICACIÓN EXITOSA ---
    
    # 1. Almacenar el estado de la sesión
    st.session_state["authentication_status"] = True
    st.session_state["name"] = name
    st.session_state["username"] = username
    
    st.toast("✅ ¡Inicio de sesión exitoso!", icon="🎉")
    
    # 2. REDIRECCIÓN: Mueve al usuario directamente a la aplicación principal
    # RUTA CORREGIDA: Usamos la carpeta, pero SIN la extensión .py
    st.switch_page("pages/App_Principal.py") 
    
elif authentication_status == False:
    # --- CREDENCIALES INCORRECTAS ---
    #st.session_state["authentication_status"] = False
    st.error('❌ Nombre de usuario o contraseña incorrectos.')

elif authentication_status == None:
    # --- NO HA INICIADO SESIÓN O ESPERANDO INPUT ---
    #st.session_state["authentication_status"] = False
    # No mostramos un error si solo está esperando el input del usuario
    st.info('🔒 Por favor, introduce tus credenciales para acceder.')