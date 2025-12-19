import os
import time
import json
from datetime import datetime

import cv2
import face_recognition
import numpy as np
import pyttsx3
import speech_recognition as sr
import warnings

# Suprimir aviso depreciação vindo de face_recognition_models/pkg_resources
warnings.filterwarnings("ignore", category=UserWarning, module="face_recognition_models")

# ===============================
# Configurações
# ===============================
CAMERA_INDEX = 0
TOLERANCE = 0.5
USERS_DIR = "usuarios"
DB_FILE = os.path.join(USERS_DIR, "usuarios.json")
IMAGEM_REFERENCIA = os.path.join(os.getcwd(), "testefoto.jpg")
FRAME_DOWNSCALE = 0.5
PROCESS_EVERY_N_FRAMES = 2

os.makedirs(USERS_DIR, exist_ok=True)

# ===============================
# Voz (fala)
# ===============================
voz = pyttsx3.init()
voz.setProperty("rate", 170)

def falar(texto):
    print("ROBÔ:", texto)
    voz.say(texto)
    voz.runAndWait()

# ===============================
# Voz → Texto
# ===============================
def ouvir(timeout=5):
    r = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            r.adjust_for_ambient_noise(source, duration=0.5)
            audio = r.listen(source, timeout=timeout)
    except Exception:
        return None

    try:
        texto = r.recognize_google(audio, language="pt-BR").lower()
        print("🎤", texto)
        return texto
    except Exception:
        return None

# ===============================
# Usuários (salvar / carregar)
# ===============================
def salvar_usuario(nome, idade, humor, encoding, frame):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    foto_path = os.path.join(USERS_DIR, f"{nome}_{timestamp}.jpg")
    cv2.imwrite(foto_path, frame)

    usuario = {
        "nome": nome,
        "idade": idade,
        "humor": humor,
        "foto": foto_path,
        "encoding": encoding.tolist()
    }

    with open(DB_FILE, "a", encoding="utf-8") as f:
        json.dump(usuario, f, ensure_ascii=False)
        f.write("\n")

    falar(f"{nome}, você foi cadastrado com sucesso.")
    return usuario

def carregar_usuarios():
    dados = []
    encodings = []

    if not os.path.exists(DB_FILE):
        return dados, encodings

    with open(DB_FILE, "r", encoding="utf-8") as f:
        for linha in f:
            if not linha.strip():
                continue
            u = json.loads(linha)
            dados.append(u)
            encodings.append(np.array(u["encoding"]))

    return dados, encodings

# ===============================
# Carregar dados e referência
# ===============================
dados, encodings = carregar_usuarios()

ref_encoding = None
ref_info = None
if os.path.exists(IMAGEM_REFERENCIA):
    try:
        ref_img = face_recognition.load_image_file(IMAGEM_REFERENCIA)
        ref_encs = face_recognition.face_encodings(ref_img)
        if ref_encs:
            ref_encoding = ref_encs[0]
            # Dados fictícios associados à imagem de referência
            ref_info = {"nome": "Luiz", "idade": 22, "humor": "Neutro"}
    except Exception:
        ref_encoding = None

# ===============================
# Inicialização da câmera
# ===============================
cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    falar("Erro ao acessar a câmera.")
    raise RuntimeError("Não foi possível acessar a câmera.")

falar("Sistema iniciado. Reconhecimento facial ativado.")

ultimo_nome_falado = None
ultimo_tempo = 0

try:
    frame_counter = 0
    last_faces_info = []
    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue

        frame_counter += 1

        # Reduzir para processamento mais rápido
        small_frame = cv2.resize(frame, (0, 0), fx=FRAME_DOWNSCALE, fy=FRAME_DOWNSCALE)
        rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        if frame_counter % PROCESS_EVERY_N_FRAMES == 0:
            locations_small = face_recognition.face_locations(rgb_small, model='hog')
            encs_small = face_recognition.face_encodings(rgb_small, locations_small)

            faces = []
            for (top_s, right_s, bottom_s, left_s), face_enc in zip(locations_small, encs_small):
                # Escalar coordenadas para o frame original
                top = int(top_s / FRAME_DOWNSCALE)
                right = int(right_s / FRAME_DOWNSCALE)
                bottom = int(bottom_s / FRAME_DOWNSCALE)
                left = int(left_s / FRAME_DOWNSCALE)

                nome = "Desconhecido"
                fonte = None
                idade = "-"
                humor = "-"

                # Verificar contra DB de usuários
                if encodings:
                    matches = face_recognition.compare_faces(encodings, face_enc, TOLERANCE)
                    if True in matches:
                        idx = matches.index(True)
                        usuario = dados[idx]
                        nome = usuario.get("nome", "Desconhecido")
                        idade = usuario.get("idade", "-")
                        humor = usuario.get("humor", "-")
                        fonte = "db"

                # Se não encontrado no DB, verificar imagem de referência
                if nome == "Desconhecido" and ref_encoding is not None:
                    if face_recognition.compare_faces([ref_encoding], face_enc, TOLERANCE)[0]:
                        nome = ref_info["nome"] if ref_info else "Referência"
                        info = ref_info or {}
                        idade = info.get("idade", "-")
                        humor = info.get("humor", "-")
                        fonte = "ref"

                faces.append({"box": (left, top, right, bottom), "nome": nome, "fonte": fonte, "idade": idade, "humor": humor, "enc": face_enc})

            last_faces_info = faces

        # Desenhar última detecção (ou atual se processado neste frame)
        for face in last_faces_info:
            left, top, right, bottom = face["box"]
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

            nome = face["nome"]
            fonte = face["fonte"]
            idade = face.get("idade", "-")
            humor = face.get("humor", "-")

            if fonte == "db":
                agora = time.time()
                if nome != ultimo_nome_falado or agora - ultimo_tempo > 10:
                    falar(f"Olá, {nome}")
                    ultimo_nome_falado = nome
                    ultimo_tempo = agora

                cv2.putText(frame, f"Nome: {nome}", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, f"Idade: {idade}", (left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(frame, f"Humor: {humor}", (left, bottom + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            elif fonte == "ref":
                cv2.putText(frame, f"Nome: {nome}", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv2.putText(frame, f"Idade: {idade}", (left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                cv2.putText(frame, f"Humor: {humor}", (left, bottom + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            else:
                cv2.putText(frame, "Desconhecido", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                # Perguntar via voz se deseja cadastrar (somente quando processado)
                if frame_counter % PROCESS_EVERY_N_FRAMES == 0:
                    falar("Rosto desconhecido. Deseja se cadastrar?")
                    resposta = ouvir()

                    if resposta and "sim" in resposta:
                        falar("Diga seu nome")
                        nome_novo = ouvir()

                        falar("Diga sua idade")
                        idade_novo = ouvir()

                        falar("Como você está se sentindo")
                        humor_novo = ouvir()

                        if nome_novo and idade_novo and humor_novo:
                            usuario = salvar_usuario(nome_novo, idade_novo, humor_novo, face.get("enc"), frame)
                            dados.append(usuario)
                            encodings.append(face.get("enc"))
                        else:
                            falar("Cadastro cancelado.")

                    elif resposta and "não" in resposta:
                        falar("Tudo bem. Cadastro ignorado.")

                    elif resposta and ("sair" in resposta or "encerrar" in resposta):
                        falar("Encerrando sistema.")
                        raise SystemExit

                    time.sleep(2)

        cv2.imshow("Reconhecimento Facial por Voz", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except KeyboardInterrupt:
    pass
finally:
    cap.release()
    cv2.destroyAllWindows()
