import os  # operações de sistema (manipular arquivos e caminhos)
import time  # utilitários de tempo (sleep, time)
import json  # leitura/escrita de JSON
from datetime import datetime  # trabalhar com data e hora

import cv2  # OpenCV para captura e desenho no frame
import face_recognition  # detecção e reconhecimento facial
import numpy as np  # arrays e operações numéricas (encodings)
import pyttsx3  # texto-para-fala (TTS)
import speech_recognition as sr  # reconhecimento de fala (STT)
import warnings  # manipular avisos do Python

# Suprimir aviso depreciação vindo de face_recognition_models/pkg_resources
warnings.filterwarnings("ignore", category=UserWarning, module="face_recognition_models")  # suprime aviso conhecido da lib

# ===============================
# Configurações
# ===============================
CAMERA_INDEX = 0  # índice da câmera (0 = primeira câmera do sistema)
TOLERANCE = 0.5  # distância máxima entre encodings para considerar igual (menor = mais estrito)
USERS_DIR = "usuarios"  # diretório onde serão salvas fotos e DB
DB_FILE = os.path.join(USERS_DIR, "usuarios.json")  # arquivo simples de registros (uma linha por JSON)
IMAGEM_REFERENCIA = os.path.join(os.getcwd(), "testefoto.jpg")  # imagem de referência opcional
FRAME_DOWNSCALE = 0.5  # reduz resolução para acelerar processamento
PROCESS_EVERY_N_FRAMES = 50  # processa a cada N frames para economia de CPU

os.makedirs(USERS_DIR, exist_ok=True)  # cria diretório de usuários caso não exista

# ===============================
# Voz (fala)
# ===============================
voz = pyttsx3.init()  # inicializa o motor TTS
voz.setProperty("rate", 100)  # ajusta a velocidade da fala

def falar(texto):
    print("ROBÔ:", texto)  # imprime mensagem no console
    voz.say(texto)  # enfileira texto para falar
    voz.runAndWait()  # executa a fala (bloqueante)

# ===============================
# Voz → Texto
# ===============================
def ouvir(timeout=5):
    r = sr.Recognizer()  # cria um reconhecedor de fala
    try:
        with sr.Microphone() as source:  # usa microfone como fonte de áudio
            r.adjust_for_ambient_noise(source, duration=0.5)  # ajusta para o ruído ambiente
            audio = r.listen(source, timeout=timeout)  # captura áudio com timeout
    except Exception:
        return None  # falha ao abrir microfone ou timeout

    try:
        texto = r.recognize_google(audio, language="pt-BR").lower()  # usa API Google para STT e normaliza
        print("🎤", texto)  # exibe texto reconhecido
        return texto  # retorna string reconhecida
    except Exception:
        return None  # erro no reconhecimento

# ===============================
# Usuários (salvar / carregar)
# ===============================
def salvar_usuario(nome, idade, humor, encoding, frame):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # cria timestamp para nome único
    foto_path = os.path.join(USERS_DIR, f"{nome}_{timestamp}.jpg")  # caminho onde foto será salva
    cv2.imwrite(foto_path, frame)  # grava a imagem no disco

    usuario = {
        "nome": nome,  # nome capturado por voz
        "idade": idade,  # idade capturada por voz
        "humor": humor,  # humor capturado por voz
        "foto": foto_path,  # caminho da foto salva
        "encoding": encoding.tolist()  # converte numpy array para lista serializável
    }

    with open(DB_FILE, "a", encoding="utf-8") as f:  # abre arquivo em modo append
        json.dump(usuario, f, ensure_ascii=False)  # escreve JSON
        f.write("\n")  # separa registros por linha

    falar(f"{nome}, você foi cadastrado com sucesso.")  # confirma por voz
    return usuario  # retorna o registro salvo

def carregar_usuarios():
    dados = []  # lista de registros de usuários
    encodings = []  # lista de encodings (numpy arrays)

    if not os.path.exists(DB_FILE):  # se não existir, retorna listas vazias
        return dados, encodings

    with open(DB_FILE, "r", encoding="utf-8") as f:  # lê cada linha como um JSON
        for linha in f:
            if not linha.strip():  # ignora linhas vazias
                continue
            u = json.loads(linha)  # desserializa JSON
            dados.append(u)  # armazena metadados
            encodings.append(np.array(u["encoding"]))  # converte lista para numpy array

    return dados, encodings  # retorna dados e encodings carregados

# ===============================
# Carregar dados e referência
# ===============================
dados, encodings = carregar_usuarios()  # carrega usuários existentes e seus encodings

ref_encoding = None  # encoding da imagem de referência (se existir)
ref_info = None  # metadados da referência
if os.path.exists(IMAGEM_REFERENCIA):  # tenta carregar encoding da imagem de referência
    try:
        ref_img = face_recognition.load_image_file(IMAGEM_REFERENCIA)  # carrega arquivo de imagem
        ref_encs = face_recognition.face_encodings(ref_img)  # obtém encodings
        if ref_encs:
            ref_encoding = ref_encs[0]  # usa o primeiro encoding encontrado
            # Dados fictícios associados à imagem de referência
            ref_info = {"nome": "Luiz", "idade": 22, "humor": "Neutro"}
    except Exception:
        ref_encoding = None  # ignora referência se falhar

# ===============================
# Inicialização da câmera
# ===============================
cap = cv2.VideoCapture(CAMERA_INDEX)  # inicia captura de vídeo da câmera selecionada
if not cap.isOpened():  # verifica se a câmera abriu corretamente
    falar("Erro ao acessar a câmera.")  # avisa por voz
    raise RuntimeError("Não foi possível acessar a câmera.")  # aborta execução

falar("Sistema iniciado. Reconhecimento facial ativado.")  # anuncia que o sistema iniciou

ultimo_nome_falado = None  # último nome anunciado (para não repetir)
ultimo_tempo = 0  # timestamp da última fala

try:
    frame_counter = 0  # contador de frames
    last_faces_info = []  # guarda última lista de faces detectadas
    while True:  # loop principal
        ret, frame = cap.read()  # captura um frame da câmera
        if not ret:  # se não conseguiu ler, espera e tenta novamente
            time.sleep(0.01)
            continue

        frame_counter += 1  # incrementa contador

        # Reduzir para processamento mais rápido
        small_frame = cv2.resize(frame, (0, 0), fx=FRAME_DOWNSCALE, fy=FRAME_DOWNSCALE)  # redimensiona
        rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)  # converte para RGB

        if frame_counter % PROCESS_EVERY_N_FRAMES == 0:  # processa a cada N frames
            locations_small = face_recognition.face_locations(rgb_small, model='hog')  # detecta locais de faces
            encs_small = face_recognition.face_encodings(rgb_small, locations_small)  # calcula encodings

            faces = []  # lista temporária de faces
            for (top_s, right_s, bottom_s, left_s), face_enc in zip(locations_small, encs_small):
                # Escalar coordenadas para o frame original
                top = int(top_s / FRAME_DOWNSCALE)
                right = int(right_s / FRAME_DOWNSCALE)
                bottom = int(bottom_s / FRAME_DOWNSCALE)
                left = int(left_s / FRAME_DOWNSCALE)

                nome = "Desconhecido"  # padrão
                fonte = None  # fonte da informação (db/ref)
                idade = "-"  # idade desconhecida
                humor = "-"  # humor desconhecido

                # Verificar contra DB de usuários
                if encodings:
                    matches = face_recognition.compare_faces(encodings, face_enc, TOLERANCE)  # compara face com DB
                    if True in matches:  # se encontrar correspondência
                        idx = matches.index(True)  # pega índice da primeira correspondência
                        usuario = dados[idx]  # recupera dados do usuário
                        nome = usuario.get("nome", "Desconhecido")
                        idade = usuario.get("idade", "-")
                        humor = usuario.get("humor", "-")
                        fonte = "db"  # marcado como vindo do DB

                # Se não encontrado no DB, verificar imagem de referência
                if nome == "Desconhecido" and ref_encoding is not None:
                    if face_recognition.compare_faces([ref_encoding], face_enc, TOLERANCE)[0]:  # compara com referência
                        nome = ref_info["nome"] if ref_info else "Referência"
                        info = ref_info or {}
                        idade = info.get("idade", "-")
                        humor = info.get("humor", "-")
                        fonte = "ref"  # marcado como vindo da referência

                faces.append({"box": (left, top, right, bottom), "nome": nome, "fonte": fonte, "idade": idade, "humor": humor, "enc": face_enc})

            last_faces_info = faces  # atualiza lista de faces

        # Desenhar última detecção (ou atual se processado neste frame)
        for face in last_faces_info:
            left, top, right, bottom = face["box"]  # coordenadas do rosto
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)  # desenha retângulo

            nome = face["nome"]  # nome identificado
            fonte = face["fonte"]  # origem da identificação
            idade = face.get("idade", "-")  # idade (se houver)
            humor = face.get("humor", "-")  # humor (se houver)

            if fonte == "db":  # se identificado via DB
                agora = time.time()  # tempo atual
                if nome != ultimo_nome_falado or agora - ultimo_tempo > 10:  # evita repetir nome com frequência
                    falar(f"Olá, {nome}")  # diz olá por voz
                    ultimo_nome_falado = nome  # atualiza último nome falado
                    ultimo_tempo = agora  # atualiza tempo

                cv2.putText(frame, f"Nome: {nome}", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)  # escreve nome na imagem
                cv2.putText(frame, f"Idade: {idade}", (left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)  # escreve idade
                cv2.putText(frame, f"Humor: {humor}", (left, bottom + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)  # escreve humor
            elif fonte == "ref":  # se identificado pela referência
                cv2.putText(frame, f"Nome: {nome}", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv2.putText(frame, f"Idade: {idade}", (left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                cv2.putText(frame, f"Humor: {humor}", (left, bottom + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            else:  # rosto desconhecido
                cv2.putText(frame, "Desconhecido", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                # Perguntar via voz se deseja cadastrar (somente quando processado)
                if frame_counter % PROCESS_EVERY_N_FRAMES == 0:
                    falar("Rosto desconhecido. Deseja se cadastrar?")  # pergunta por voz
                    resposta = ouvir()  # espera resposta do usuário

                    if resposta and "sim" in resposta:  # se usuário disser sim
                        falar("Diga seu nome")  # pede nome
                        nome_novo = ouvir()  # lê nome

                        falar("Diga sua idade")  # pede idade
                        idade_novo = ouvir()  # lê idade

                        falar("Como você está se sentindo")  # pede humor
                        humor_novo = ouvir()  # lê humor

                        if nome_novo and idade_novo and humor_novo:  # valida campos
                            usuario = salvar_usuario(nome_novo, idade_novo, humor_novo, face.get("enc"), frame)  # salva usuário
                            dados.append(usuario)  # adiciona em memória
                            encodings.append(face.get("enc"))  # atualiza encodings em memória
                        else:
                            falar("Cadastro cancelado.")  # cancela se faltar informação

                    elif resposta and "não" in resposta:  # se usuário disser não
                        falar("Tudo bem. Cadastro ignorado.")  # informa que não irá cadastrar

                    elif resposta and ("sair" in resposta or "encerrar" in resposta):  # se disser sair
                        falar("Encerrando sistema.")  # avisa que vai encerrar
                        raise SystemExit  # termina o programa

                    time.sleep(2)  # pause breve

        cv2.imshow("Reconhecimento Facial por Voz", frame)  # exibe janela com o frame anotado

        if cv2.waitKey(1) & 0xFF == ord('q'):  # se 'q' for pressionado, sai
            break
except KeyboardInterrupt:
    pass  # Ctrl+C interrompe o loop sem traceback
finally:
    cap.release()  # libera dispositivo de captura
    cv2.destroyAllWindows()  # fecha janelas abertas
    falar("Sistema encerrado. Até logo!")  # avisa que o sistema foi encerrado