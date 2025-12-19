import cv2
import face_recognition

# ---- Dados fictícios ----
nome = "Luiz"
humor = "Neutro"
idade = 22
aniversario = "15/08/2003"

# ---- Imagem de referência ----
imagem_referencia = "testefoto.jpg"

# Carregar e codificar rosto conhecido
ref_img = face_recognition.load_image_file(imagem_referencia)
ref_encodings = face_recognition.face_encodings(ref_img)

if not ref_encodings:
    raise RuntimeError("Nenhum rosto encontrado na imagem de referência.")

ref_encoding = ref_encodings[0]

# ---- Inicializar câmera ----
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Não foi possível acessar a câmera.")

print("Câmera iniciada. Pressione 'q' para sair.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # OpenCV usa BGR → converter para RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 1️⃣ Localizar rostos
    face_locations = face_recognition.face_locations(rgb_frame)

    # 2️⃣ Gerar embeddings USANDO as localizações
    face_encodings = face_recognition.face_encodings(
        rgb_frame, face_locations
    )

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        match = face_recognition.compare_faces(
            [ref_encoding], face_encoding, tolerance=0.5
        )[0]

        # Desenhar retângulo
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        if match:
            cv2.putText(frame, f"Nome: {nome}", (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"Idade: {idade}", (left, bottom + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(frame, f"Humor: {humor}", (left, bottom + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        else:
            cv2.putText(frame, "Desconhecido", (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow("Reconhecimento Facial", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
