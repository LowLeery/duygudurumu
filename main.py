import cv2
import numpy as np
import time
from fer import FER
import tensorflow as tf

gpu_var = tf.config.list_physical_devices('GPU')
gpu_kullaniliyor = False
if gpu_var:
    print(">>> GPU kullanılacak:", gpu_var[0].name)
    gpu_kullaniliyor = True
else:
    print(">>> GPU BULUNAMADI, CPU kullanılıyor!")

duygu_cevir = {
    "angry": "Sinirli",
    "disgust": "Igrenmis",
    "fear": "Korkmus",
    "happy": "Mutlu",
    "sad": "Uzgun",
    "surprise": "Sasirmis",
    "neutral": "Notr"
}

kamera = cv2.VideoCapture(0)
detector = FER(mtcnn=True)

cv2.namedWindow("Ruh Hali Tespiti", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Ruh Hali Tespiti", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
tam_ekran = True

analiz_modu = False
analiz_baslangic = 0
analiz_suresi = 15
font = cv2.FONT_HERSHEY_DUPLEX

def buhulu_giris_ekrani(frame):
    blur = cv2.GaussianBlur(frame, (61, 61), 0)
    h, w = frame.shape[:2]
    mesaj = "Duygu Durumunu Analiz Etmek Icin SPACE Tusuna Bas"
    (text_w, text_h), _ = cv2.getTextSize(mesaj, font, 1.2, 3)
    text_x = (w - text_w) // 2
    text_y = (h + text_h) // 2
    cv2.putText(blur, mesaj, (text_x, text_y), font, 1.2, (255, 255, 255), 3, cv2.LINE_AA)
    return blur

def sayac_ciz(frame, kalan):
    merkez = (frame.shape[1] - 80, 80)
    yaricap = 35
    kalinlik = 6
    oran = kalan / analiz_suresi
    bas = -90
    bitis = int(360 * oran) - 90
    cv2.ellipse(frame, merkez, (yaricap, yaricap), 0, bas, bitis, (0, 255, 255), kalinlik)

def kenarlara_renk(frame, renk):
    alpha = 0.35
    overlay = frame.copy()
    kalinlik = 50
    h, w = frame.shape[:2]
    cv2.rectangle(overlay, (0, 0), (w, kalinlik), renk, -1)
    cv2.rectangle(overlay, (0, h - kalinlik), (w, h), renk, -1)
    cv2.rectangle(overlay, (0, 0), (kalinlik, h), renk, -1)
    cv2.rectangle(overlay, (w - kalinlik, 0), (w, h), renk, -1)
    return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

# Sol üstte GPU/CPU yazısı
#def yazdir_islemci_tipi(frame):
    #yazi = "GPU Aktif" if gpu_kullaniliyor else "CPU Modu"
    #renk = (0, 255, 0) if gpu_kullaniliyor else (0, 0, 255)
    #cv2.putText(frame, yazi, (20, 40), font, 1, renk, 2, cv2.LINE_AA)

while True:
    ret, frame = kamera.read()
    if not ret:
        break

    frame = cv2.resize(frame, (1920, 1080))

    if not analiz_modu:
        ekran = buhulu_giris_ekrani(frame)
        yazdir_islemci_tipi(ekran)
        cv2.imshow("Ruh Hali Tespiti", ekran)
    else:
        kalan = analiz_suresi - int(time.time() - analiz_baslangic)
        if kalan <= 0:
            analiz_modu = False
            continue

        results = detector.detect_emotions(frame)

        for result in results:
            (x, y, w, h) = result["box"]
            emotions = result["emotions"]
            if not emotions:
                continue
            emotion = max(emotions, key=emotions.get)
            score = emotions[emotion]

            # Renk belirle
            if emotion == "happy":
                renk = (0, 255, 255)
            elif emotion == "sad":
                renk = (255, 0, 0)
            elif emotion == "angry":
                renk = (0, 0, 255)
            else:
                renk = (0, 255, 0)

            duygu = duygu_cevir.get(emotion, emotion)

            frame = kenarlara_renk(frame, renk)
            cv2.rectangle(frame, (x, y), (x + w, y + h), renk, 2)
            cv2.putText(frame, f"{duygu} ({int(score * 100)}%)", (x, y - 10),
                        font, 0.9, (255, 255, 255), 2)

        sayac_ciz(frame, kalan)
        yazdir_islemci_tipi(frame)
        cv2.imshow("Ruh Hali Tespiti", frame)

    key = cv2.waitKey(1)
    if key == 27:
        tam_ekran = not tam_ekran
        cv2.setWindowProperty("Ruh Hali Tespiti", cv2.WND_PROP_FULLSCREEN,
                              cv2.WINDOW_FULLSCREEN if tam_ekran else cv2.WINDOW_NORMAL)
    elif key == 32 and not analiz_modu:  # SPACE tuşu
        analiz_modu = True
        analiz_baslangic = time.time()

kamera.release()
cv2.destroyAllWindows()
