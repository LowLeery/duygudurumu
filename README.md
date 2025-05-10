# 🎭 Ruh Hali Tespit Sistemi

Bu Python projesi, kameradan alınan görüntüler üzerinden **yüz ifadelerini analiz ederek** kişinin anlık **duygu durumunu tespit eder**. `FER (Facial Expression Recognition)` kütüphanesi ile çalışır ve hem CPU hem de GPU ile kullanılabilir.

## 🖼️ Özellikler

- Anlık kamera görüntüsü ile yüz ifadelerini tanır.
- Duyguları Türkçe olarak ekrana yansıtır.
- 15 saniyelik analiz süresi başlatmak için `SPACE` tuşuna basılır.
- Tespit edilen duyguya göre kenarlık renkleri değişir.
- GPU varsa otomatik olarak kullanır.
- Tam ekran ve normal pencere arasında geçiş için `ESC`.

## 📸 Tespit Edilen Duygular

- 😠 Sinirli
- 🤢 İğrenmiş
- 😱 Korkmuş
- 😄 Mutlu
- 😢 Üzgün
- 😲 Şaşırmış
- 😐 Nötr

## 🚀 Kurulum

<details> <summary>📦 Gereksinimler (requirements.txt)</summary>

```bash
git clone https://github.com/LowLeery/duygudurumu
cd duygudurumu
py -3.10 -m pip install opencv-python fer tensorflow-gpu numpy
python main.py
```
|3.10.x|EN GUNCEL PYTHON VERSIYONU|
|----------|------------|
|  ✔️ |            ❌            |

</details>
