# Gerçek Zamanlı Trafik Analiz Uygulaması

Bu uygulama, trafik kamera görüntülerini analiz ederek her şeritteki araç sayısını ve trafik yoğunluğunu tespit eder.

## Özellikler

- Gerçek zamanlı araç tespiti
- Her şerit için ayrı araç sayımı
- Yeşil kutularla araç işaretleme
- Acil durum şeridini göz ardı etme
- Her şerit için doluluk oranı hesaplama

## Kurulum

1. Gerekli paketleri yükleyin:
```bash
pip install -r requirements.txt
```

2. YOLOv8 modelini indirin (ilk çalıştırmada otomatik olarak indirilecektir)

## Kullanım

1. Video kaynağını belirtin:
```python
analyzer = TrafficAnalyzer("video_kaynagi.mp4")
```

2. Uygulamayı çalıştırın:
```bash
python traffic_analyzer.py
```

## Gereksinimler

- Python 3.8+
- OpenCV
- YOLOv8
- Supervision
- NumPy

## Not

Uygulama çıkış yapmak için 'q' tuşuna basın. 