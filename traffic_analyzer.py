import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
from collections import defaultdict
import time
import torch
from ultralytics.nn.tasks import DetectionModel
from sklearn.cluster import DBSCAN
import json
from datetime import datetime

class TrafficAnalyzer:
    def __init__(self, source_path):
        self.source = source_path
        
        # GPU kullanımını kontrol et
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Cihaz kullanımı: {self.device}")
        
        # Nesne tespiti için YOLO modeli
        self.detector = YOLO("yolov8n.pt", task='detect').to(self.device)
        
        # Object tracker
        self.tracker = sv.ByteTrack()
        
        # Video boyutlarını al
        cap = cv2.VideoCapture(self.source)
        self.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(cap.get(cv2.CAP_PROP_FPS))
        self.frame_time = 1000 / self.fps  # Her kare için gereken süre (ms)
        cap.release()
        
        # UI Renkleri - Modern Web Tasarımı
        self.COLORS = {
            'background': (18, 18, 18),      # Koyu arka plan (#121212)
            'surface': (30, 30, 30),         # Panel arka planı (#1E1E1E)
            'primary': (66, 165, 245),       # Ana renk (#42A5F5)
            'high_traffic': (239, 83, 80),   # Yoğun trafik (#EF5350)
            'mid_traffic': (255, 167, 38),   # Orta trafik (#FFA726)
            'low_traffic': (102, 187, 106),  # Düşük trafik (#66BB6A)
            'text': (255, 255, 255),         # Ana metin (#FFFFFF)
            'text_secondary': (158, 158, 158) # İkincil metin (#9E9E9E)
        }
        
        # Font ayarları - Modern görünüm ve Türkçe karakter desteği
        self.FONT = cv2.FONT_HERSHEY_COMPLEX
        self.FONT_SCALES = {
            'title': 1.0,      # Başlık (küçült)
            'subtitle': 0.8,   # Alt başlık
            'body': 0.6,       # Ana metin
            'caption': 0.5     # Küçük metin
        }
        
        # Panel boyutları
        self.side_panel_width = 400  # Genişlet (350 -> 400)
        self.canvas_width = self.frame_width + self.side_panel_width
        self.canvas_height = self.frame_height
        
        # Kalibrasyon için parametreler
        self.calibration_frames = self.fps * 3  # 3 saniyelik kalibrasyon
        self.current_frame = 0
        self.is_calibrated = False
        
        # Şerit tespiti için parametreler
        self.lane_detection_line_y = int(self.frame_height * 0.3)  # %40'tan %30'a çekildi
        self.counting_line_y = int(self.frame_height * 0.7)
        
        # ROI parametreleri
        self.roi_start_y = int(self.frame_height * 0.2)  # Daha uzaktan başla
        self.roi_end_y = self.frame_height
        
        # Şerit bilgileri
        self.lanes = {}  # {lane_id: {"center_x": x, "direction": angle, "count": 0}}
        self.lane_width = 100
        self.min_lanes = 2  # Minimum şerit sayısı (emniyet şeridi hariç)
        self.max_lanes = 4  # Maximum şerit sayısı (emniyet şeridi hariç)
        
        # Araç takibi için parametreler
        self.vehicle_states = {}
        self.min_track_points = 5
        self.direction_threshold = 15
        
        # Şerit yeniden numaralandırma için
        self.temp_lanes = defaultdict(list)  # Geçici şerit pozisyonları
        
        # İstatistikler
        self.last_minute_counts = defaultdict(list)
        self.last_minute_timestamp = time.time()
        
        # Performans optimizasyonu için buffer
        self.frame_buffer = None
        self.skip_frames = 2

    def draw_card(self, panel, title, content_func, x, y, width, height):
        """Modern kart çiz"""
        # Kart arka planı
        cv2.rectangle(panel, 
                     (x, y), 
                     (x + width, y + height),
                     self.COLORS['card'], 
                     -1)
        
        # Kart kenarlığı
        cv2.rectangle(panel, 
                     (x, y), 
                     (x + width, y + height),
                     self.COLORS['border'], 
                     1)
        
        # Başlık arka planı
        cv2.rectangle(panel,
                     (x, y),
                     (x + width, y + 30),
                     self.COLORS['header'],
                     -1)
        
        # Başlık metni
        cv2.putText(panel,
                   title,
                   (x + 10, y + 22),
                   self.FONTS['header'],
                   self.FONT_SCALES['normal'],
                   self.COLORS['text'],
                   1)
        
        # İçerik
        content_func(panel, x + 10, y + 40)
        
    def create_stats_panel(self):
        """Modern istatistik paneli"""
        # Ana panel - Koyu arka plan
        panel = np.full((self.frame_height, self.side_panel_width, 3), 
                       self.COLORS['background'], 
                       dtype=np.uint8)
        
        # Üst panel arka planı - Gradient efekti
        gradient_height = 60  # Küçült (80 -> 60)
        for y in range(gradient_height):
            alpha = 1 - (y / gradient_height)
            color = tuple(int(c * alpha) for c in self.COLORS['primary'])
            cv2.line(panel, (0, y), (self.side_panel_width, y), color, 1)
        
        # Başlık - Modern font ve kontrast
        title_text = "TRAFIK ANALIZI"  # Türkçe karakter kullanma
        title_size = cv2.getTextSize(title_text, self.FONT, self.FONT_SCALES['title'], 2)[0]
        title_x = (self.side_panel_width - title_size[0]) // 2
        
        # Başlık gölgesi
        cv2.putText(panel,
                   title_text,
                   (title_x + 2, 42),
                   self.FONT,
                   self.FONT_SCALES['title'],
                   (0, 0, 0),
                   2)
        
        # Başlık metni
        cv2.putText(panel,
                   title_text,
                   (title_x, 40),
                   self.FONT,
                   self.FONT_SCALES['title'],
                   self.COLORS['text'],
                   2)
        
        y_offset = 80  # Yukarı kaydır (100 -> 80)
        card_height = 60  # Küçült (70 -> 60)
        
        # Normal şeritleri sırala ve yoğunluğa göre renklendir
        sorted_lanes = sorted(self.lanes.items(), key=lambda x: float(x[0]))
        
        # Şerit yoğunluklarını hesapla
        lane_counts = [(lane_id, lane_info['count']) for lane_id, lane_info in sorted_lanes]
        if lane_counts:
            max_count = max(count for _, count in lane_counts)
            min_count = min(count for _, count in lane_counts)
            
            for lane_id, lane_info in sorted_lanes:
                count = lane_info['count']
                
                # Yoğunluğa göre renk seç
                if count == max_count:
                    color = self.COLORS['high_traffic']
                elif count == min_count:
                    color = self.COLORS['low_traffic']
                else:
                    color = self.COLORS['mid_traffic']
                
                # Şerit kartı arka planı
                cv2.rectangle(panel,
                            (20, y_offset - 10),
                            (self.side_panel_width - 20, y_offset + card_height),
                            (30, 30, 30),
                            -1)
                
                # Şerit başlığı
                lane_title = "Serit {}".format(lane_id)  # Türkçe karakter kullanma
                cv2.putText(panel,
                          lane_title,
                          (35, y_offset + 20),
                          self.FONT,
                          self.FONT_SCALES['subtitle'],
                          color,
                          2)
                
                # Son 1 dakika sayısı
                last_minute_count = len([t for t in self.last_minute_counts.get(lane_id, [])
                                       if time.time() - t <= 60])
                
                # İstatistik metni
                stats_text = "Toplam: {}".format(count)
                minute_text = "Son 1dk: {}".format(last_minute_count)
                
                cv2.putText(panel,
                          stats_text,
                          (35, y_offset + 45),
                          self.FONT,
                          self.FONT_SCALES['body'],
                          self.COLORS['text'],
                          1)
                
                cv2.putText(panel,
                          minute_text,
                          (180, y_offset + 45),
                          self.FONT,
                          self.FONT_SCALES['body'],
                          self.COLORS['text_secondary'],
                          1)
                
                y_offset += card_height + 10
        
        # Alt bilgi kartı
        footer_height = 50  # Küçült (60 -> 50)
        cv2.rectangle(panel,
                     (20, self.frame_height - footer_height - 20),  # En alta sabitle
                     (self.side_panel_width - 20, self.frame_height - 20),
                     (30, 30, 30),
                     -1)
        
        # Toplam araç sayısı
        total_vehicles = sum(lane_info['count'] for lane_info in self.lanes.values())
        total_text = "Toplam Arac: {}".format(total_vehicles)  # Türkçe karakter kullanma
        
        cv2.putText(panel,
                   total_text,
                   (35, self.frame_height - footer_height + 5),
                   self.FONT,
                   self.FONT_SCALES['subtitle'],
                   self.COLORS['primary'],
                   2)
        
        return panel

    def analyze_traffic(self):
        """Ana analiz döngüsü"""
        cap = cv2.VideoCapture(self.source)
        
        # FPS kontrolü için değişkenler
        prev_time = 0
        
        while cap.isOpened():
            # FPS kontrolü
            time_elapsed = time.time() - prev_time
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Kareyi işle
            processed_frame = self.process_frame(frame)
            
            # İstatistik panelini oluştur
            stats_panel = self.create_stats_panel()
            
            # Görüntüleri birleştir
            combined_frame = np.zeros((self.canvas_height, self.canvas_width, 3), 
                                    dtype=np.uint8)
            combined_frame[:, :self.frame_width] = processed_frame
            combined_frame[:, self.frame_width:] = stats_panel
            
            # Sonucu göster
            cv2.imshow("Trafik Analizi", combined_frame)
            
            # FPS kontrolü - minimum bekleme süresi
            wait_time = max(1, int(self.frame_time - time_elapsed * 1000))
            if cv2.waitKey(wait_time) & 0xFF == ord('q'):
                break
            
            prev_time = time.time()
        
        cap.release()
        cv2.destroyAllWindows()

    def detect_road(self, frame):
        """YOLO-seg ile yol alanını tespit et"""
        results = self.segmentor(frame, classes=[0])  # 0: road class
        if len(results) > 0 and hasattr(results[0], 'masks') and results[0].masks is not None:
            mask = results[0].masks.data[0].cpu().numpy()
            return mask.astype(np.uint8) * 255
        return np.ones((self.frame_height, self.frame_width), dtype=np.uint8) * 255

    def calculate_direction(self, positions):
        """Araç hareket yönünü hesapla"""
        if len(positions) < 2:
            return None
            
        # Son 5 noktayı kullan
        positions = positions[-5:] if len(positions) > 5 else positions
        
        # İlk ve son nokta arasındaki açıyı hesapla
        start = positions[0]
        end = positions[-1]
        
        # Yatay eksene göre açıyı hesapla (derece cinsinden)
        angle = np.arctan2(end[1] - start[1], end[0] - start[0]) * 180 / np.pi
        return angle

    def assign_lane(self, x_pos, direction):
        """Araca şerit ata"""
        if direction is None:
            return None
            
        # Kalibrasyon sırasında şerit pozisyonlarını topla
        if not self.is_calibrated:
            self.temp_lanes[int(x_pos // self.lane_width)].append(x_pos)
            return None
            
        # Mevcut şeritleri kontrol et ve en yakın iki şeridi bul
        closest_lanes = []
        for lane_id, lane_info in self.lanes.items():
            distance = abs(lane_info["center_x"] - x_pos)
            closest_lanes.append((lane_id, distance))
        
        # Mesafeye göre sırala
        closest_lanes.sort(key=lambda x: x[1])
        
        # Eğer tek şerit yakınsa, direkt ata
        if len(closest_lanes) == 1 or (
            len(closest_lanes) >= 2 and 
            closest_lanes[1][1] - closest_lanes[0][1] > self.lane_width * 0.5
        ):
            return closest_lanes[0][0]
        
        # İki şerit arasında çekişme varsa
        if len(closest_lanes) >= 2 and closest_lanes[1][1] - closest_lanes[0][1] <= self.lane_width * 0.5:
            lane1_id, lane1_dist = closest_lanes[0]
            lane2_id, lane2_dist = closest_lanes[1]
            
            # Şerit numaralarını integer'a çevir
            lane1_num = int(lane1_id)
            lane2_num = int(lane2_id)
            
            # Ardışık olmayan şerit numaraları varsa
            if abs(lane1_num - lane2_num) > 1:
                # Diğer şeritlere olan mesafeleri kontrol et
                other_lanes = [(lid, info["center_x"]) for lid, info in self.lanes.items() 
                             if lid not in [lane1_id, lane2_id]]
                
                if other_lanes:
                    # En yakın diğer şeride olan mesafeleri karşılaştır
                    for other_id, other_x in other_lanes:
                        other_num = int(other_id)
                        
                        # Eğer diğer şerit, çekişen şeritlerden birinin yanındaysa
                        if abs(other_num - lane1_num) == 1 or abs(other_num - lane2_num) == 1:
                            # Mesafeleri karşılaştır
                            dist_to_other = abs(x_pos - other_x)
                            
                            # Eğer diğer şeride daha yakınsa, ona göre karar ver
                            if dist_to_other < max(lane1_dist, lane2_dist):
                                return lane1_id if abs(other_num - lane1_num) == 1 else lane2_id
            
            # Varsayılan olarak en yakın şeridi döndür
            return lane1_id
        
        return None

    def calibrate_lanes(self):
        """Şeritleri kalibre et"""
        if len(self.temp_lanes) < self.min_lanes:
            return False
        
        # Şerit merkezlerini hesapla
        lane_centers = []
        for positions in self.temp_lanes.values():
            if len(positions) >= 2:  # En az 2 gözlem
                std_dev = np.std(positions)
                if std_dev < self.lane_width * 0.3:  # Pozisyon tutarlılığı kontrolü
                    avg_pos = sum(positions) / len(positions)
                    lane_centers.append(avg_pos)
        
        if len(lane_centers) < self.min_lanes:
            return False
        
        # Şeritleri soldan sağa sırala
        lane_centers.sort()
        
        # Minimum şerit genişliği kontrolü
        min_lane_width = self.lane_width * 0.7  # %70 tolerans
        valid_centers = [lane_centers[0]]
        
        for i in range(1, len(lane_centers)):
            if lane_centers[i] - valid_centers[-1] >= min_lane_width:
                valid_centers.append(lane_centers[i])
        
        # Şerit sayısı kontrolü (emniyet şeridi hariç)
        normal_lanes = len(valid_centers)
        if normal_lanes > self.max_lanes:
            # Fazla şeritleri ele
            valid_centers = valid_centers[:self.max_lanes]
        elif normal_lanes < self.min_lanes:
            return False
        
        # Şeritleri yeniden numaralandır
        self.lanes.clear()
        
        # Önce emniyet şeridini kontrol et
        if valid_centers[0] < self.frame_width * 0.15:  # Sol kenardan %15 içeride
            self.lanes["0"] = {
                "center_x": valid_centers[0],
                "count": 0,
                "type": "emergency"
            }
            valid_centers = valid_centers[1:]  # İlk şeridi emniyet şeridi yap
        
        # Normal şeritleri numaralandır
        for i, center_x in enumerate(valid_centers, 1):
            self.lanes[str(i)] = {
                "center_x": center_x,
                "count": 0,
                "type": "normal"
            }
        
        # Şerit numaralarında tutarsızlık kontrolü
        lane_ids = sorted([int(lid) for lid in self.lanes.keys()])
        for i in range(len(lane_ids)-1):
            if lane_ids[i+1] - lane_ids[i] > 1:  # Ardışık olmayan şeritler
                # Şeritleri yeniden kontrol et
                problematic_lanes = []
                for lid in self.lanes:
                    if int(lid) in [lane_ids[i], lane_ids[i+1]]:
                        problematic_lanes.append((lid, self.lanes[lid]["center_x"]))
                
                # Diğer şeritlerle karşılaştır
                other_lanes = [(lid, info["center_x"]) for lid, info in self.lanes.items() 
                             if int(lid) not in [lane_ids[i], lane_ids[i+1]]]
                
                if other_lanes:
                    # En yakın diğer şeridi bul
                    for prob_id, prob_x in problematic_lanes:
                        min_dist = float('inf')
                        closest_id = None
                        for other_id, other_x in other_lanes:
                            dist = abs(prob_x - other_x)
                            if dist < min_dist:
                                min_dist = dist
                                closest_id = other_id
                        
                        # Şerit numarasını düzelt
                        if closest_id:
                            new_id = str(int(closest_id) + 1)
                            self.lanes[new_id] = self.lanes.pop(prob_id)
        
        print(f"{len(self.lanes)} şerit tespit edildi ({sum(1 for l in self.lanes.values() if l['type']=='normal')} normal, {sum(1 for l in self.lanes.values() if l['type']=='emergency')} emniyet şeridi)")
        return True

    def process_frame(self, frame):
        """Her kareyi işle"""
        self.current_frame += 1
        current_time = time.time()
        
        # Frame buffer kontrolü
        if self.frame_buffer is not None and self.current_frame % self.skip_frames != 0:
            return self.frame_buffer
        
        # Kalibrasyon durumunu kontrol et
        if not self.is_calibrated:
            if self.current_frame >= self.calibration_frames:
                self.is_calibrated = self.calibrate_lanes()
                if not self.is_calibrated:
                    self.current_frame = 0
                    self.temp_lanes.clear()
                    print("Kalibrasyon basarisiz, yeniden baslatiliyor...")
                else:
                    print("Kalibrasyon tamamlandi!")
            else:
                # Kalibrasyon sırasında çizgileri göster
                cv2.line(frame, (0, self.lane_detection_line_y),
                        (frame.shape[1], self.lane_detection_line_y),
                        self.COLORS['primary'], 2)
                
                cv2.putText(frame,
                          "Kalibrasyon: {}/{}".format(self.current_frame, self.calibration_frames),
                          (20, 50),
                          self.FONT,
                          self.FONT_SCALES['subtitle'],
                          self.COLORS['text'],
                          2)
        
        # Tespit çizgisini kesikli çiz
        line_length = 30  # Çizgi uzunluğu
        gap_length = 20   # Boşluk uzunluğu
        x = 0
        while x < frame.shape[1]:
            start_x = x
            end_x = min(x + line_length, frame.shape[1])
            cv2.line(frame, 
                    (start_x, self.lane_detection_line_y),
                    (end_x, self.lane_detection_line_y),
                    self.COLORS['primary'], 1)
            x = end_x + gap_length
        
        # Sayma çizgisini çiz
        cv2.line(frame, (0, self.counting_line_y),
                (frame.shape[1], self.counting_line_y),
                self.COLORS['high_traffic'], 2)
        
        cv2.putText(frame,
                  "Sayma Cizgisi",
                  (10, self.counting_line_y - 10),
                  self.FONT,
                  self.FONT_SCALES['caption'],
                  self.COLORS['high_traffic'],
                  1)
        
        # ROI bölgesini kırp
        roi = frame[self.roi_start_y:self.roi_end_y, :]
        
        # Araç tespiti
        detections = self.detector(roi, classes=[2, 3, 5, 7], imgsz=320)[0]
        boxes = detections.boxes.cpu().numpy()
        
        if len(boxes.xyxy) > 0:
            boxes.xyxy[:, [1, 3]] += self.roi_start_y
        
        # Tracker'ı güncelle
        detections = sv.Detections(
            xyxy=boxes.xyxy,
            confidence=boxes.conf,
            class_id=np.zeros_like(boxes.cls.astype(int)),
            tracker_id=None
        )
        
        if len(detections) > 0:
            detections = self.tracker.update_with_detections(detections)
            
            # Her aracı işle
            for i in range(len(detections)):
                xyxy = detections.xyxy[i]
                tracker_id = detections.tracker_id[i]
                
                # Araç merkezi
                center_x = int((xyxy[0] + xyxy[2]) / 2)
                center_y = int((xyxy[1] + xyxy[3]) / 2)
                current_pos = (center_x, center_y)
                
                # Araç geçmişini güncelle
                if tracker_id not in self.vehicle_states:
                    self.vehicle_states[tracker_id] = {
                        "positions": [current_pos],
                        "lane_id": None,
                        "counted": False
                    }
                else:
                    self.vehicle_states[tracker_id]["positions"].append(current_pos)
                
                # Şerit tespiti
                if (center_y > self.lane_detection_line_y and
                    len(self.vehicle_states[tracker_id]["positions"]) >= self.min_track_points):
                    
                    direction = self.calculate_direction(
                        self.vehicle_states[tracker_id]["positions"]
                    )
                    
                    lane_id = self.assign_lane(center_x, direction)
                    if lane_id and not self.vehicle_states[tracker_id]["lane_id"]:
                        self.vehicle_states[tracker_id]["lane_id"] = lane_id
                
                # Araç sayımı
                if (self.is_calibrated and 
                    self.vehicle_states[tracker_id]["lane_id"] and 
                    not self.vehicle_states[tracker_id]["counted"] and 
                    center_y > self.counting_line_y):
                    
                    lane_id = self.vehicle_states[tracker_id]["lane_id"]
                    self.lanes[lane_id]["count"] += 1
                    self.vehicle_states[tracker_id]["counted"] = True
                    
                    # Son 1 dakika sayacını güncelle
                    self.last_minute_counts[lane_id].append(current_time)
                    self.last_minute_counts[lane_id] = [t for t in self.last_minute_counts[lane_id]
                                                      if current_time - t <= 60]
                
                # Aracı çiz
                cv2.rectangle(frame, 
                            (int(xyxy[0]), int(xyxy[1])),
                            (int(xyxy[2]), int(xyxy[3])), 
                            self.COLORS['primary'], 2)
                
                # Şerit numarasını yaz
                if self.is_calibrated and self.vehicle_states[tracker_id]["lane_id"]:
                    lane_text = "Lane {}".format(self.vehicle_states[tracker_id]["lane_id"])
                    cv2.putText(frame,
                              lane_text,
                              (int(xyxy[0]), int(xyxy[1]) - 5),
                              self.FONT,
                              self.FONT_SCALES['caption'],
                              (0, 0, 255),  # Kırmızı renk
                              1)
                
                # Hareket vektörünü çiz
                positions = self.vehicle_states[tracker_id]["positions"]
                if len(positions) >= 2:
                    cv2.line(frame,
                            positions[-2],
                            positions[-1],
                            self.COLORS['mid_traffic'],
                            2)
        
        # Frame buffer'ı güncelle
        self.frame_buffer = frame.copy()
        
        return frame

    def calculate_speed(self, positions, timestamps):
        """Araç hızını hesapla (km/saat)"""
        if len(positions) < 2 or len(timestamps) < 2:
            return None
            
        pixel_distance = np.sqrt(
            (positions[-1][0] - positions[0][0])**2 +
            (positions[-1][1] - positions[0][1])**2
        )
        
        time_diff = timestamps[-1] - timestamps[0]  # saniye
        if time_diff == 0:
            return None
            
        # Piksel/saniye'yi km/saat'e çevir
        meter_per_pixel = self.real_distance / self.speed_measurement_distance
        speed = (pixel_distance * meter_per_pixel / time_diff) * 3.6
        
        return round(speed, 1)

    def detect_lane_violation(self, positions):
        """Şerit ihlali tespiti"""
        if len(positions) < 3:
            return False
            
        # Son 3 noktayı kullan
        recent_positions = positions[-3:]
        
        # Ani şerit değişimi kontrolü
        x_positions = [pos[0] for pos in recent_positions]
        max_deviation = max(x_positions) - min(x_positions)
        
        return max_deviation > self.lane_width * 1.5

    def save_stats(self):
        """İstatistikleri JSON olarak kaydet"""
        stats = {
            "timestamp": datetime.now().isoformat(),
            "lanes": self.lanes,
            "total_violations": sum(1 for v in self.vehicle_states.values() if v.get("violation", False))
        }
        
        self.stats_history.append(stats)
        
        with open("traffic_stats.json", "w", encoding="utf-8") as f:
            json.dump({
                "history": self.stats_history,
                "last_update": datetime.now().isoformat()
            }, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    analyzer = TrafficAnalyzer("test.mp4")
    analyzer.analyze_traffic() 