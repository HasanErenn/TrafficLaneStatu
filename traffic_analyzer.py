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
        self.calibration_frames = self.fps * 5  # 5 saniyelik kalibrasyon
        self.current_frame = 0
        self.is_calibrated = False
        self.calibration_vehicles = []  # Kalibrasyon sırasındaki araç pozisyonları
        # Geçiş çizgisi ile bir kez sayma için sayaç
        self.counted_ids = set()
        
        # Şerit tespiti için parametreler
        self.lane_detection_line_y = int(self.frame_height * 0.4)
        self.counting_line_y1 = int(self.frame_height * 0.55)  # İlk çizgi yukarı taşındı (0.65 -> 0.55)
        self.counting_line_y2 = int(self.frame_height * 0.85)  # İkinci çizgi aşağı taşındı (0.75 -> 0.85)
        
        # ROI parametreleri
        self.roi_start_y = int(self.frame_height * 0.3)
        self.roi_end_y = self.frame_height
        
        # Şerit bilgileri
        self.lanes = {}
        self.lane_width = 100  # Başlangıç değeri, kalibrasyon sırasında güncellenecek
        self.min_lanes = 2
        self.max_lanes = 3  # Maksimum şerit sayısı 4'ten 3'e düşürüldü
        
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
        
        # Sayılan araçların bilgilerini tutmak için yeni değişken
        self.counted_vehicles = {}  # {tid: {'lane_id': lane_id, 'last_pos': (x1, y1)}}

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
        cap = cv2.VideoCapture(self.source)
        
        # Video bitene kadar devam et
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Frame'i işle
            frame = self.process_frame(frame)
            
            # Görüntüyü göster
            cv2.imshow("Trafik Analizi", frame)
            
            # 1ms bekle ve q tuşuna basılırsa çık
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
        # Kaynakları serbest bırak
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
            # Sadece 1, 2 ve 3 numaralı şeritleri kontrol et
            if int(lane_id) > 3:
                continue
            distance = abs(lane_info["center_x"] - x_pos)
            closest_lanes.append((lane_id, distance))
        
        if not closest_lanes:  # Eğer uygun şerit bulunamazsa
            return None
        
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

    def detect_lane_lines(self, frame):
        """Yoldaki beyaz çizgileri tespit et"""
        # ROI bölgesini al
        roi = frame[self.roi_start_y:self.roi_end_y, :]
        
        # Gri tonlamaya çevir
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Gürültüyü azalt
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Kenarları tespit et
        edges = cv2.Canny(blur, 50, 150)
        
        # Beyaz renk maskesi
        lower_white = np.array([200])
        upper_white = np.array([255])
        white_mask = cv2.inRange(gray, lower_white, upper_white)
        
        # Kenarlar ve beyaz maske birleştir
        combined_mask = cv2.bitwise_and(edges, white_mask)
        
        # Hough dönüşümü ile çizgileri bul
        lines = cv2.HoughLinesP(
            combined_mask,
            rho=1,
            theta=np.pi/180,
            threshold=50,
            minLineLength=100,
            maxLineGap=50
        )
        
        if lines is None:
            return []
        
        # Çizgileri filtrele (dikeye yakın olanları ele)
        valid_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 - x1 == 0:  # Dikey çizgi
                continue
            
            angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            if 70 < angle < 110:  # Yataya yakın çizgileri ele
                continue
                
            valid_lines.append((x1, y1 + self.roi_start_y, x2, y2 + self.roi_start_y))
        
        return valid_lines

    def cluster_lane_lines(self, lines):
        """Tespit edilen çizgileri şeritlere göre grupla"""
        if not lines:
            return []
            
        # X koordinatlarını topla
        x_coords = []
        for x1, _, x2, _ in lines:
            x_coords.append((x1 + x2) / 2)
        
        # DBSCAN ile gruplama yap
        clustering = DBSCAN(eps=self.lane_width * 0.7, min_samples=2).fit(np.array(x_coords).reshape(-1, 1))
        
        # Her grup için ortalama x koordinatı hesapla
        clusters = {}
        for i, label in enumerate(clustering.labels_):
            if label == -1:  # Gürültü
                continue
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(x_coords[i])
        
        # Ortalama konumları hesapla ve sırala
        lane_centers = []
        for points in clusters.values():
            avg_x = sum(points) / len(points)
            lane_centers.append(avg_x)
        
        return sorted(lane_centers)

    def verify_lane_consistency(self, lane_centers):
        """Şerit pozisyonlarının tutarlılığını kontrol et"""
        if not lane_centers or len(lane_centers) < 2:
            return False, []
            
        # Şeritler arası mesafeleri kontrol et
        distances = []
        for i in range(len(lane_centers) - 1):
            dist = lane_centers[i+1] - lane_centers[i]
            distances.append(dist)
        
        # Mesafelerin ortalaması ve standart sapması
        avg_distance = sum(distances) / len(distances)
        std_distance = np.std(distances)
        
        # Mesafe tutarlılığı kontrolü
        if std_distance > avg_distance * 0.3:  # %30'dan fazla sapma varsa şüpheli
            return False, []
            
        # Minimum ve maksimum mesafe kontrolü
        min_allowed = self.lane_width * 0.7  # Minimum şerit genişliği
        max_allowed = self.lane_width * 1.5  # Maksimum şerit genişliği
        
        if avg_distance < min_allowed or avg_distance > max_allowed:
            return False, []
            
        return True, lane_centers

    def calibrate_lanes(self):
        """Araçların yan yana dizilimine göre şeritleri tespit et"""
        if len(self.calibration_vehicles) < 10:  # En az 10 gözlem olmalı
            return False
            
        # Araçları x koordinatına göre grupla
        vehicle_clusters = defaultdict(list)
        
        for vehicles in self.calibration_vehicles:
            # Her frame'deki araçları x koordinatına göre sırala
            sorted_vehicles = sorted(vehicles)
            
            if len(sorted_vehicles) >= 2:  # En az 2 araç yan yana olmalı
                # Araçlar arası mesafeleri hesapla
                for i in range(len(sorted_vehicles) - 1):
                    distance = sorted_vehicles[i+1] - sorted_vehicles[i]
                    if distance > 50 and distance < 200:  # Makul mesafe kontrolü
                        mid_point = (sorted_vehicles[i] + sorted_vehicles[i+1]) / 2
                        cluster_id = int(mid_point // 50)  # 50 piksellik gruplama
                        vehicle_clusters[cluster_id].append(distance)
        
        if not vehicle_clusters:
            return False
            
        # Ortalama şerit genişliğini hesapla
        all_distances = []
        for distances in vehicle_clusters.values():
            if len(distances) >= 3:  # Her grup için en az 3 gözlem
                avg_distance = sum(distances) / len(distances)
                all_distances.append(avg_distance)
        
        if not all_distances:
            return False
            
        # Şerit genişliğini güncelle
        self.lane_width = sum(all_distances) / len(all_distances)
        
        # Son frame'deki araçları kullanarak şeritleri belirle
        if self.calibration_vehicles:
            last_vehicles = sorted(self.calibration_vehicles[-1])
            
            if len(last_vehicles) < self.min_lanes:
                return False
                
            # Şeritleri oluştur
            self.lanes.clear()
            
            # İlk aracı referans al
            first_x = last_vehicles[0]
            
            # Emniyet şeridi kontrolü
            if first_x < self.frame_width * 0.15:  # Sol kenardan %15 içeride
                self.lanes["0"] = {
                    "center_x": first_x,
                    "count": 0,
                    "type": "emergency"
                }
                last_vehicles = last_vehicles[1:]  # İlk aracı atla
            
            # Kalan araçları normal şerit olarak işaretle
            for i, x_pos in enumerate(last_vehicles, 1):
                if i > self.max_lanes:  # Maksimum şerit sayısı kontrolü
                    break
                    
                self.lanes[str(i)] = {
                    "center_x": x_pos,
                    "count": 0,
                    "type": "normal"
                }
            
            # Debug görüntüsü
            debug_frame = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)
            
            # Şeritleri çiz
            for lane_id, lane_info in self.lanes.items():
                color = (0, 255, 0) if lane_info["type"] == "normal" else (0, 0, 255)
                
                # Şerit çizgisi
                cv2.line(debug_frame,
                        (int(lane_info["center_x"]), 0),
                        (int(lane_info["center_x"]), self.frame_height),
                        color, 2)
                
                # Şerit etiketi
                cv2.putText(debug_frame,
                          f"Lane {lane_id}",
                          (int(lane_info["center_x"]) - 30, 30),
                          self.FONT,
                          self.FONT_SCALES['caption'],
                          (255, 255, 255),
                          1)
            
            cv2.imshow("Lane Calibration", debug_frame)
            cv2.waitKey(1)
            
            print(f"{len(self.lanes)} şerit tespit edildi ({sum(1 for l in self.lanes.values() if l['type']=='normal')} normal, {sum(1 for l in self.lanes.values() if l['type']=='emergency')} emniyet şeridi)")
            print(f"Ortalama şerit genişliği: {self.lane_width:.1f} piksel")
            return True
            
        return False

    def draw_checkmark(self, frame, x, y):
        """Özel onay işareti çiz"""
        # Onay işareti boyutları
        size = 30
        thickness = 3
        
        # Onay işareti noktaları
        pt1 = (x, y)
        pt2 = (x + int(size * 0.3), y + int(size * 0.5))
        pt3 = (x + size, y - int(size * 0.6))
        
        # Onay işaretini çiz
        cv2.line(frame, pt1, pt2, (0, 0, 255), thickness)
        cv2.line(frame, pt2, pt3, (0, 0, 255), thickness)

    def process_frame(self, frame):
        """Her kareyi işler ve araç yollarını çizer."""
        # Tam frame üzerinde tespit
        results = self.detector(frame, classes=[2,3,5,7], imgsz=320)[0]
        detections = sv.Detections(
            xyxy=results.boxes.xyxy.cpu().numpy(),
            confidence=results.boxes.conf.cpu().numpy(),
            class_id=results.boxes.cls.cpu().numpy().astype(int)
        )
        tracked = self.tracker.update_with_detections(detections)

        # Mevcut frame'deki araç ID'lerini topla
        current_vehicle_ids = set()

        # Araç kutularını çiz ve mevcut araçları işaretle
        for i, xyxy in enumerate(tracked.xyxy):
            tid = tracked.tracker_id[i]
            current_vehicle_ids.add(tid)
            x1, y1, x2, y2 = map(int, xyxy)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # İz pozisyonlarını güncelle ve X pozisyonlarını topla
        cx_info = []
        for i, xyxy in enumerate(tracked.xyxy):
            tid = tracked.tracker_id[i]
            x1, y1, x2, y2 = map(int, xyxy)
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            self.vehicle_states.setdefault(tid, []).append((cx, cy))
            cx_info.append((tid, cx, x1, y1))
            
            # Eğer bu araç daha önce sayıldıysa, bilgilerini güncelle
            if tid in self.counted_vehicles:
                self.counted_vehicles[tid]['last_pos'] = (x1, y1)

        # Görüntüden çıkan araçları temizle
        for tid in list(self.counted_vehicles.keys()):
            if tid not in current_vehicle_ids:
                del self.counted_vehicles[tid]

        # Kalibrasyon
        self.current_frame += 1
        if not self.is_calibrated:
            self.calibration_vehicles.append([cx for _, cx, _, _ in cx_info])
            if self.current_frame >= self.calibration_frames and self.calibrate_lanes():
                self.is_calibrated = True

        # Şerit atama ve sayım
        if self.is_calibrated:
            # İki geçiş çizgisini çiz
            cv2.line(frame,
                     (0, self.counting_line_y1),
                     (self.frame_width, self.counting_line_y1),
                     (255, 0, 0), 2)
            cv2.line(frame,
                     (0, self.counting_line_y2),
                     (self.frame_width, self.counting_line_y2),
                     (255, 0, 0), 2)

            # Her araç için merkez noktayı çiz
            for lane_id, lane_info in self.lanes.items():
                if int(lane_id) <= 3:
                    cv2.circle(frame, (int(lane_info["center_x"]), self.counting_line_y1), 4, (0, 255, 255), -1)
                    cv2.circle(frame, (int(lane_info["center_x"]), self.counting_line_y2), 4, (0, 255, 255), -1)

            for tid, cx, x1, y1 in cx_info:
                positions = self.vehicle_states.get(tid, [])
                lane_id = None
                
                if len(positions) >= 2:
                    prev_cy = positions[-2][1]
                    curr_cy = positions[-1][1]
                    
                    # Her iki çizgiden geçişi kontrol et
                    crossed_line1 = prev_cy < self.counting_line_y1 <= curr_cy
                    crossed_line2 = prev_cy < self.counting_line_y2 <= curr_cy
                    
                    # Herhangi bir çizgiden geçtiyse ve daha önce sayılmadıysa
                    if (crossed_line1 or crossed_line2) and tid not in self.counted_ids:
                        lane_id = self.assign_lane(cx, self.calculate_direction(positions))
                        if lane_id is not None and int(lane_id) <= 3:
                            self.lanes[lane_id]['count'] += 1
                            self.last_minute_counts[lane_id].append(time.time())
                            self.counted_ids.add(tid)
                            # Araç bilgilerini kaydet
                            self.counted_vehicles[tid] = {
                                'lane_id': lane_id,
                                'last_pos': (x1, y1)
                            }

            # Sayılan tüm araçlar için işaretleri göster
            for tid, info in self.counted_vehicles.items():
                x1, y1 = info['last_pos']
                lane_id = info['lane_id']
                
                # Etiketi kutunun üstüne yaz
                cv2.putText(frame,
                          f"Serit {lane_id}",
                          (x1, y1 - 40),
                          self.FONT,
                          self.FONT_SCALES['caption'],
                          (0, 255, 0),
                          1)
                
                # Özel onay işaretini çiz
                self.draw_checkmark(frame, x1, y1 - 15)

        # Sağ paneli ekleyip tek bir görüntü oluştur
        panel = self.create_stats_panel()
        canvas = np.zeros((self.canvas_height, self.canvas_width, 3), dtype=np.uint8)
        canvas[:, :self.frame_width] = frame
        canvas[:, self.frame_width:] = panel
        return canvas

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

    def detect_lanes_by_vehicle_movement(self, tracked_vehicles):
        """Araçların hareket doğrultularını ve mesafelerini kullanarak şerit tespiti yap"""
        # Araçların merkez noktalarını ve doğrultularını topla
        vehicle_positions = []
        vehicle_directions = []
        for vehicle in tracked_vehicles:
            positions = vehicle['positions']
            if len(positions) >= 2:
                # Son iki pozisyonu kullanarak doğrultuyu hesapla
                direction = self.calculate_direction(positions)
                if direction is not None:
                    vehicle_positions.append(positions[-1])  # Son pozisyonu al
                    vehicle_directions.append(direction)

        # Kümeleme için veri hazırlığı
        data = np.array([[pos[0], pos[1], dir] for pos, dir in zip(vehicle_positions, vehicle_directions)])

        # DBSCAN ile kümeleme yap
        clustering = DBSCAN(eps=50, min_samples=2).fit(data)
        labels = clustering.labels_

        # Şerit merkezlerini hesapla
        lane_centers = defaultdict(list)
        for label, pos in zip(labels, vehicle_positions):
            if label != -1:  # Gürültü noktaları hariç
                lane_centers[label].append(pos[0])  # X koordinatını ekle

        # Her kümenin merkezini hesapla
        self.lanes.clear()
        for label, x_coords in lane_centers.items():
            center_x = np.mean(x_coords)
            self.lanes[str(label)] = {"center_x": center_x, "count": 0, "type": "normal"}

        print(f"{len(self.lanes)} şerit tespit edildi.")

if __name__ == "__main__":
    analyzer = TrafficAnalyzer("test.mp4")
    analyzer.analyze_traffic() 