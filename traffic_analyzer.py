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
        # Yol segmentasyonu için YOLO-seg modeli
        self.segmentor = YOLO("yolov8n-seg.pt", task='segment').to(self.device)
        
        # Object tracker
        self.tracker = sv.ByteTrack()
        
        # Video boyutlarını al
        cap = cv2.VideoCapture(self.source)
        self.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(cap.get(cv2.CAP_PROP_FPS))
        cap.release()
        
        # Kalibrasyon için parametreler
        self.calibration_frames = self.fps * 2  # 2 saniyelik kalibrasyon
        self.current_frame = 0
        self.is_calibrated = False
        
        # Şerit tespiti için parametreler
        self.lane_detection_line_y = int(self.frame_height * 0.7)
        self.counting_line_y = int(self.frame_height * 0.5)
        
        # Araç takibi için parametreler
        self.vehicle_states = {}
        self.min_track_points = 5
        self.direction_threshold = 15
        
        # Şerit bilgileri
        self.lanes = {}  # {lane_id: {"center_x": x, "direction": angle, "count": 0, "is_emergency": bool}}
        self.lane_width = 100
        self.max_lanes = 4  # Maksimum şerit sayısı (1 emniyet + 3 ana şerit)
        self.emergency_lane_x = int(self.frame_width * 0.1)  # Emniyet şeridi konumu
        
        # Araç sınıfları
        self.vehicle_classes = {
            2: "Araba",
            3: "Motosiklet",
            5: "Otobüs",
            7: "Kamyon"
        }
        
        # Sınıf bazlı sayaçlar
        self.class_counts = defaultdict(lambda: defaultdict(int))
        
        # Hız hesaplama için parametreler
        self.speed_measurement_distance = abs(self.counting_line_y - self.lane_detection_line_y)  # piksel
        self.real_distance = 10  # metre (kalibre edilmeli)
        
        # İstatistik kayıt
        self.stats_history = []
        
        # Görselleştirme parametreleri
        self.LANE_COLOR = (0, 0, 255)  # Normal şerit rengi (Kırmızı)
        self.EMERGENCY_LANE_COLOR = (0, 165, 255)  # Emniyet şeridi rengi (Turuncu)
        self.VEHICLE_COLOR = (0, 255, 0)
        self.TEXT_COLOR = (255, 255, 255)
        self.VECTOR_COLOR = (255, 0, 0)
        self.VIOLATION_COLOR = (0, 0, 255)
        
        # İstatistik paneli için
        self.canvas_width = self.frame_width + 300
        self.canvas_height = self.frame_height

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
            
        # Emniyet şeridini kontrol et
        if x_pos < self.emergency_lane_x + self.lane_width/2:
            if "0" not in self.lanes:
                self.lanes["0"] = {
                    "center_x": self.emergency_lane_x,
                    "direction": direction,
                    "count": 0,
                    "is_emergency": True
                }
            return "0"
            
        # Mevcut şeritleri kontrol et (emniyet şeridi hariç)
        for lane_id, lane_info in self.lanes.items():
            if lane_info["is_emergency"]:
                continue
                
            if (abs(lane_info["center_x"] - x_pos) < self.lane_width and
                abs(lane_info["direction"] - direction) < self.direction_threshold):
                return lane_id
        
        # Yeni şerit oluştur (maksimum şerit sayısını kontrol et)
        if len(self.lanes) < self.max_lanes:
            new_lane_id = str(len(self.lanes))
            # Emniyet şeridi varsa yeni ID'yi bir artır
            if "0" in self.lanes:
                new_lane_id = str(len(self.lanes))
            
            self.lanes[new_lane_id] = {
                "center_x": x_pos,
                "direction": direction,
                "count": 0,
                "is_emergency": False
            }
            return new_lane_id
        
        # En yakın şeridi bul
        min_distance = float('inf')
        closest_lane = None
        for lane_id, lane_info in self.lanes.items():
            if lane_info["is_emergency"]:
                continue
            distance = abs(lane_info["center_x"] - x_pos)
            if distance < min_distance:
                min_distance = distance
                closest_lane = lane_id
        
        return closest_lane

    def process_frame(self, frame):
        """Her kareyi işle"""
        self.current_frame += 1
        current_time = time.time()
        
        # Kalibrasyon durumunu kontrol et
        if not self.is_calibrated:
            if self.current_frame >= self.calibration_frames:
                self.is_calibrated = True
                print("Kalibrasyon tamamlandı, analiz başlıyor...")
                
                # Emniyet şeridini oluştur
                self.lanes["0"] = {
                    "center_x": self.emergency_lane_x,
                    "direction": 0,
                    "count": 0,
                    "is_emergency": True
                }
            else:
                # Kalibrasyon sırasında çizgileri göster
                cv2.line(frame, (0, self.lane_detection_line_y),
                        (frame.shape[1], self.lane_detection_line_y),
                        (255, 0, 0), 2)
                
                cv2.line(frame, (0, self.counting_line_y),
                        (frame.shape[1], self.counting_line_y),
                        (255, 255, 0), 2)
                
                cv2.putText(frame,
                          f"Kalibrasyon: {self.current_frame}/{self.calibration_frames}",
                          (20, 50),
                          cv2.FONT_HERSHEY_SIMPLEX,
                          1.0, self.TEXT_COLOR, 2)
                return frame
        
        # Yol alanını tespit et
        road_mask = self.detect_road(frame)
        
        # Araçları tespit et
        detections = self.detector(frame, classes=[2, 3, 5, 7])[0]
        boxes = detections.boxes.cpu().numpy()
        
        # Tracker'ı güncelle
        detections = sv.Detections(
            xyxy=boxes.xyxy,
            confidence=boxes.conf,
            class_id=boxes.cls.astype(int),
            tracker_id=None
        )
        
        if len(detections) > 0:
            detections = self.tracker.update_with_detections(detections)
            
            # Her aracı işle
            for i in range(len(detections)):
                xyxy = detections.xyxy[i]
                tracker_id = detections.tracker_id[i]
                class_id = detections.class_id[i]
                
                # Araç merkezi
                center_x = int((xyxy[0] + xyxy[2]) / 2)
                center_y = int((xyxy[1] + xyxy[3]) / 2)
                current_pos = (center_x, center_y)
                
                # Araç geçmişini güncelle
                if tracker_id not in self.vehicle_states:
                    self.vehicle_states[tracker_id] = {
                        "positions": [current_pos],
                        "timestamps": [current_time],
                        "lane_id": None,
                        "counted": False,
                        "class_id": class_id,
                        "violation": False,
                        "speed": None
                    }
                else:
                    self.vehicle_states[tracker_id]["positions"].append(current_pos)
                    self.vehicle_states[tracker_id]["timestamps"].append(current_time)
                
                # Şerit tespiti (lane_detection_line üzerinde)
                if (not self.vehicle_states[tracker_id]["lane_id"] and 
                    center_y > self.lane_detection_line_y and
                    len(self.vehicle_states[tracker_id]["positions"]) >= self.min_track_points):
                    
                    # Hareket yönünü hesapla
                    direction = self.calculate_direction(
                        self.vehicle_states[tracker_id]["positions"]
                    )
                    
                    if direction is not None:
                        # Şerit ataması yap
                        lane_id = self.assign_lane(center_x, direction)
                        if lane_id:
                            self.vehicle_states[tracker_id]["lane_id"] = lane_id
                
                # Araç sayımı (counting_line üzerinde)
                if (self.vehicle_states[tracker_id]["lane_id"] and 
                    not self.vehicle_states[tracker_id]["counted"] and 
                    center_y > self.counting_line_y):
                    
                    lane_id = self.vehicle_states[tracker_id]["lane_id"]
                    self.lanes[lane_id]["count"] += 1
                    self.vehicle_states[tracker_id]["counted"] = True
                    
                    # Sınıf bazlı sayım
                    vehicle_class = self.vehicle_classes[class_id]
                    self.class_counts[lane_id][vehicle_class] += 1
                    
                    # Hız hesaplama
                    speed = self.calculate_speed(
                        self.vehicle_states[tracker_id]["positions"],
                        self.vehicle_states[tracker_id]["timestamps"]
                    )
                    if speed:
                        self.vehicle_states[tracker_id]["speed"] = speed
                
                # Şerit ihlali kontrolü
                if len(self.vehicle_states[tracker_id]["positions"]) >= 3:
                    violation = self.detect_lane_violation(
                        self.vehicle_states[tracker_id]["positions"]
                    )
                    self.vehicle_states[tracker_id]["violation"] = violation
                
                # Aracı çiz
                color = self.VIOLATION_COLOR if self.vehicle_states[tracker_id]["violation"] else self.VEHICLE_COLOR
                cv2.rectangle(frame, 
                            (int(xyxy[0]), int(xyxy[1])),
                            (int(xyxy[2]), int(xyxy[3])), 
                            color, 2)
                
                # Araç bilgilerini göster
                vehicle_class = self.vehicle_classes[class_id]
                speed = self.vehicle_states[tracker_id].get("speed", None)
                info_text = f"{vehicle_class}"
                if speed:
                    info_text += f" {speed} km/s"
                
                cv2.putText(frame,
                          info_text,
                          (int(xyxy[0]), int(xyxy[1] - 10)),
                          cv2.FONT_HERSHEY_SIMPLEX,
                          0.5, self.TEXT_COLOR, 2)
                
                # Hareket vektörünü çiz
                positions = self.vehicle_states[tracker_id]["positions"]
                if len(positions) >= 2:
                    cv2.line(frame,
                            positions[-2],
                            positions[-1],
                            self.VECTOR_COLOR, 2)
        
        # Her 30 karede bir istatistikleri kaydet
        if self.current_frame % 30 == 0:
            self.save_stats()
        
        return frame

    def create_stats_panel(self):
        """İstatistik paneli oluştur"""
        # Boş panel oluştur
        panel = np.zeros((self.frame_height, 300, 3), dtype=np.uint8)
        
        # Başlık
        cv2.putText(panel,
                  "TRAFİK ANALİZİ",
                  (20, 30),
                  cv2.FONT_HERSHEY_SIMPLEX,
                  0.7, self.TEXT_COLOR, 2)
        
        y_offset = 70
        
        # Şerit bazlı istatistikler
        for lane_id, lane_info in self.lanes.items():
            lane_type = "Emniyet Şeridi" if lane_info["is_emergency"] else f"{int(lane_id)}. Şerit"
            total_count = lane_info["count"]
            
            # Şerit başlığı
            cv2.putText(panel,
                      f"{lane_type}:",
                      (20, y_offset),
                      cv2.FONT_HERSHEY_SIMPLEX,
                      0.6, self.TEXT_COLOR, 2)
            y_offset += 25
            
            # Toplam araç sayısı
            cv2.putText(panel,
                      f"Toplam: {total_count}",
                      (40, y_offset),
                      cv2.FONT_HERSHEY_SIMPLEX,
                      0.5, self.TEXT_COLOR, 1)
            y_offset += 20
            
            # Araç sınıflarına göre sayımlar
            for vehicle_class, count in self.class_counts[lane_id].items():
                cv2.putText(panel,
                          f"{vehicle_class}: {count}",
                          (40, y_offset),
                          cv2.FONT_HERSHEY_SIMPLEX,
                          0.5, self.TEXT_COLOR, 1)
                y_offset += 20
            
            # Doluluk oranı (son 10 saniye)
            recent_vehicles = sum(1 for v in self.vehicle_states.values()
                                if v["lane_id"] == lane_id and
                                time.time() - v["timestamps"][-1] < 10)
            occupancy = min(100, int((recent_vehicles / 5) * 100))  # 5 araç = %100 doluluk
            
            cv2.putText(panel,
                      f"Doluluk: %{occupancy}",
                      (40, y_offset),
                      cv2.FONT_HERSHEY_SIMPLEX,
                      0.5, self.TEXT_COLOR, 1)
            y_offset += 30
        
        # İhlal istatistikleri
        total_violations = sum(1 for v in self.vehicle_states.values() if v.get("violation", False))
        cv2.putText(panel,
                  f"Şerit İhlalleri: {total_violations}",
                  (20, y_offset),
                  cv2.FONT_HERSHEY_SIMPLEX,
                  0.6, self.TEXT_COLOR, 2)
        y_offset += 30
        
        # Ortalama hız
        speeds = [v["speed"] for v in self.vehicle_states.values() 
                 if v.get("speed") is not None]
        if speeds:
            avg_speed = sum(speeds) / len(speeds)
            cv2.putText(panel,
                      f"Ort. Hız: {avg_speed:.1f} km/s",
                      (20, y_offset),
                      cv2.FONT_HERSHEY_SIMPLEX,
                      0.6, self.TEXT_COLOR, 2)
        
        return panel

    def analyze_traffic(self):
        """Ana analiz döngüsü"""
        cap = cv2.VideoCapture(self.source)
        
        while cap.isOpened():
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
            cv2.imshow("Traffic Analysis", combined_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

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
            "class_counts": dict(self.class_counts),
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