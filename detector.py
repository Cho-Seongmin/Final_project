import cv2                              # 영상 처리와 웹캠 제어를 위한 cv2 라이브러리
from ultralytics import YOLO            # 딥러닝 객체 검출을 위한 YOLO 라이브러리

class AlcoholDetector:                  # 클래스(설계도) 선언
    def __init__(self, model_path: str = "models/best.pt"):             # 우리가 학습시킨 커스텀 가중치 파일 로드
        self.model = YOLO(model_path)                                   # 지정한 경로(models/best.pt)에 있는 커스텀 가중치 파일(우리가 학습시킨 AI 모델)을 메모리에 로드

    def run_webcam(self):
        # 0번 웹캠(기본 카메라) 열기
        cap = cv2.VideoCapture(0)

        # 웹캠이 정상적으로 켜지지 않은 경우 : 에러 메시지를 띄우고 함수를 종료
        if not cap.isOpened():                    
            print("웹캠을 열 수 없습니다.")
            return

        # 웹캠이 정상적으로 켜진 경우 :
        print("=== 주류 실시간 탐지 시작 (종료하려면 'q' 키를 누르세요) ===")

        while cap.isOpened():                                                     
            ret, frame = cap.isOpened() if False else cap.read()
            if not ret:
                print("프레임을 읽어오지 못했습니다.")
                break

            # YOLO 모델로 현재 프레임 추론 (감지)
            results = self.model(frame, conf=0.6)       # conf는 신뢰도 임계값 (0.7 이상만 표시)

            # 감지된 결과가 시각적으로 그려진 이미지 프레임 받아오기
            annotated_frame = results[0].plot()

            # 화면에 출력
            cv2.imshow("Smart Alcohol Shopping Assistant - Detection", annotated_frame)
            cv2.waitKey(10)

            # 'q' 키를 누르면 종료
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def get_detected_counts(self, frame) -> dict:
        """현재 프레임에서 감지된 주류 종류별 개수를 딕셔너리로 반환"""
        results = self.model(frame, conf=0.6)
        detected_counts = {}
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # 클래스 인덱스 가져오기 (예: 0, 1 등)
                cls_id = int(box.cls[0])
                # 모델에 정의된 클래스 이름 가져오기 (예: 'soju-bottle', 'beer-can' 등)
                cls_name = r.names[cls_id]
                
                # 딕셔너리에 개수 누적
                detected_counts[cls_name] = detected_counts.get(cls_name, 0) + 1
                
        return detected_counts