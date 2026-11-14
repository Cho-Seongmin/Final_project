import cv2
from detector import AlcoholDetector
from logic import AlcoholCalculator

def main():
    print("=== [Smart Alcohol Shopping Assistant] 시작 ===")
    
    # 1. 터미널에서 인원수와 인당 목표 주량(소주 병 수) 입력받기
    try:
        people_count = int(input("참여 인원수를 입력하세요 (예: 2): "))
        target_soju_bottles = float(input("인당 권장/목표 소주 병 수를 입력하세요 (예: 1.0): "))
    except ValueError:
        print("잘못된 입력입니다. 숫자로 입력해 주세요.")
        return

    # 2. 계산기 모듈 초기화 및 입력받은 값으로 목표 설정
    calculator = AlcoholCalculator()
    target_grams = calculator.set_group_tolerance_by_soju_bottles(people_count, target_soju_bottles)
    print(f"\n[설정 완료] 총 목표 순수 알코올: {target_grams}g (인원: {people_count}명, 인당: {target_soju_bottles}병 기준)\n")

    # 3. YOLO 탐지기 초기화 및 웹캠 켜기
    detector = AlcoholDetector(model_path="models/best.pt")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("오류: 웹캠을 열 수 없습니다.")
        return

    print("=== 실시간 장바구니 분석 시작 (종료하려면 'q' 키를 누르세요) ===")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # A. 감지된 주류 개수 추출
        counts = detector.get_detected_counts(frame)

        # B. 알코올 및 BAC 계산 결과 가져오기
        msg = calculator.evaluate_purchase(counts)
        current_grams = calculator.calculate_current_grams(counts)
        min_bac, max_bac = calculator.calculate_bac_range(counts)

        # C. 화면에 시각화 (YOLO 박스 + 텍스트 오버레이)
        results = detector.model(frame, conf=0.7)
        annotated_frame = results[0].plot()

        # 화면 상단에 텍스트 렌더링
        cv2.putText(annotated_frame, f"Current Alcohol: {current_grams}g", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        
        cv2.putText(annotated_frame, f"혈중농도알코올 (50~80kg): {min_bac}% ~ {max_bac}%", (20, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        
        cv2.putText(annotated_frame, msg[:60], (20, 120), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("Smart Alcohol Shopping", annotated_frame)

        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("=== 프로그램을 종료합니다 ===")

if __name__ == "__main__":
    main()