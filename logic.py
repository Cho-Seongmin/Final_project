class AlcoholCalculator:
    def __init__(self):
        # 기준: 소주 1병 = 360ml, 16.5도 -> 순수 알코올 약 46.9g
        self.standard_soju_grams = 360 * (16.5 / 100) * 0.79
        
        # YOLO 감지된 주종별 규격 (ml, 도수)
        self.drink_specs = {
            "soju-bottle": {"volume": 360, "abv": 16.5},
            "beer-bottle": {"volume": 500, "abv": 4.5},
            "beer-can": {"volume": 355, "abv": 4.5}
        }
        self.alcohol_density = 0.79     # 알코올 비중 상수
        self.total_target_grams = 0.0   # 그룹의 순수 알코올 총량

        # 혈중 농도 알코올 계산 = 순수 알코올 양(g) / 체중(kg) x 성별 계수(남자 0.68, 여자 0.55) x 10
        # 고정 상수 (50kg ~ 80kg 범위, 통합 성별 계수 0.6)
        self.people_count = 0
        self.min_weight = 50.0
        self.max_weight = 80.0
        self.widmark_sex = 0.6

    # 목표 주량 설정 메서드: 주량(소주병 단위)를 알코올로 환산
    def set_group_tolerance_by_soju_bottles(self, people_count: int, target_soju_bottles_per_person: float):
        """사용자에게 '인원'과 '인당 주량(소주병 단위)'만 입력받아 설정"""
        self.people_count = people_count    # 입력받은 인원수
        grams_per_person = target_soju_bottles_per_person * self.standard_soju_grams    # 인당 목표 알코올 양(g)
        self.total_target_grams = people_count * grams_per_person   # 그룹 총 목표 알코올 양(g)
        return round(self.total_target_grams, 1)

    # 현재 장바구니 알코올 총량 계산 메서드
    def calculate_current_grams(self, detected_counts: dict) -> float:
        """카메라로 감지된 주류들을 순수 알코올 총량(g)으로 계산"""
        total_grams = 0.0   # 누적 알코올 양
        for drink_type, count in detected_counts.items():
            if drink_type in self.drink_specs:
                spec = self.drink_specs[drink_type]
                grams_per_item = spec["volume"] * (spec["abv"] / 100) * self.alcohol_density
                total_grams += grams_per_item * count
        return round(total_grams, 1)

    def calculate_bac_range(self, detected_counts: dict) -> tuple[float, float]:
        """
        체중 50kg ~ 80kg 기준 예상 혈중알코올농도(BAC) 구간(최소%, 최대%) 반환
        """
        if self.people_count <= 0:
            return (0.0, 0.0)
            
        total_grams = self.calculate_current_grams(detected_counts)
        per_person_grams = total_grams / self.people_count
        
        # 80kg일 때 최소 농도, 50kg일 때 최대 농도
        min_bac = per_person_grams / (self.max_weight * self.widmark_sex * 10)
        max_bac = per_person_grams / (self.min_weight * self.widmark_sex * 10)
        
        return (round(min_bac, 3), round(max_bac, 3))

    def evaluate_purchase(self, detected_counts: dict) -> str:
        """구매 적정성 + 체중 50~80kg 기준 예상 혈중농도알코올 범위 안내"""
        if self.total_target_grams <= 0 or self.people_count <= 0:
            return "먼저 인원과 주량을 입력해 주세요."

        current_grams = self.calculate_current_grams(detected_counts)
        diff_grams = round(self.total_target_grams - current_grams, 1)

        # :.1f를 붙여서 소수점 첫째 자리까지만 예쁘게 출력되도록 수정
        if diff_grams > 15.0:
            return f"부족함: 목표 주량보다 알코올이 부족합니다. (현재 {current_grams:.1f}g / 목표 {self.total_target_grams:.1f}g)"
        elif -15.0 <= diff_grams <= 15.0:
            return f"적정함: 딱 알맞은 주량 범위 내에 있습니다! (현재 {current_grams:.1f}g)"
        else:
            return f"초과함: 목표 주량보다 알코올이 초과되었습니다! (현재 {current_grams:.1f}g / 목표 {self.total_target_grams:.1f}g)"

    def get_medical_bac_status(self, detected_counts: dict) -> tuple:
        """
        제시된 5단계 의학적·생리적 단계 구분에 따른 예상 혈중알코올농도 상태 및 색상(BGR) 반환
        - 0.03% 미만: 가벼운 취기 (초록색)
        - 0.03% ~ 0.10% 미만: 취기 및 감각 저하 (노란색)
        - 0.10% ~ 0.20% 미만: 만취 및 운동 조정 능력 상실 (주황색)
        - 0.20% ~ 0.30% 미만: 심각한 중독 (진한 주황색)
        - 0.30% 이상: 위험 수준 (빨간색)
        """
        min_bac, max_bac = self.calculate_bac_range(detected_counts)
        avg_bac = (min_bac + max_bac) / 2.0

        if avg_bac < 0.03:
            return (0, 255, 0)             # 초록색
        elif 0.03 <= avg_bac < 0.10:
            return (0, 255, 255)           # 노란색
        elif 0.10 <= avg_bac < 0.20:
            return (0, 165, 255)           # 주황색
        elif 0.20 <= avg_bac < 0.30:
            return (0, 69, 255)            # 진한 주황색
        else:
            return (0, 0, 255)             # 빨간색