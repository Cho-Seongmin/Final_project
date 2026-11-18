import os
from collections import Counter
import cv2
import gradio as gr
from ultralytics import YOLO

# =====================================================================
# 1. 주류 표준 데이터 및 모델 경로 설정
# =====================================================================
# 학습된 클래스명: 'soju-bottle', 'beer-bottle', 'beer-can'
# 순수 알코올(g) = 용량(ml) * (도수 / 100) * 0.8 (알코올 비중)
STANDARD_SOJU_ML = 360
STANDARD_SOJU_ABV = 16.0
STANDARD_SOJU_ALCOHOL_G = STANDARD_SOJU_ML * (STANDARD_SOJU_ABV / 100.0) * 0.8  # 46.08g

BEER_BOTTLE_ML = 500
BEER_BOTTLE_ABV = 4.5
BEER_BOTTLE_ALCOHOL_G = BEER_BOTTLE_ML * (BEER_BOTTLE_ABV / 100.0) * 0.8  # 18.00g

MAX_PEOPLE = 8 # UI에서 개별 주량을 입력받을 최대 인원수


def find_best_model() -> str:
    """프로젝트 폴더 내에서 학습된 best.pt 파일의 위치를 자동으로 탐색합니다."""
    candidate_paths = [
        "best.pt",
        "weights/best.pt",
        "models/best.pt",
        "runs/detect/drink_detector_s/weights/best.pt",
        "runs/detect/drink_detector_s_v2/weights/best.pt",
    ]
    for path in candidate_paths:
        if os.path.exists(path):
            return path
    return "best.pt"


MODEL_PATH = find_best_model()
loaded_model = None
model_load_error = None

try:
    if os.path.exists(MODEL_PATH):
        loaded_model = YOLO(MODEL_PATH)
    else:
        model_load_error = (
            f"'{MODEL_PATH}' 파일을 찾을 수 없습니다. "
            "코랩에서 다운받은 best.pt 파일을 프로젝트 최상위 폴더에 넣어주세요."
        )
except Exception as e:
    model_load_error = f"모델 로딩 중 오류가 발생했습니다: {str(e)}"


# =====================================================================
# 2. UI 동적 제어 및 알코올 계산 핵심 로직
# =====================================================================
def update_member_sliders(num_people: int):
    """선택된 참석 인원수만큼만 개별 주량 슬라이더를 화면에 표시합니다."""
    updates = []
    for i in range(MAX_PEOPLE):
        updates.append(gr.update(visible=(i < int(num_people))))
    return updates


def build_result_html(
    counts: Counter,
    total_soju_target: float,
    can_size_option: str,
    error_msg: str = None,
) -> str:
    """분석 결과를 시각적인 대시보드 형태의 HTML 카드로 생성합니다."""
    if error_msg:
        return f"""
        <div style="padding: 20px; border-radius: 14px; background: #FEF2F2; border: 1px solid #FCA5A5; color: #991B1B;">
            <h3 style="margin: 0 0 8px 0; font-size: 18px;">⚠️ 실행 오류</h3>
            <p style="margin: 0; font-size: 14px;">{error_msg}</p>
        </div>
        """

    # 캔맥주 선택 용량에 따른 알코올 양 계산
    can_ml = 355 if "355ml" in can_size_option else 500
    beer_can_alcohol_g = can_ml * (4.5 / 100.0) * 0.8

    drink_specs = {
        "soju-bottle": {
            "label": "🍶 소주병 (360ml · 16%)",
            "unit": "병",
            "alcohol_g": STANDARD_SOJU_ALCOHOL_G,
        },
        "beer-bottle": {
            "label": "🍺 병맥주 (500ml · 4.5%)",
            "unit": "병",
            "alcohol_g": BEER_BOTTLE_ALCOHOL_G,
        },
        "beer-can": {
            "label": f"🥫 캔맥주 ({can_ml}ml · 4.5%)",
            "unit": "캔",
            "alcohol_g": beer_can_alcohol_g,
        },
    }

    # 현재 인식된 술들의 순수 알코올 합계 계산
    current_alcohol_g = 0.0
    rows_html = ""
    for cls_key, spec in drink_specs.items():
        cnt = counts.get(cls_key, 0)
        subtotal_g = cnt * spec["alcohol_g"]
        current_alcohol_g += subtotal_g
        badge_bg = "#EEF2FF" if cnt > 0 else "#F3F4F6"
        badge_color = "#3730A3" if cnt > 0 else "#6B7280"
        rows_html += f"""
        <div style="display: flex; justify-content: space-between; align-items: center; padding: 10px 14px; background: #FFFFFF; border-radius: 10px; border: 1px solid #E5E7EB; margin-bottom: 8px;">
            <span style="font-weight: 600; color: #1F2937; font-size: 14px;">{spec['label']}</span>
            <div>
                <span style="background: {badge_bg}; color: {badge_color}; padding: 4px 10px; border-radius: 999px; font-weight: 700; font-size: 13px; margin-right: 8px;">
                    {cnt}{spec['unit']}
                </span>
                <span style="color: #4B5563; font-size: 13px; font-weight: 500;">({subtotal_g:.1f}g)</span>
            </div>
        </div>
        """

    # 목표 알코올 및 차이 계산
    target_alcohol_g = total_soju_target * STANDARD_SOJU_ALCOHOL_G
    diff_g = target_alcohol_g - current_alcohol_g
    progress_pct = min(100.0, (current_alcohol_g / target_alcohol_g * 100.0)) if target_alcohol_g > 0 else 100.0

    # 상태별 추천 배너 생성 (소주 반 잔 = 약 5.8g을 적정 오차 범위로 설정)
    if diff_g > 6.0:
        need_soju = diff_g / STANDARD_SOJU_ALCOHOL_G
        need_beer_bottle = diff_g / BEER_BOTTLE_ALCOHOL_G
        need_beer_can = diff_g / beer_can_alcohol_g
        status_bg = "#FFFBEB"
        status_border = "#F59E0B"
        status_title_color = "#B45309"
        bar_color = "#F59E0B"
        status_title = f"⚠️ 술이 부족합니다! (부족량: 약 {diff_g:.1f}g)"
        status_desc = f"""
        <ul style="margin: 8px 0 0 0; padding-left: 20px; color: #78350F; font-size: 14px; line-height: 1.6;">
            <li><strong>소주</strong>만 추가할 경우: 약 <strong>{need_soju:.1f}병</strong> 더 구매하세요.</li>
            <li><strong>병맥주(500ml)</strong>만 추가할 경우: 약 <strong>{need_beer_bottle:.1f}병</strong> 더 구매하세요.</li>
            <li><strong>캔맥주({can_ml}ml)</strong>만 추가할 경우: 약 <strong>{need_beer_can:.1f}캔</strong> 더 구매하세요.</li>
        </ul>
        """
    elif diff_g < -6.0:
        over_soju = abs(diff_g) / STANDARD_SOJU_ALCOHOL_G
        status_bg = "#FEF2F2"
        status_border = "#EF4444"
        status_title_color = "#B91C1C"
        bar_color = "#EF4444"
        status_title = f"🛑 목표 주량을 초과했습니다! (초과량: 약 {abs(diff_g):.1f}g)"
        status_desc = f"""
        <p style="margin: 8px 0 0 0; color: #7F1D1D; font-size: 14px; line-height: 1.5;">
            참석자 주량 총합보다 <strong>소주 기준 약 {over_soju:.1f}병</strong> 분량이 더 많습니다. 무리하지 않게 조절해 드세요!
        </p>
        """
    else:
        status_bg = "#ECFDF5"
        status_border = "#10B981"
        status_title_color = "#047857"
        bar_color = "#10B981"
        status_title = "✅ 완벽합니다! 주량에 딱 맞는 양입니다."
        status_desc = """
        <p style="margin: 8px 0 0 0; color: #065F46; font-size: 14px;">
            현재 준비된 술의 총 알코올 양이 참석자들의 주량 총합과 거의 일치합니다. 즐거운 시간 보내세요!
        </p>
        """

    return f"""
    <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #F9FAFB; padding: 20px; border-radius: 16px; border: 1px solid #E5E7EB;">
        <h3 style="margin: 0 0 12px 0; font-size: 17px; color: #111827;">📊 탐지된 주류 및 알코올 환산</h3>
        {rows_html}

        <div style="margin-top: 16px; padding: 14px; background: #FFFFFF; border-radius: 12px; border: 1px solid #E5E7EB;">
            <div style="display: flex; justify-content: space-between; font-size: 13px; font-weight: 600; color: #374151; margin-bottom: 6px;">
                <span>현재 알코올: {current_alcohol_g:.1f}g</span>
                <span>목표 주량: {target_alcohol_g:.1f}g (소주 {total_soju_target:.1f}병)</span>
            </div>
            <div style="width: 100%; height: 12px; background: #E5E7EB; border-radius: 999px; overflow: hidden;">
                <div style="width: {progress_pct:.1f}%; height: 100%; background: {bar_color}; transition: width 0.3s ease;"></div>
            </div>
        </div>

        <div style="margin-top: 16px; padding: 16px; border-radius: 12px; background: {status_bg}; border: 1px solid {status_border};">
            <h4 style="margin: 0; font-size: 16px; color: {status_title_color};">{status_title}</h4>
            {status_desc}
        </div>
    </div>
    """


def analyze_drinks(
    image,
    conf_threshold: float,
    can_size_option: str,
    num_people: int,
    *member_capacities,
):
    """입력된 이미지에서 YOLO 추론을 실행하고 참석자 주량과 비교합니다."""
    if model_load_error or loaded_model is None:
        return None, build_result_html(Counter(), 0.0, can_size_option, error_msg=model_load_error)

    if image is None:
        return None, build_result_html(
            Counter(),
            0.0,
            can_size_option,
            error_msg="먼저 술병 사진을 업로드하거나 웹캠으로 촬영해 주세요.",
        )

    try:
        # Gradio는 RGB 배열을 넘겨주므로 OpenCV/YOLO 시각화를 위해 BGR로 변환
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # YOLO 모델 추론 실행
        results = loaded_model.predict(source=image_bgr, conf=conf_threshold, verbose=False)
        result = results[0]

        # 바운딩 박스가 그려진 결과 이미지를 다시 RGB로 변환
        annotated_bgr = result.plot()
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

        # 검출된 클래스 개수 집계
        names = loaded_model.names
        detected_classes = [names[int(box.cls)] for box in result.boxes]
        counts = Counter(detected_classes)

        # 선택된 인원수만큼의 개인별 주량 합산 (소주 병 기준)
        active_capacities = member_capacities[: int(num_people)]
        total_soju_target = sum(float(c) for c in active_capacities)

        report_html = build_result_html(counts, total_soju_target, can_size_option)
        return annotated_rgb, report_html

    except Exception as e:
        return None, build_result_html(
            Counter(),
            0.0,
            can_size_option,
            error_msg=f"이미지 분석 중 오류가 발생했습니다: {str(e)}",
        )


# =====================================================================
# 3. Gradio 웹 프론트엔드 UI 구성
# =====================================================================
with gr.Blocks(title="🍻 AI 주류 탐지 & 주량 계산기") as demo:
    gr.Markdown(
        """
        # 🍻 AI 술병 인식 & 주량 맞춤형 계산기
        사진을 올리거나 웹캠으로 테이블 위 술병을 비추면, **YOLOv8 모델(`best.pt`)**이 술의 종류와 개수를 파악해
        참석자들의 주량 대비 **부족한 술 또는 초과된 양**을 즉시 계산해 줍니다.
        """
    )

    with gr.Row():
        # [좌측 컬럼] 인원수 & 주량 설정 + 카메라 입력
        with gr.Column(scale=1):
            gr.Markdown("### 1️⃣ 참석 인원 및 개인별 주량 설정")
            num_people_slider = gr.Slider(
                minimum=1,
                maximum=MAX_PEOPLE,
                value=3,
                step=1,
                label="👥 참석 인원수 (명)",
            )

            member_sliders = []
            default_caps = [1.5, 1.0, 2.0, 1.0, 1.5, 1.0, 1.0, 1.0]
            for idx in range(MAX_PEOPLE):
                slider = gr.Slider(
                    minimum=0.0,
                    maximum=5.0,
                    value=default_caps[idx],
                    step=0.5,
                    label=f"🧑 {idx + 1}번 참석자 주량 (소주 병 단위)",
                    visible=(idx < 3),
                )
                member_sliders.append(slider)

            gr.Markdown("### 2️⃣ 주류 사진 입력 및 모델 옵션")
            input_image = gr.Image(
                sources=["upload", "webcam"],
                type="numpy",
                label="📸 술병 사진 업로드 또는 웹캠 촬영",
            )

            with gr.Accordion("⚙️ 세부 인식 옵션 열기", open=False):
                conf_slider = gr.Slider(
                    minimum=0.10,
                    maximum=0.95,
                    value=0.65,
                    step=0.05,
                    label="🎯 탐지 신뢰도 기준 (Confidence) — F1 최적값 0.65~0.70 권장",
                )
                can_size_radio = gr.Radio(
                    choices=["500ml (큰캔)", "355ml (일반캔)"],
                    value="500ml (큰캔)",
                    label="🥫 캔맥주 표준 용량 설정",
                )

            analyze_btn = gr.Button("🔍 술병 인식 및 주량 계산하기", variant="primary", size="lg")

        # [우측 컬럼] YOLO 바운딩 박스 결과 + 알코올 분석 리포트
        with gr.Column(scale=1):
            gr.Markdown("### 3️⃣ AI 인식 결과 및 구매 추천")
            output_image = gr.Image(type="numpy", label="🖼️ YOLO 객체 탐지 화면")
            output_report = gr.HTML(
                value="""
                <div style="padding: 24px; border-radius: 14px; background: #F3F4F6; color: #4B5563; text-align: center;">
                    왼쪽에서 참석자 주량을 설정하고 술병 사진을 입력한 뒤 <strong>[술병 인식 및 주량 계산하기]</strong> 버튼을 눌러주세요.
                </div>
                """
            )

    # 인원수 변경 시 개별 주량 슬라이더 개수 동적 업데이트
    num_people_slider.change(
        fn=update_member_sliders,
        inputs=[num_people_slider],
        outputs=member_sliders,
    )

    # 분석 버튼 클릭 이벤트 연결
    analyze_btn.click(
        fn=analyze_drinks,
        inputs=[input_image, conf_slider, can_size_radio, num_people_slider, *member_sliders],
        outputs=[output_image, output_report],
    )


if __name__ == "__main__":
    # 브라우저에서 자동으로 열리며, 로컬 네트워크에서도 접근 가능
    demo.launch(inbrowser=True)