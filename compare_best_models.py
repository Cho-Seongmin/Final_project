import os
import yaml
import pandas as pd
from ultralytics import YOLO

# =====================================================================
# 1. 비교할 팀원들의 모델 목록 설정
# 프로젝트 폴더(OSS_Final_Team_Project)에 각자의 모델 파일을 넣고 이름과 경로를 적어주세요.
# =====================================================================
CANDIDATE_MODELS = {
    "태유_YOLOv26s": "taeyu_best_26s.pt",
    "태유_YOLOv8s": "taeyu_best_8s.pt",
    "성민_YOLOv8n": "seongmin_best.pt",
    #"재하_YOLOv8m": "jaeha_best.pt",
}

ORIGINAL_YAML_PATH = "주류탐지_데이터셋/data.yaml"
TEMP_YAML_PATH = "_temp_eval_data.yaml"
CONF_THRESHOLD = 0.62  # 웹 앱(code_artifact.py)과 동일한 신뢰도 임계값 적용


def prepare_local_yaml(original_yaml: str, temp_yaml: str) -> str:
    """
    코랩(Colab)이나 다른 PC에서 작성된 data.yaml의 경로 오류를 방지하기 위해
    현재 맥북의 절대 경로로 맞춘 임시 평가용 yaml 파일을 생성합니다.
    """
    if not os.path.exists(original_yaml):
        raise FileNotFoundError(f"데이터셋 설정 파일을 찾을 수 없습니다: {original_yaml}")

    with open(original_yaml, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    dataset_abs_dir = os.path.abspath(os.path.dirname(original_yaml))
    data["path"] = dataset_abs_dir
    data["train"] = "train/images"
    data["val"] = "valid/images"
    data["test"] = "test/images"

    with open(temp_yaml, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True)

    return temp_yaml


def evaluate_models():
    """등록된 모든 후보 모델을 동일한 test 데이터셋으로 평가하고 순위표를 출력합니다."""
    eval_results = []

    try:
        yaml_to_use = prepare_local_yaml(ORIGINAL_YAML_PATH, TEMP_YAML_PATH)

        for model_name, weight_path in CANDIDATE_MODELS.items():
            # 폴더에 해당 모델 파일이 없으면 안내 후 다음 모델로 넘어감
            if not os.path.exists(weight_path):
                print(f"⚠️ 건너뜀: '{weight_path}' 파일이 폴더에 없습니다.")
                continue

            print(f"🔍 [{model_name}] 테스트 데이터셋 평가 진행 중...")
            try:
                model = YOLO(weight_path)

                # 학습에 전혀 쓰이지 않은 'test' 분할 데이터로 공정하게 채점
                # macOS 환경에서 NMS 시간 초과 병목을 피하기 위해 device="cpu" 사용
                metrics = model.val(
                    data=yaml_to_use,
                    split="test",
                    conf=CONF_THRESHOLD,
                    device="cpu",
                    verbose=False,
                )

                # 핵심 성능 지표 추출
                precision = float(metrics.box.mp)
                recall = float(metrics.box.mr)
                f1_score = (2 * precision * recall) / (precision + recall + 1e-9)
                map50 = float(metrics.box.map50)
                map50_95 = float(metrics.box.map)
                inference_ms = float(metrics.speed["inference"])

                eval_results.append({
                    "모델명": model_name,
                    "파일명": weight_path,
                    "mAP50-95(종합)": round(map50_95, 4),
                    "mAP50": round(map50, 4),
                    "F1-Score": round(f1_score, 4),
                    "Precision(오탐방지)": round(precision, 4),
                    "Recall(미탐방지)": round(recall, 4),
                    "속도(ms)": round(inference_ms, 1),
                })

            except Exception as model_err:
                print(f"❌ [{model_name}] 평가 중 오류 발생: {model_err}")

        # 결과가 1개 이상 있으면 mAP50-95 기준으로 내림차순 정렬 후 출력
        if eval_results:
            df = pd.DataFrame(eval_results)
            df = df.sort_values(by="mAP50-95(종합)", ascending=False).reset_index(drop=True)
            df.index = df.index + 1  # 순위를 1등부터 표시

            print("\n" + "=" * 85)
            print("🏆 팀원별 모델 성능 비교 순위표 (기준: test 데이터셋)")
            print("=" * 85)
            print(df.to_string())
            print("=" * 85)

            # 주피터 노트북(.ipynb) 환경인 경우 깔끔한 HTML 표로도 함께 렌더링
            try:
                from IPython.display import display
                display(df)
            except ImportError:
                pass
        else:
            print("\n평가할 수 있는 모델 파일이 없습니다. CANDIDATE_MODELS의 파일 경로를 확인해 주세요.")

    except Exception as e:
        print(f"🚨 전체 평가 실행 오류: {e}")

    finally:
        # 평가 종료 후 생성된 임시 yaml 파일 자동 정리
        if os.path.exists(TEMP_YAML_PATH):
            os.remove(TEMP_YAML_PATH)


if __name__ == "__main__":
    evaluate_models()