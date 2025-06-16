import matplotlib.pyplot as plt
import re

def visualize_training_log(log_file_path):
    """
    훈련 로그 파일에서 반복 횟수와 손실 값을 추출하여 그래프로 시각화합니다.

    Args:
        log_file_path (str): 분석할 로그 파일의 경로.
    """
    iterations = []
    losses = []

    # 정규 표현식을 사용하여 "숫자\t숫자\t숫자" 형식의 훈련 데이터 라인을 찾습니다.
    # 예: "001\t0.2848\t-5.5000"
    log_pattern = re.compile(r"^\d+\s+([\d\.]+)\s+.*$")

    try:
        with open(log_file_path, 'r') as f:
            for line in f:
                match = log_pattern.match(line.strip())
                if match:
                    # 첫 번째 그룹(손실 값)만 사용합니다.
                    # 반복 횟수는 리스트의 인덱스를 사용하여 순차적으로 표현합니다.
                    losses.append(float(match.group(1)))

        # 데이터가 없는 경우를 대비
        if not losses:
            print("로그 파일에서 유효한 훈련 데이터를 찾을 수 없습니다.")
            return

        iterations = range(1, len(losses) + 1)

        # Matplotlib을 사용하여 그래프 생성
        plt.figure(figsize=(12, 6))
        plt.plot(iterations, losses, label='Training Loss')

        # 그래프 제목 및 라벨 설정
        plt.title('Training Loss over Iterations')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        # 그래프를 이미지 파일로 저장
        save_path = 'loss_visualization.png'
        plt.savefig(save_path)
        print(f"그래프가 '{save_path}' 파일로 저장되었습니다.")

    except FileNotFoundError:
        print(f"오류: 파일을 찾을 수 없습니다 - {log_file_path}")
    except Exception as e:
        print(f"오류가 발생했습니다: {e}")

# 'train_log.txt' 파일에 대해 시각화를 실행합니다.
# 파일이 다른 경로에 있다면 전체 경로를 입력해주세요.
visualize_training_log('train_log.txt')