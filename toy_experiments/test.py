import cv2
import numpy as np

def create_difference_image(image_path1, image_path2, output_path='C:/Users/coldbrew/VTON-project/tryon-images/diff/difference_image.jpg'):
    # 이미지 읽기
    img1 = cv2.imread(image_path1)
    img2 = cv2.imread(image_path2)
    
    # 이미지 로드 확인
    if img1 is None or img2 is None:
        print("Error: 하나 이상의 이미지를 불러올 수 없습니다.")
        return
    
    # 이미지 크기 일치 확인 및 조정
    if img1.shape != img2.shape:
        # 두 이미지 크기를 더 작은 쪽에 맞춤
        height = min(img1.shape[0], img2.shape[0])
        width = min(img1.shape[1], img2.shape[1])
        img1 = cv2.resize(img1, (width, height))
        img2 = cv2.resize(img2, (width, height))
    
    # 차영상 계산 (절대값 사용)
    difference = cv2.absdiff(img1, img2)
    
    # 차영상의 합 계산
    diff_sum = np.sum(difference)
    print(f"차영상의 픽셀 값 합계: {diff_sum}")
    
    # 채널별 합계 (BGR 각각)
    diff_sum_per_channel = np.sum(difference, axis=(0, 1))
    print(f"채널별 합계 (B, G, R): {diff_sum_per_channel}")
    
    # 결과 이미지 저장
    cv2.imwrite(output_path, difference)
    
    # 결과 이미지 표시 (선택사항)
    # cv2.imshow('Difference Image', difference)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    print(f"차영상이 {output_path}에 저장되었습니다.")
    return diff_sum  # 필요 시 합계를 반환

# 사용 예시
if __name__ == "__main__":
    # 이미지 파일 경로 지정
    image1_path = "C:/Users/coldbrew/VTON-project/tryon-images/result_1740402688.png"
    image2_path = "C:/Users/coldbrew/VTON-project/tryon-images/result_1740410723.png"
    
    # 함수 실행
    diff_sum = create_difference_image(image1_path, image2_path)
    if diff_sum is not None:
        print(f"반환된 차영상 합계: {diff_sum}")