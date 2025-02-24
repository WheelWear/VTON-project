import cv2
import numpy as np

def create_difference_image(image_path1, image_path2, output_path='difference_image.jpg'):
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
    
    # 결과 이미지 저장
    cv2.imwrite(output_path, difference)
    
    # 결과 이미지 표시 (선택사항)
    # cv2.imshow('Difference Image', difference)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    print(f"차영상이 {output_path}에 저장되었습니다.")

# 사용 예시
if __name__ == "__main__":
    # 이미지 파일 경로 지정
    image1_path = "C:/Users/coldbrew/VTON-project/tryon-images/result_1740402371.png"  # 첫 번째 이미지 경로
    image2_path = "C:/Users/coldbrew/VTON-project/tryon-images/result_1740402688.png"  # 두 번째 이미지 경로
    
    # 함수 실행
    create_difference_image(image1_path, image2_path)