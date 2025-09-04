import cv2
import os
import glob
import dlib
import numpy as np
from pathlib import Path

def save_high_quality_frames(video_path, quality_threshold=100, save_folder='frames'):
    # 비디오 파일을 로드합니다.
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Cannot open video.")
        return

    # frames 폴더가 없으면 생성합니다.
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 프레임의 화질을 평가합니다. 라플라시안의 분산을 사용합니다.
        variance_of_laplacian = cv2.Laplacian(frame, cv2.CV_64F).var()

        # 화질이 임계값 이상인 경우에만 이미지를 저장합니다.
        if variance_of_laplacian > quality_threshold:
            cv2.imwrite(f"{save_folder}/frame_{frame_id:05d}.jpg", frame)
            print(f"Frame {frame_id:05d} saved, quality: {variance_of_laplacian:.2f}")
        
        frame_id += 1

    cap.release()
    print("Done!")






def find_and_save_edges(input_folder, output_folder):
    # 폴더 경로를 Path 객체로 변환
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)

    # 출력 폴더가 없으면 생성
    output_folder.mkdir(exist_ok=True)

    # 입력 폴더 내 모든 이미지 파일을 반복 처리
    for image_path in input_folder.glob('*.jpg'):
        # 이미지 읽기
        image = cv2.imread(str(image_path))

        # 엣지 검출 (Canny 엣지 검출기 사용)
        edges = cv2.Canny(image, 100, 200)

        # 결과 이미지 파일 경로 생성
        output_image_path = output_folder / image_path.name

        # 결과 이미지 저장
        cv2.imwrite(str(output_image_path), edges)

        print(f"Processed {image_path.name}, saved to {output_image_path}")



# def visualize_and_save_sift_features(input_folder, output_folder):
#     # 입력 및 출력 폴더 경로 검증
#     input_folder = Path(input_folder)
#     output_folder = Path(output_folder)
#     output_folder.mkdir(exist_ok=True)  # 출력 폴더가 없으면 생성

#     # SIFT 피쳐 검출기 초기화
#     sift = cv2.SIFT_create()

#     # 입력 폴더 내의 모든 이미지에 대해 반복
#     for image_path in input_folder.glob('*.*'):  # 모든 파일 포맷을 고려
#         # 이미지 읽기
#         image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
#         if image is None:
#             continue  # 이미지 파일이 아니라면 건너뛰기

#         # SIFT 특징 추출
#         keypoints, _ = sift.detectAndCompute(image, None)

#         # 키 포인트 시각화
#         sift_image = cv2.drawKeypoints(image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

#         # 시각화된 이미지 저장
#         output_image_path = output_folder / f'sift_{image_path.name}'
#         cv2.imwrite(str(output_image_path), sift_image)

#         print(f"Processed {image_path.name}, saved SIFT features to {output_image_path}")

# 함수 실행
#visualize_and_save_sift_features('frames', 'sift')


def crop_center_and_save(input_folder, output_folder, height=400):
    # 폴더 경로를 Path 객체로 변환
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)

    # 출력 폴더가 없으면 생성
    output_folder.mkdir(exist_ok=True)

    # 입력 폴더 내 모든 이미지 파일을 반복 처리
    for image_path in input_folder.glob('*.jpg'):
        # 이미지 읽기
        image = cv2.imread(str(image_path))
        if image is None:
            continue  # 이미지 파일이 아니라면 건너뛰기

        # 이미지 중앙을 기준으로 상하 300픽셀을 남기고 잘라내기
        center_y = image.shape[0] // 2
        cropped_image = image[(center_y - height):(center_y + height), :]

        # 결과 이미지 파일 경로 생성
        output_image_path = output_folder / f"cropped_{image_path.name}"

        # 결과 이미지 저장
        cv2.imwrite(str(output_image_path), cropped_image)

        print(f"Processed {image_path.name}, saved to {output_image_path}")





def align_images(input_folder, output_folder):
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(exist_ok=True)  # 출력 폴더 생성

    # SIFT 초기화
    sift = cv2.SIFT_create()

    # 첫 번째 이미지 로드 및 SIFT 특징 추출 (그레이스케일)
    base_image_path = next(input_folder.glob('*.jpg'))
    base_image_color = cv2.imread(str(base_image_path))  # 컬러 이미지 로드
    base_image_gray = cv2.cvtColor(base_image_color, cv2.COLOR_BGR2GRAY)  # 그레이스케일 변환
    base_keypoints, base_descriptors = sift.detectAndCompute(base_image_gray, None)

    image_files = list(input_folder.glob('*.jpg'))
    # 이미지 목록을 첫 이미지부터 순서대로 처리
    for image_path in image_files:
        # 컬러 이미지 로드 및 그레이스케일 변환
        image_color = cv2.imread(str(image_path))
        image_gray = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)

        # SIFT 특징 추출
        keypoints, descriptors = sift.detectAndCompute(image_gray, None)

        # FLANN 매처 설정
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        # 매칭 및 좋은 매칭 필터링
        matches = flann.knnMatch(descriptors, base_descriptors, k=2)
        good_matches = [m for m, n in matches if m.distance < 0.7*n.distance]
        src_pts = np.float32([keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([base_keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # 호모그래피 계산
        M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        # 컬러 이미지에 호모그래피 적용
        h, w, _ = base_image_color.shape
        aligned_image_color = cv2.warpPerspective(image_color, M, (w, h))

        # 변환된 컬러 이미지 저장
        output_image_path = output_folder / f"aligned_{image_path.name}"
        cv2.imwrite(str(output_image_path), aligned_image_color)

        print(f"Image {image_path.name} aligned and saved to {output_image_path}")



def generate_max_mean_min_images(input_folder, output_folder):
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(exist_ok=True)

    # Initialize lists to hold the image data
    image_data = []

    # Read images and append the image data to the list
    for image_path in sorted(input_folder.glob('*.jpg')):
        img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if img is not None:
            image_data.append(img)
        else:
            print(f"Failed to load image: {image_path}")

    # Stack images along a new dimension
    stack = np.stack(image_data, axis=3)

    # Calculate max, mean, and min images
    max_img = np.max(stack, axis=3)
    mean_img = np.mean(stack, axis=3).astype(np.uint8)
    min_img = np.min(stack, axis=3)

    # Save the generated images
    cv2.imwrite(str(output_folder / 'max_image.jpg'), max_img)
    cv2.imwrite(str(output_folder / 'mean_image.jpg'), mean_img)
    cv2.imwrite(str(output_folder / 'min_image.jpg'), min_img)
    cv2.imwrite(str(output_folder / 'max_min_image.jpg'), max_img-min_img)
    cv2.imwrite(str(output_folder / 'max_mean_image.jpg'), max_img-mean_img)
    cv2.imwrite(str(output_folder / 'mean_min_image.jpg'), mean_img-min_img)
    print(f"Saved max, mean, and min images in {str(output_folder)}")






##################################
# extract frames
save_high_quality_frames('print_video.mp4', quality_threshold=100, save_folder='frames')


# edge
find_and_save_edges('./frames', './edge')

# crop
crop_center_and_save('frames', 'cropped_images')

# align
align_images('cropped_images', 'aligned_images')

# output
generate_max_mean_min_images('aligned_images', 'output')
