import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

# 변수 초기화
tracking = False
bbox = None
tracker = cv2.legacy.TrackerCSRT_create()  # CSRT 추적기 사용
points = []  # 두 점 저장
origin_ = None  # 원점 저장
cm_per_pixel = 0  # 픽셀 단위의 1cm

# 클릭 이벤트 처리 함수
def click_event(event, x, y, flags, param):
    global tracking, bbox, points, cm_per_pixel, origin_
    if event == cv2.EVENT_LBUTTONDOWN:
        if origin_ is None:
            origin_ = (x, y)  # 첫 번째 클릭으로 원점 지정
        elif len(points) < 2:
            points.append((x, y))
            if len(points) == 2:
                # 두 점 사이의 거리 계산
                distance = np.linalg.norm(np.array(points[0]) - np.array(points[1]))
                cm_per_pixel = 1 / distance  # 1cm를 픽셀 단위로 환산
        else:
            bbox = (x, y, 100, 100)
            tracking = True

# 비디오 파일 열기
cap = cv2.VideoCapture('video04.mp4')

# 비디오 정보 가져오기
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
delay = int(1000 / fps) * 1  # 10분의 1 속도로 재생하기 위한 지연 시간 (밀리초)
duration = frame_count / fps

# 1초 후의 프레임으로 이동
target_time = 2  # 1초
cap.set(cv2.CAP_PROP_POS_MSEC, target_time * 1000)

# 1초 후의 프레임 읽기
ret, first_frame = cap.read()
if not ret:
    print("비디오를 열 수 없거나 1초 후의 프레임을 가져올 수 없습니다.")
    cap.release()
    exit()

# 창 생성 및 클릭 이벤트 설정
cv2.namedWindow('Frame')
cv2.setMouseCallback('Frame', click_event)

# 첫 번째 프레임에서 두 점 선택
while True:
    frame_copy = first_frame.copy()
    if origin_:
        cv2.circle(frame_copy, origin_, 5, (255, 0, 0), -1)  # 원점 표시
    for point in points:
        cv2.circle(frame_copy, point, 5, (0, 255, 0), -1)
    if len(points) == 2:
        cv2.line(frame_copy, points[0], points[1], (0, 255, 0), 2)
        
    cv2.imshow('Frame', frame_copy)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or tracking:
        break

if tracking:
    tracker.init(first_frame, bbox)


# 객체 중심점 좌표 배열
center_points = []
timestamps = []
start_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    if tracking:
        ret, bbox = tracker.update(frame)
        if ret:
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
            
            # 중심점 계산 및 저장
            center_x = int(bbox[0] + bbox[2] / 2)
            center_y = int(bbox[1] + bbox[3] / 2)
            center_points.append((center_x, center_y))
            timestamps.append(time.time() - start_time)
            
            # 궤적 그리기
            for i in range(1, len(center_points)):
                cv2.line(frame, center_points[i - 1], center_points[i], (0, 255, 0), 2)

    cv2.imshow('Frame', frame)

    key = cv2.waitKey(delay) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
timestamps = np.array(timestamps)
# 시간에 따른 객체 중심점 좌표 그래프 출력
center_points = np.array(center_points)


# 픽셀 좌표를 cm 좌표로 변환
print(cm_per_pixel)
# 원점 기준으로 좌표 변환
center_points = center_points - np.array(origin_)
center_points_cm = center_points * cm_per_pixel

plt.plot(center_points_cm[:, 0], center_points_cm[:, 1], marker='o')
plt.title('Object Center Points Over Time')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.gca().invert_yaxis()  # 이미지 좌표계를 맞추기 위해 Y축 반전
plt.show()

# 시간에 따른 X 좌표 그래프
plt.figure(figsize=(10, 5))
plt.plot(timestamps, center_points_cm[:, 0], marker='o')
plt.title('X Coordinate Over Time')
plt.xlabel('Time (s)')
plt.ylabel('X Coordinate')
plt.show()

# 시간에 따른 Y 좌표 그래프
plt.figure(figsize=(10, 5))
plt.plot(timestamps, center_points_cm[:, 1], marker='o')
plt.title('Y Coordinate Over Time')
plt.xlabel('Time (s)')
plt.ylabel('Y Coordinate')
plt.show()