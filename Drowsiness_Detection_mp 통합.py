import cv2
import mediapipe as mp
import numpy as np
import time
from statistics import mode, mean
import winsound as sd
import threading
import pygame


class Drowsiness_Detection_mp():
    # function: initialize the class
    def __init__(self):
        # 얼굴 검출 객체
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            refine_landmarks=True,
            static_image_mode=True,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # fash mesh 그리기 객체
        self.mp_drawing = mp.solutions.drawing_utils
        self.drawing_spec = self.mp_drawing.DrawingSpec(thickness=2, circle_radius=1, color=(0, 0, 255))

        # 인식 status
        self.detection_status = 0

        # 얼굴 부위 리스트화
        self.parts = ['left_eye', 'right_eye', 'top_lip', 'bottom_lip', 'entire_lip', 'face_contour']

        # 얼굴 부위별 인덱스값 딕셔너리화; values는 list
        self.part_idx = {self.parts[0]: [133, 173, 157, 158, 159, 160, 161, 246, 33, 7, 163, 144, 145, 153, 154, 155],
                         self.parts[1]: [362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381,
                                         382],
                         self.parts[2]: [78, 191, 80, 82, 13, 312, 310, 415, 308],
                         self.parts[3]: [78, 95, 88, 87, 14, 317, 318, 324, 308],
                         self.parts[4]: [0, 37, 40, 61, 91, 84, 17, 314, 321, 291, 270, 267],
                         self.parts[5]: [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378,
                                         400, 377,
                                         152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67,
                                         109, 10]}

        # 기준값 찾기 위해 declare
        self.standard_exemption_left_count = 0
        self.standard_exemption_right_count = 0
        self.eye_slope_threshold = 0
        self.eye_ratio_threshold = 0
        self.standards = ['eye_slope_left', 'eye_slope_right', 'eye_ratio_left', 'eye_ratio_right']
        self.standard_list = {self.standards[0]: [],
                              self.standards[1]: [],
                              self.standards[2]: [],
                              self.standards[3]: []}
        self.standard_list2 = {self.standards[0]: [],
                               self.standards[1]: [],
                               self.standards[2]: [],
                               self.standards[3]: []}

        # eye_predict 함수에 쓰일 것 declare
        self.eye_fps_predictions = []
        self.eye_fps_mode = 0
        eye_queue_len = 20
        self.eye_queue = [0 for x in range(eye_queue_len)]

        # mouth predict 관련 변수
        self.frequency = 0
        self.frame_count = 0  # 프레임 카운트 초기화

        # 입 벌림 예측을 위한 큐 및 threshold 설정
        self.mouth_open_queue = [0 for _ in range(100)]
        self.mouth_open_duration = 0
        self.mouth_open_threshold = 10

    # input: self
    # function: process the cam using MediaPipe facemesh
    def process_cam(self):
        pygame.mixer.init()
        self.alert_sound = pygame.mixer.Sound('siren_sound.wav')
        start_time = time.time()
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        while self.cap.isOpened():
            current_time = time.time()
            elapsed_time = current_time - start_time

            success, self.frame = self.cap.read()
            # print(self.cap.get(cv2.CAP_PROP_FPS))
            if not success:
                print('웹캠을 찾을 수 없습니다.')
                break

            self.find_coordinates()

            face_detection_time = 10
            if elapsed_time <= face_detection_time:
                if not self.find_standard():
                    self.put_text_if_not_detected('noCenter')
                if self.detection_status != -1:
                    self.find_standard()
                    self.draw_bbox(mode='processing_on')
                else:
                    self.put_text_if_not_detected('noDetection')
            elif face_detection_time < elapsed_time < face_detection_time + 1:
                if self.detection_status != -1:
                    self.find_threshold()
                    self.draw_bbox(mode='processing_on')
                else:
                    self.put_text_if_not_detected('noDetection')
            else:
                if self.detection_status != -1:
                    self.predict()
                else:
                    self.put_text_if_not_detected('noDetection')

            cv2.imshow('frm', self.frame)
            if cv2.waitKey(1) == 27:
                self.cap.release()
                break

    # find face part coordinates
    # return:
    def find_coordinates(self):
        self.part_coord = {self.parts[0]: [],
                           self.parts[1]: [],
                           self.parts[2]: [],
                           self.parts[3]: [],
                           self.parts[4]: [],
                           self.parts[5]: []}

        self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        self.frame_height, self.frame_width, _ = self.frame.shape

        # landmark 추출
        self.result = self.face_mesh.process(self.frame)

        # face-mesh 전체 좌표 따기
        self.coordinates = []
        if self.result.multi_face_landmarks:
            for face_landmarks in self.result.multi_face_landmarks:
                for idx, landmark in enumerate(face_landmarks.landmark):
                    x = int(landmark.x * self.frame_width)
                    y = int(landmark.y * self.frame_height)
                    self.coordinates.append([x, y])

        # 얼굴 부위별 좌표 따기
        for part in self.parts:
            for idx in self.part_idx[part]:
                try:
                    self.part_coord[part].append(self.coordinates[idx])
                    self.detection_status = 0
                except IndexError:
                    self.detection_status = -1
                except TypeError:
                    self.detection_status = -1

    # function: draw circle of face part landmarks coordinates
    def draw_circle(self):
        for part in self.parts:
            for idx in self.part_idx[part]:
                cv2.circle(self.frame, (self.coordinates[idx][0], self.coordinates[idx][1]), 2, (0, 255, 0), -1)

    # function: draw line of face parts
    def draw_line(self):
        landmark_specs = [self.mp_face_mesh.FACEMESH_RIGHT_EYE,
                          self.mp_face_mesh.FACEMESH_LEFT_EYE,
                          self.mp_face_mesh.FACEMESH_LIPS,
                          self.mp_face_mesh.FACEMESH_FACE_OVAL]
        for landmark_spec in landmark_specs:
            for single_face_landmarks in self.result.multi_face_landmarks:
                self.mp_drawing.draw_landmarks(
                    image=self.frame,
                    landmark_list=single_face_landmarks,
                    connections=landmark_spec,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.drawing_spec
                )

    # function: draw basics of bbox
    def draw_bbox_basic(self, color):
        for part in ['left_eye', 'right_eye', 'entire_lip']:
            x, y, w, h = cv2.boundingRect(np.array(self.part_coord[part]))
            cv2.rectangle(self.frame, (x - 5, y - 5), (x + w + 5, y + h + 5), color, 2)
        x, y, w, h = cv2.boundingRect(np.array(self.part_coord['face_contour']))
        cv2.rectangle(self.frame, (x - 5, y - 5), (x + w + 5, y + h + 5), color, 2)

    # function: draw bounding box of face part landmarks coordinates
    def draw_bbox(self, mode, alert=None):
        global color
        color = (255, 0, 0)
        if alert == 2 or alert == 1:
            color = (0, 0, 255)
        x, y, w, h = cv2.boundingRect(np.array(self.part_coord['face_contour']))
        if mode == 'processing_on':
            cv2.putText(self.frame, 'Now detecting your face', (x - 5, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 2)
            self.draw_bbox_basic(color)
        elif mode == 'processing_off' and alert == 2:
            cv2.putText(self.frame, 'ALERT: WAKE UP!', (x - 5, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            self.draw_bbox_basic(color)
        elif mode == 'processing_off' and alert == 1:
            cv2.putText(self.frame, 'You are DROWSY now. Be careful!', (x - 5, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        color, 2)
            self.draw_bbox_basic(color)
        elif mode == 'processing_off':
            cv2.putText(self.frame, 'Drowsiness Detection: ON', (x - 5, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1)

    def put_text_if_not_detected(self, mode):
        height, width, _ = self.frame.shape
        if mode == 'noDetection':
            cv2.putText(self.frame, 'CANNOT detect your face', (int(width / 2), int(height / 2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        elif mode == 'noCenter':
            cv2.putText(self.frame, 'Face misalignment detected. Please adjust to center and face the screen straight',
                        (int(width / 2), int(height / 2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # function: 경고음 송출
    def beepsound(self):
        fr = 2000  # range : 37 ~ 32767
        du = 1000  # 1000 ms ==1second
        sd.Beep(fr, du)  # winsound.Beep(frequency, duration)

    # input: self
    # function: proceed algorithm by using eye slope
    # output: return summary of sum of weighted eyes slopes difference
    def cal_eye_slope(self):
        sides = ['left', 'right']
        eyes_coord = {sides[0]: self.part_coord['left_eye'],
                      sides[1]: self.part_coord['right_eye']}

        # calculate slope of consecutive coordinates
        # append them to the slope_list
        slope_list = {sides[0]: [],
                      sides[1]: []}
        for side in sides:
            for i, j in zip(eyes_coord[side][:-1], eyes_coord[side][1:]):
                try:
                    # print(j[1], i[1], j[0], i[0])
                    slope_list[side].append(abs((j[1] - i[1]) / (j[0] - i[0])))
                except ZeroDivisionError:
                    slope_list[side].append(abs(j[1] - i[1]))
            try:
                slope_list[side].append(abs((eyes_coord[side][0][1] - eyes_coord[side][-1][1])
                                            / (eyes_coord[side][0][0] - eyes_coord[side][-1][0])))
            except ZeroDivisionError:
                slope_list[side].append(abs(eyes_coord[side][0][1] - eyes_coord[side][-1][1]))

        # calculate square difference with weights
        sq_list = {sides[0]: [],
                   sides[1]: []}
        summary = {}
        for side in sides:
            for idx, i, j in zip(range(len(slope_list[side][:-1])), slope_list[side][:-1], slope_list[side][1:]):
                if idx == 0 or idx == 1 or idx == 6 or idx == 7:
                    sq_list[side].append(abs(i ** 2 - j ** 2) * 1.5)
                elif idx == 2 or idx == 5:
                    sq_list[side].append(abs(i ** 2 - j ** 2) * 1.2)
                elif idx == 11 or idx == 12:
                    sq_list[side].append(abs(i ** 2 - j ** 2) * 0.7)
                else:
                    sq_list[side].append(abs(i ** 2 - j ** 2))
            if side == sides[0]:
                summary[sides[0]] = sum(sq_list[side])
            elif side == sides[1]:
                summary[sides[1]] = sum(sq_list[side])
        return summary

    # input: self
    # function: predict if frame is open or close by eye slope
    # output: return 0 if frame is open
    #         return 1 if frame is closed
    def eye_slope_predict_fps(self):
        eye_slope_summary = self.cal_eye_slope()
        threshold = self.eye_slope_threshold
        classes = {'open': 0, 'close': 1}
        left_eye_sum, right_eye_sum = [float(i) for i in eye_slope_summary.values()]
        if left_eye_sum > threshold and right_eye_sum > threshold:
            return 0
        elif left_eye_sum < threshold and right_eye_sum < threshold:
            return 1
        else:
            if ((left_eye_sum + right_eye_sum) / 2) > threshold:
                return 0
            elif ((left_eye_sum + right_eye_sum) / 2) < threshold:
                return 1

    # input: self
    # function: calculate eye ratio of both eyes
    # output: summary of eye ratio in dict
    def cal_eye_ratio(self):
        # normal_ratio는 앞 관찰을 통해 관찰자의 평소 ratio를 의미하며 그 크기의 평소 1/8에서 1/10 수준까지 작아
        # 질 경우를 수면 생태로 규정한다.
        x1 = self.part_coord['left_eye'][0]
        x2 = self.part_coord['left_eye'][8]
        w = (x1[0] - x2[0]) ** 2 + (x1[1] - x2[1]) ** 2
        w = w ** 0.5
        x3 = self.part_coord['left_eye'][4]
        x4 = self.part_coord['left_eye'][12]
        h = (x3[0] - x4[0]) ** 2 + (x3[1] - x4[1]) ** 2
        h = h ** 0.5
        left_ratio = h / w

        x1 = self.part_coord['right_eye'][0]
        x2 = self.part_coord['right_eye'][8]
        w = (x1[0] - x2[0]) ** 2 + (x1[1] - x2[1]) ** 2
        w = w ** 0.5
        x3 = self.part_coord['right_eye'][4]
        x4 = self.part_coord['right_eye'][12]
        h = (x3[0] - x4[0]) ** 2 + (x3[1] - x4[1]) ** 2
        h = h ** 0.5
        right_ratio = h / w

        summary = {'left': left_ratio, 'right': right_ratio}
        return summary

    # input: self
    # function: predict if frame is open or close by eye area ratio
    # output: return 0 if frame is open
    #         return 1 if frame is closed
    def eye_ratio_predict_fps(self):
        eye_ratio_summary = self.cal_eye_ratio()
        threshold = self.eye_ratio_threshold
        if eye_ratio_summary['left'] < threshold and eye_ratio_summary['right'] < threshold:
            return 1
        else:
            return 0

    # input: self
    # output: return mean of eye_slope_predict_fps and eye_ratio_predict_fps
    def eye_predict_fps(self):
        a = self.eye_slope_predict_fps()
        b = self.eye_ratio_predict_fps()
        avg = (a + b) / 2
        return avg

    # input: self
    # function: predict whether driver's eye is drowsy or not
    # output: 0 if eyes are open, 1 if eyes are half-closed, 2 if eyes are definitely closed
    def eye_predict(self):
        threshold_2 = 10
        threshold_1 = 10
        self.eye_fps_predictions.append(self.eye_predict_fps())
        if len(self.eye_fps_predictions) == 5:
            self.eye_fps_mode = mode(self.eye_fps_predictions)
            self.eye_fps_predictions = []
            self.eye_queue.append(self.eye_fps_mode)
            del self.eye_queue[0]
            if self.eye_queue.count(1) > threshold_2:
                return 2
            elif self.eye_queue.count(0.5) > threshold_1:
                return 1
            else:
                return 0
        else:
            return 0

    # input: self
    # function: predict if frame is open or close by mouth height
    # output: return 0 if frame is open
    #         return 1 if frame is closed
    def cal_mouth_angle(self):
        summary = {}
        for i in range(4):
            point1 = self.coordinates[self.part_idx['bottom_lip'][0]]
            point2 = self.coordinates[self.part_idx['top_lip'][i + 1]]
            point3 = self.coordinates[self.part_idx['bottom_lip'][i + 1]]

            angle_rad = np.arctan2(point3[1] - point1[1], point3[0] - point1[0]) - \
                        np.arctan2(point2[1] - point1[1], point2[0] - point1[0])
            angle_deg = np.degrees(angle_rad)
            summary[i + 1] = angle_deg
        return summary

    def mouth_angle_predict_fps(self):
        mouth_angle_summary = self.cal_mouth_angle()

        angle_threshold = 94

        for angle in mouth_angle_summary.values():
            if angle > angle_threshold:
                return 1
        return 0

    def mouth_height_predict_fps(self):
        v_a = self.part_coord['bottom_lip'][3][1] - self.part_coord['top_lip'][3][1]
        v_b = self.part_coord['bottom_lip'][4][1] - self.part_coord['top_lip'][4][1]
        v_c = self.part_coord['bottom_lip'][5][1] - self.part_coord['top_lip'][5][1]

        coord_vertical_mean = (v_a + v_b + v_c) / 3

        mouth_threshold = 16

        if coord_vertical_mean >= mouth_threshold:
            self.frequency += 1
            if self.frequency > 50:
                return 1
            else:
                return 0
        else:  # coord_vertical_mean < mouth_threshold
            if self.frequency < 50:
                return 0  # no wawn
            else:  # freqeuncy >= 40
                self.frequency = self.frequency - 20
                if self.frequency < 0:
                    self.frequency = 0
                return 1  # wawn

    def update_mouth_open_duration(self):
        if self.mouth_height_predict_fps() == 1 or self.mouth_angle_predict_fps() == 1:
            self.mouth_open_duration += 1
        else:
            self.mouth_open_duration = 0

    def yawn_predict_fps(self):
        # 입이 열린 상태가 일정 시간 이상 지속되면 하품으로 판단
        self.update_mouth_open_duration()

        if self.mouth_open_duration >= self.mouth_open_threshold:
            return 1  # 하품
        else:
            return 0  # 깨어있음

    def mouth_predict(self):  # predict_func(self)에서 이름 변경

        if self.frame_count < 10:
            self.frame_count += 1
            return 0
        else:
            # 큐에 현재 프레임의 예측값 추가
            prediction = self.yawn_predict_fps()

            self.mouth_open_queue.append(prediction)

            del self.mouth_open_queue[0]

            if self.mouth_open_queue.count(1) <= 5:
                # (mode(self.mouth_open_queue) == 0):

                return 0
            elif self.mouth_open_queue.count(1) >= 5:
                # (mode(self.mouth_open_queue) == 1):

                if all(value == 1 for value in self.mouth_open_queue):
                    return 2
                else:
                    return 1

    # input: self
    # function: find the standard value for each algorithm
    def find_standard(self):
        restart_num = 10
        frame_count = 15

        if self.standard_exemption_left_count <= 10 or self.standard_exemption_left_count <= 10:
            eye_slope = self.cal_eye_slope()
            eye_slope_left, eye_slope_right = [float(i) for i in eye_slope.values()]
            eye_ratio = self.cal_eye_ratio()
            eye_ratio_left, eye_ratio_right = [float(i) for i in eye_ratio.values()]
            self.standard_list[self.standards[0]].append(eye_slope_left)
            self.standard_list[self.standards[1]].append(eye_slope_right)
            self.standard_list[self.standards[2]].append(eye_ratio_left)
            self.standard_list[self.standards[3]].append(eye_ratio_right)
            if len(self.standard_list[self.standards[-1]]) == frame_count:
                for i in range(len(self.standard_list)):
                    avg = mean(self.standard_list[self.standards[i]])
                    if i == 0:
                        if avg > restart_num:
                            self.standard_exemption_left_count += 1
                    elif i == 1:
                        if avg > restart_num:
                            self.standard_exemption_left_count += 1
                    self.standard_list[self.standards[i]] = []
                    self.standard_list2[self.standards[i]].append(avg)
                    print(self.standard_exemption_left_count, self.standard_exemption_right_count)
            return True
        else:
            return False

    # input: self
    # function: find the threshold of eye_slope, eye_ratio
    # output: return thresholds
    def find_threshold(self):
        eye_slope_list = self.standard_list2['eye_slope_left'] + self.standard_list2['eye_slope_right']
        eye_ratio_list = self.standard_list2['eye_ratio_left'] + self.standard_list2['eye_ratio_right']
        # 재확인 필요
        self.eye_slope_threshold = mean(eye_slope_list) - 2
        self.eye_ratio_threshold = mean(eye_ratio_list) / 3

    # input: self
    # function: print if driver is drowsy
    # 소리 재생을 위한 스레드 함수
    def play_sound_thread(self):
        self.alert_sound.play()

    # 소리 재생 함수
    def play_sound(self):
        # 소리 재생을 위한 스레드 생성 및 시작
        thread = threading.Thread(target=self.play_sound_thread)
        thread.start()

    # 소리 정지 함수
    def stop_sound(self):
        self.alert_sound.stop()

    def predict(self):
        eye_prediction = self.eye_predict()
        mouth_prediction = self.mouth_predict()
        if (eye_prediction + mouth_prediction) >= 2:
            self.draw_bbox(mode='processing_off', alert=eye_prediction)
            self.play_sound()
        elif (eye_prediction + mouth_prediction) == 1:
            self.draw_bbox(mode='processing_off', alert=eye_prediction)
        else:
            self.draw_bbox(mode='processing_off')
            self.stop_sound()


# 테스트 코드
cam_sample = Drowsiness_Detection_mp()
cam_sample.process_cam()
