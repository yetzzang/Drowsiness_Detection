import cv2
import mediapipe as mp
import numpy as np
import time


class Drowsiness_Detection_mp():
    # function: initialize the class
    # input:
    #        drawing: list of drawing shapes want to be drawed, if None= nothing to draw
    #        ex) ['line', 'circle', 'bbox']
    # return:
    #        frame of converted color : self.frame
    #        list of face parts : self.parts
    #        list of all face-meshed coordinates : self.coordinates
    #        dictionary of facial parts index : self.part_idx
    #           ex) {'left_eye': [a1, a2, ...]}
    #        dictionary of facial parts coordinates : self.part_coord
    #           ex) {'left_eye': [(x1, y1), (x2, y2), ...]}
    #        self.detect_status: 0 if NOT detected / 1 if detected
    #        if drawing exists == cam/video while cap opened
    def __init__(self, drawing=None):
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

        self.frequency = 0

        self.result_list = []





        print(self.cap.get(cv2.CAP_PROP_FPS))

        self.prev_time = 0
        self.frame_counter = 0

        self.cap.set(cv2.CAP_PROP_FPS, 30)

        print(self.cap.get(cv2.CAP_PROP_FPS))

        while self.cap.isOpened():
            success, self.frame = self.cap.read()
            if not success:
                print('웹캠을 찾을 수 없습니다.')
                break

            self.find_coordinates()








            if drawing:
                self.draw(drawing)




            #print(self.mouth_height_predict_fps())
            print(self.mouth_prediction())
            prediction_result = self.mouth_prediction()


            # FPS 측정 및 표시
            self.frame_counter += 1
            current_time = time.time()
            elapsed_time = current_time - self.prev_time
            if elapsed_time >= 1:
                fps = self.frame_counter / elapsed_time
                print(f"FPS: {fps:.2f}")
                self.prev_time = current_time
                self.frame_counter = 0

            if prediction_result == 1:
                x, y, w, h = cv2.boundingRect(np.array(self.part_coord['entire_lip']))
                cv2.putText(self.frame, 'yawning', (x + w // 2 - 40, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),
                            2)
            elif prediction_result ==2:
                x, y, w, h = cv2.boundingRect(np.array(self.part_coord['entire_lip']))
                cv2.putText(self.frame, 'sleeping', (x + w // 2 - 40, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),
                            2)

            cv2.imshow('frm', self.frame)
            if cv2.waitKey(1) == 27:
                self.cap.release()
                break

    # find face part coordinates
    # return:
    # if 'cam' & IndexError: print('얼굴 인식을 할 수 없습니다')
    def find_coordinates(self):
        self.part_coord = {self.parts[0]: [],
                           self.parts[1]: [],
                           self.parts[2]: [],
                           self.parts[3]: [],
                           self.parts[4]: [],
                           self.parts[5]: []}

        self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        image_height, image_width, _ = self.frame.shape

        # landmark 추출
        self.result = self.face_mesh.process(self.frame)

        # face-mesh 전체 좌표 따기
        self.coordinates = []
        if self.result.multi_face_landmarks:
            for face_landmarks in self.result.multi_face_landmarks:
                for idx, landmark in enumerate(face_landmarks.landmark):
                    x = int(landmark.x * image_width)
                    y = int(landmark.y * image_height)
                    self.coordinates.append([x, y])

        # 얼굴 부위별 좌표 따기
        for part in self.parts:
            for idx in self.part_idx[part]:
                try:
                    self.part_coord[part].append(self.coordinates[idx])
                except IndexError:
                    print('얼굴을 인식하지 못 했습니다')
                except TypeError:
                    print('얼굴을 인식하지 못 했습니다')

    # draw circle of face part landmarks coordinates
    def draw_circle(self):
        for part in self.parts:
            for idx in self.part_idx[part]:
                cv2.circle(self.frame, (self.coordinates[idx][0], self.coordinates[idx][1]), 2, (0, 255, 0), -1)

    # draw bounding box of face part landmarks coordinates
    def draw_bbox(self):
        for part in ['left_eye', 'right_eye', 'entire_lip']:
            x, y, w, h = cv2.boundingRect(np.array(self.part_coord[part]))
            cv2.rectangle(self.frame, (x - 5, y - 5), (x + w + 5, y + h + 5), (255, 0, 0), 2)
        x, y, w, h = cv2.boundingRect(np.array(self.part_coord['face_contour']))
        cv2.rectangle(self.frame, (x - 5, y - 5), (x + w + 5, y + h + 5), (255, 0, 0), 2)
        cv2.putText(self.frame, 'face detected!', (x - 5, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # draw line of face parts
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

    # draw selection of type
    # input: list of types
    # ex) ['circle', 'line']
    def draw(self, kinds):
        for kind in kinds:
            if kind == 'circle':
                self.draw_circle()
            elif kind == 'line':
                self.draw_line()
            else:
                self.draw_bbox()

    # input: self
    # function: predict if frame is open or close by eye slope
    # output: return 0 if frame is open
    #         return 1 if frame is closed
    def mouth_height_predict_fps(self):
        v_a = self.part_coord['bottom_lip'][3][1] - self.part_coord['top_lip'][3][1]
        v_b = self.part_coord['bottom_lip'][4][1] - self.part_coord['top_lip'][4][1]
        v_c = self.part_coord['bottom_lip'][5][1] - self.part_coord['top_lip'][5][1]

        coord_vertical_mean = (v_a + v_b + v_c) / 3

        mouth_threshold = 20

        if coord_vertical_mean >= mouth_threshold:
            self.frequency += 1
            if (self.frequency > 70 and self.frequency < 300) :
                return 1
            elif self.frequency >= 300:
                return 2

            else:
                return 0
        else:  # coord_vertical_mean < mouth_threshold
            if self.frequency < 70:
                return 0  # no wawn
            else:  # freqeuncy >= 40
                self.frequency = self.frequency - 20
                if self.frequency < 0:
                    self.frequency = 0
                return 1  # wawn



    def mouth_prediction(self):
        self.result_list.append(self.mouth_height_predict_fps())

        if len(self.result_list) > 10:
            self.result_list.pop(0)

        #return self.result_list
        if self.result_list.count(1) >= 5:
        # case1 : yawning
            return 1
        elif self.result_list.count(2) >= 6:
        # case2 : sleeping
            return 2
        else:
        # case3 : awake
            return 0 






# 테스트 코드
cam_sample = Drowsiness_Detection_mp(drawing=['line', 'circle', 'bbox'])
