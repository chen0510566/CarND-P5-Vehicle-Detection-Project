from lesson_functions import *
import pickle

# load model and parameters from the pickle
model = pickle.load(open('model.p', 'rb'))
svc = model['svc']

image = cv2.imread('./test_images/test3.jpg')

window_settings = [((64, 64), [400, 600], (0.5, 0.5)),
                   ((96, 96), [400, 600], (0.8, 0.8)),
                   ((128, 128), [400, 600], (0.9, 0.9)),
                   ((128, 128), [450, 600], (0.8, 0.8)),
                   ((256, 256), [400, 700], (0.5, 0.5))]

candidate_windows = get_sliding_windows(image_size=(720, 1280), settings=window_settings)

cap = cv2.VideoCapture()

all_img = np.zeros((360, 960, 3), np.uint8)

video_writer = cv2.VideoWriter()

video_writer.open('./project_video_result.mp4', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps=cap.get(5),
                               frameSize=(960, 360))

object_windows_list = []
rolling_window = 20
heat_threshold = 15

while (cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        break

    object_windows, labels, img_sliding_windows, img_object_windows, heatmap, img_labels = find_cars(frame,
                                                                                                     candidate_windows,
                                                                                                     model)

    # record last detection;
    object_windows_list.append(object_windows)
    if len(object_windows_list) > rolling_window:
        object_windows_list.pop(0)  # pop the oldest windows

    # add all the detected object winds;
    heat = np.zeros_like(frame[:, :, 0]).astype(np.float)
    for windows in object_windows_list:
        heat = add_heat(heat, windows)

    # apply threshold, rescale heatmap to visualize the result;
    heat = apply_threshold(heat, heat_threshold)
    heatmap = np.clip(heat, 0, 255)
    if np.max(heatmap) > 0.0:
        factor = 255.0 / np.max(heatmap)
        heatmap *= factor

    labels = label(heatmap)
    img_labels = draw_labeled_boxes(frame, labels)

    # copy images together
    all_img[0:360, 0:640] = cv2.resize(img_labels, (640, 360))
    all_img[0:180, 640:960] = cv2.resize(img_object_windows, (320, 180))
    all_img[180:360, 640:960] = cv2.resize(np.dstack((np.zeros_like(heatmap), np.zeros_like(heatmap), heatmap)),
                                           (320, 180))

    video_writer.write(all_img)
    cv2.imshow('frame', all_img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
video_writer.release()
cv2.destroyAllWindows()
