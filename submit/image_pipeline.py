from lesson_functions import *
import pickle
import matplotlib.pyplot as plt
import cv2

#load model and parameters from the pickle
model = pickle.load(open('model.p', 'rb'))
svc = model['svc']

image = cv2.imread('./test_images/test3.jpg')

window_settings = [((64, 64), [400, 600], (0.5, 0.5)),
                   ((96, 96), [400, 600], (0.8, 0.8)),
                   ((128, 128), [400, 600], (0.9, 0.9)),
                   ((128, 128), [450, 600], (0.8, 0.8)),
                   ((256, 256), [400, 700], (0.5, 0.5))]

windows = get_sliding_windows(image_size=(720, 1280), settings=window_settings)

object_windows, labels, img_sliding_windows, img_object_windows, heatmap, img_labels = find_cars(image, windows, model)

#plot figure
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8,4))
ax1.imshow(cv2.cvtColor(img_sliding_windows, cv2.COLOR_BGR2RGB))
ax1.set_title('Sliding Windows', fontsize=12)

ax2.imshow(cv2.cvtColor(img_object_windows, cv2.COLOR_BGR2RGB))
ax2.set_title('Object Sliding Windows', fontsize=12)

ax3.imshow(heatmap, cmap='hot')
ax3.set_title('Heat Image')

ax4.imshow(cv2.cvtColor(img_labels, cv2.COLOR_BGR2RGB))
ax4.set_title('Detected Vehicles')

plt.show()
exit()