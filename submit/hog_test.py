from lesson_functions import *
from glob import glob
import matplotlib.pyplot as plt

color_space = 'HSV'

# load all the training images
car_paths = glob('./vehicles/*/*.png')
noncar_paths = glob('./non-vehicles/*/*.png')
num_samples = np.min((len(car_paths), len(noncar_paths)))

idx = np.random.randint(0, num_samples)

fig, plts = plt.subplots(4, 4)

image = cv2.imread(car_paths[idx])

param_spatial_size = (32, 32)
param_hist_bins = 32
param_orient = 9
param_pix_per_cell = 8
param_cells_per_block = 2

# column 0, RGB image and corresponding hog
R_features, R_hog_image = get_hog_features(image[:, :, 2], orient=param_orient, pix_per_cell=param_pix_per_cell,
                                           cell_per_block=param_cells_per_block, vis=True)
G_features, G_hog_image = get_hog_features(image[:, :, 1], orient=param_orient, pix_per_cell=param_pix_per_cell,
                                           cell_per_block=param_cells_per_block, vis=True)
B_features, B_hog_image = get_hog_features(image[:, :, 0], orient=param_orient, pix_per_cell=param_pix_per_cell,
                                           cell_per_block=param_cells_per_block, vis=True)

plts[0][0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plts[0][0].set_title('RGB Image', fontsize=12)
plts[1][0].imshow(R_hog_image, cmap='gray')
plts[1][0].set_title('R Channel HOG', fontsize=12)
plts[2][0].imshow(G_hog_image, cmap='gray')
plts[2][0].set_title('G Channel HOG', fontsize=12)
plts[3][0].imshow(B_hog_image, cmap='gray')
plts[3][0].set_title('B Channel HOG', fontsize=12)

# column 1, HSV image and corresponding hog
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
H_features, H_hog_image = get_hog_features(hsv_image[:, :, 0], orient=param_orient, pix_per_cell=param_pix_per_cell,
                                           cell_per_block=param_cells_per_block, vis=True)
S_features, S_hog_image = get_hog_features(hsv_image[:, :, 1], orient=param_orient, pix_per_cell=param_pix_per_cell,
                                           cell_per_block=param_cells_per_block, vis=True)
V_features, V_hog_image = get_hog_features(hsv_image[:, :, 2], orient=param_orient, pix_per_cell=param_pix_per_cell,
                                           cell_per_block=param_cells_per_block, vis=True)

plts[0][1].imshow(hsv_image)
plts[0][1].set_title('HSV Image', fontsize=12)
plts[1][1].imshow(H_hog_image, cmap='gray')
plts[1][1].set_title('H Channel HOG', fontsize=12)
plts[2][1].imshow(S_hog_image, cmap='gray')
plts[2][1].set_title('S Channel HOG', fontsize=12)
plts[3][1].imshow(V_hog_image, cmap='gray')
plts[3][1].set_title('V Channel HOG', fontsize=12)

# column 2, YUV and corresponding HOG
yuv_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
Y_features, Y_hog_image = get_hog_features(yuv_image[:, :, 0], orient=param_orient, pix_per_cell=param_pix_per_cell,
                                           cell_per_block=param_cells_per_block, vis=True)
U_features, U_hog_image = get_hog_features(yuv_image[:, :, 1], orient=param_orient, pix_per_cell=param_pix_per_cell,
                                           cell_per_block=param_cells_per_block, vis=True)
V_features, V_hog_image = get_hog_features(yuv_image[:, :, 2], orient=param_orient, pix_per_cell=param_pix_per_cell,
                                           cell_per_block=param_cells_per_block, vis=True)

plts[0][2].imshow(yuv_image)
plts[0][2].set_title('YUV Image', fontsize=12)
plts[1][2].imshow(Y_hog_image, cmap='gray')
plts[1][2].set_title('Y Channel HOG', fontsize=12)
plts[2][2].imshow(U_hog_image, cmap='gray')
plts[2][2].set_title('U Channel HOG', fontsize=12)
plts[3][2].imshow(V_hog_image, cmap='gray')
plts[3][2].set_title('V Channel HOG', fontsize=12)

# column 3, YCrCb and corresponding HOG
ycrcb_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
Y_features, Y_hog_image = get_hog_features(ycrcb_image[:, :, 0], orient=param_orient, pix_per_cell=param_pix_per_cell,
                                           cell_per_block=param_cells_per_block, vis=True)
Cr_features, Cr_hog_image = get_hog_features(ycrcb_image[:, :, 1], orient=param_orient, pix_per_cell=param_pix_per_cell,
                                             cell_per_block=param_cells_per_block, vis=True)
Cb_features, Cb_hog_image = get_hog_features(ycrcb_image[:, :, 2], orient=param_orient, pix_per_cell=param_pix_per_cell,
                                             cell_per_block=param_cells_per_block, vis=True)

plts[0][3].imshow(ycrcb_image)
plts[0][3].set_title('YCrCb Image', fontsize=12)
plts[1][3].imshow(Y_hog_image, cmap='gray')
plts[1][3].set_title('Y Channel HOG', fontsize=12)
plts[2][3].imshow(Cr_hog_image, cmap='gray')
plts[2][3].set_title('Cr Channel HOG', fontsize=12)
plts[3][3].imshow(Cb_hog_image, cmap='gray')
plts[3][3].set_title('Cb Channel HOG', fontsize=12)

plt.tight_layout()
plt.show()
