from lesson_functions import *
from glob import glob
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.svm import LinearSVC
import time
import pickle

# load all the training images
car_paths = glob('./vehicles/*/*.png')
noncar_paths = glob('./non-vehicles/*/*.png')
img_example = cv2.imread(car_paths[0])

# print info of the training dataset
print('Num of cars: {}'.format(len(car_paths)))
print('Num of non-cars: {}'.format(len(noncar_paths)))
print('Image shape: {}'.format(img_example.shape))

# set parameters for feature extraction
param_color_space = 'HSV'
param_spatial_size = (32, 32)
param_hist_bins = 32
param_orient = 8
param_pix_per_cell = 8
param_cell_per_block = 2
param_hog_channel = 'ALL'  # 0, 1, 2, or ALL
param_use_spatial_feature = True
param_use_hist_feature = True
param_use_hog_feature = True

# extract features
car_features = extract_features(car_paths, color_space=param_color_space, spatial_size=param_spatial_size,
                                hist_bins=param_hist_bins, orient=param_orient, pix_per_cell=param_pix_per_cell,
                                cell_per_block=param_cell_per_block, hog_channel=param_hog_channel,
                                use_spatial_feature=param_use_spatial_feature, use_hist_feature=param_use_hist_feature,
                                use_hog_feature=param_use_hog_feature)

noncar_features = extract_features(noncar_paths, color_space=param_color_space, spatial_size=param_spatial_size,
                                   hist_bins=param_hist_bins, orient=param_orient, pix_per_cell=param_pix_per_cell,
                                   cell_per_block=param_cell_per_block, hog_channel=param_hog_channel,
                                   use_spatial_feature=param_use_spatial_feature,
                                   use_hist_feature=param_use_hist_feature, use_hog_feature=param_use_hog_feature)

#create an array stack of feature vectors
features = np.vstack((car_features, noncar_features)).astype(np.float64)
#fit a per-column scaler
X_scaler = StandardScaler().fit(features)
#apply the scaler to X
scaled_X = X_scaler.transform(features)

#set labels
y = np.hstack((np.ones(len(car_features)), np.zeros(len(noncar_features))))

#split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=rand_state)

print('Feature vector length: {}'.format(len(X_train[0])))

#use a linear svc
svc = LinearSVC()
t = time.time()

svc.fit(X_train, y_train)
t2 = time.time()

print('Time consumed for training: {} sec'.format(round(t2-t, 2)))
print('Test Accuracy of SVC = {}'.format(svc.score(X_test, y_test), 4))

#save the model and corresponding parameters
model = {}
model['svc'] = svc
model['X_scaler'] = X_scaler
model['param_color_space'] = param_color_space
model['param_spatial_size'] = param_spatial_size
model['param_hist_bins'] = param_hist_bins
model['param_orient'] = param_orient
model['param_pix_per_cell'] = param_pix_per_cell
model['param_cell_per_block'] = param_cell_per_block
model['param_hog_channel'] = param_hog_channel
model['param_use_spatial_feature'] = param_use_spatial_feature
model['param_use_hist_feature'] = param_use_hist_feature
model['param_use_hog_feature'] = param_use_hog_feature
pickle.dump(model, open('model.p', 'wb'))
