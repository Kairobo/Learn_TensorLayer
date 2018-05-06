import pprint
import cv2
import tensorlayer as tl
import scipy.io as sio
import os
import imageio
#load annotation
path='data'
path = os.path.join(path, 'mpii_human_pose')
extracted_filename = "mpii_human_pose_v1_u12_2"
mat = sio.loadmat(os.path.join(path, extracted_filename, "mpii_human_pose_v1_u12_1.mat"))
#
img_train_list, ann_train_list, img_test_list, ann_test_list = tl.files.load_mpii_pose_dataset()
#for i in range(30):
c_i = 1#choose train_id
image = tl.vis.read_image(img_train_list[c_i])
image = tl.vis.draw_mpii_pose_to_image(image, ann_train_list[c_i])#, 'image1.png')
i_ti = 5#choose the image id
p_i = 1#choose human id
pos_x = mat['RELEASE']['annolist'][0,0][0][i_ti]['annorect'][0]['objpos'][p_i][0].tolist()[0][0][0][0]
pos_y = mat['RELEASE']['annolist'][0,0][0][i_ti]['annorect'][0]['objpos'][p_i][0].tolist()[0][1][0][0]
scale_i = mat['RELEASE']['annolist'][0,0][0][i_ti]['annorect'][0]['scale'][p_i][0,0]
radius = 2000 / scale_i

imh, imw = image.shape[0:2]
thick = int((imh + imw) // 430)

x1 = max(0,pos_x - radius/2)
y1 = max(0,pos_y - radius)
x2 = min(imw, pos_x + radius/2)
y2 = min(imh, pos_y + radius)
aaa = 1

cv2.rectangle(
    image,
    (int(x1), int(y1)),
    (int(x2), int(y2)),  # up-left and botton-right
    [0, 0, 255],
    thick
)
image_path = 'image3.png'
try:  # RGB
    imageio.imwrite(image_path, image)
except Exception:  # Greyscale
    imageio.imwrite(image_path, image[:, :, 0])