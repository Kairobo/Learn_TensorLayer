import pprint
import tensorlayer as tl
img_train_list, ann_train_list, img_test_list, ann_test_list = tl.files.load_mpii_pose_dataset()
image = tl.vis.read_image(img_train_list[1])
tl.vis.draw_mpii_pose_to_image(image, ann_train_list[1], 'image1.png')
#pprint.pprint(ann_train_list[0])