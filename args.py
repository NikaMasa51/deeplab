# img_dir = './data/segmentation_full/images'
# mask_dir ='./data/segmentation_full/masks'
img_dir = './data/segmentation_sub/images'
mask_dir ='./data/segmentation_sub/masks'
out_dir = './runs'
n_epochs = 100
n_classes = 2
# classes = ['Broccoli']
classes = ['ear', 'ear_sub']
device_ids = [0,1,2,3]
batch_size = 32
lr = 0.002
beta_1 = 0.5
beta_2 = 0.999
im_size = 256
