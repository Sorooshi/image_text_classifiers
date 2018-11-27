import os, glob, shutil
import cv2
import numpy as np

new_row_dim = 224
new_col_dim = 224

images_path = "C:/Users/srsha/image_text_classifier/img_downloader/multi_labels_dataset/white watch/*.jpg"
images_file = glob.glob(images_path)
images_file_to_rotate = np.random.choice(images_file, size=round(len(images_file)/2.0))
images_file_to_adjust_gamma_half = np.random.choice(images_file, size=round(len(images_file)/2.0))
images_file_to_adjust_gamma_double = np.random.choice(images_file, size=round(len(images_file)/2.0))
print(len(images_file))
print(len(images_file_to_rotate), len(images_file_to_adjust_gamma_half), len(images_file_to_adjust_gamma_double))

dirname = images_path.split("/")[-2] + "-"
print(dirname)
if not os.path.exists(dirname):
    os.mkdir(dirname)

# image resizing the abnormal size detection:
list_not_rgb = []
for i in range(len(images_file)):
    # print("iii:",images_file[i])
    image = cv2.imread(images_file[i])
    rows, cols, chs = image.shape
    # if chs != 3 or rows !=256 or cols != 256:
    #     list_not_rgb.append(images_file[i])
    re_sized_image = cv2.resize(image,(new_row_dim,new_col_dim)) # dimension to resize the images
    new_image_name = images_path.split("/")[-2]+"-"+str(i)+".jpg"
    cv2.imwrite(os.path.join(dirname, new_image_name), re_sized_image)

IMG_INDEX = len(images_file) + 1
print("IMG_INDEX:", IMG_INDEX)
print(list_not_rgb, len(list_not_rgb))


# image rotation:
for j in range(len(images_file_to_rotate)):
    image = cv2.imread(images_file_to_rotate[j])
    re_sized_image = cv2.resize(image, (new_row_dim,new_col_dim))  # dimension to resize the images
    rows, cols, chs = re_sized_image.shape
    print(rows, cols, chs)
    rotation_angle = np.arange(90,360,90)
    rotation_matrix = cv2.getRotationMatrix2D((rows/2, cols/2), np.random.choice(rotation_angle), 1)
    rotated_image = cv2.warpAffine(re_sized_image, rotation_matrix, (cols, rows))
    new_image_name = images_path.split("/")[-2] + "-" + str(j+IMG_INDEX) + ".jpg"
    cv2.imwrite(os.path.join(dirname, new_image_name), rotated_image)
IMG_INDEX += len(images_file_to_rotate) + 1 
print("IMG_INDEX rotate:", IMG_INDEX, len(images_file_to_rotate))


# image brightness adjustment:
def adjust_gamma(image, gamma=1.0):
   invGamma = 1.0 / gamma
   table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
   return cv2.LUT(image, table)

# image contrast/brightness adjustment (gamma_half):
for k in range(len(images_file_to_adjust_gamma_half)):
    image = cv2.imread(images_file_to_adjust_gamma_half[k])
    re_sized_image = cv2.resize(image, (new_row_dim,new_col_dim))  # dimension to resize the images
    gamma_cor_image = adjust_gamma(re_sized_image, gamma=0.5)
    new_image_name = images_path.split("/")[-2] + "-" + str(IMG_INDEX+k) + ".jpg"
    cv2.imwrite(os.path.join(dirname, new_image_name), gamma_cor_image)
IMG_INDEX += len(images_file_to_adjust_gamma_half) + 1 
print("IMG_INDEX rotate:", IMG_INDEX, len(images_file_to_adjust_gamma_half))

 
# image contrast/brightness adjustment (gamma_double):
for l in range(len(images_file_to_adjust_gamma_double)):
    image = cv2.imread(images_file_to_adjust_gamma_double[l])
    re_sized_image = cv2.resize(image, (new_row_dim, new_col_dim))  # dimension to resize the images
    gamma_cor_image = adjust_gamma(re_sized_image, gamma=2.0)
    new_image_name = images_path.split("/")[-2] + "-" + str(IMG_INDEX+l)+ ".jpg"
    cv2.imwrite(os.path.join(dirname, new_image_name), gamma_cor_image)