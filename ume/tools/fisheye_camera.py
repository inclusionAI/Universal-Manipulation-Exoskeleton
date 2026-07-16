def fisheye_center_crop(img, input_shape, square_crop_size, square_crop_shift=(0, 0)):
    H, W = input_shape
    center_y, center_x = H // 2 + square_crop_shift[1], W // 2 + square_crop_shift[0]
    half_crop_size = square_crop_size // 2
    crop_img = img[..., center_y - half_crop_size : center_y + half_crop_size, center_x - half_crop_size : center_x + half_crop_size, :]
    return crop_img

def jieruiweitong_fisheye_center_crop(img):
    assert img.shape[-3] == 480 and img.shape[-2] == 640, "Input image must be of shape (480, 640, C)"
    return fisheye_center_crop(img, input_shape=(480, 640), square_crop_size=350, square_crop_shift=(6, 8))
