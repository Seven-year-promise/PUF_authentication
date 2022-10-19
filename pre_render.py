from skimage import io, draw, color, transform
from util import circle_center, circle_center_otsu, circle_area_otsu, padding, find_edge_lw, find_edge_rw
from config import *

half_size = 960
# Press the green button in the gutter to run the script.
def preprocessing():
    id_cnt = -1
    for f in ORI_PATH.iterdir():
        if f.suffix != ".rar":
            for im_p in list(f.rglob("*.jpg")):
                id_cnt += 1
                #print(im_p.name)
                #if id_cnt < 1481:
                #    continue
                im = io.imread(im_p, as_gray=True)
                #if im_p.name == "100-0350.jpg":
                #    cv2.imshow("im_special", im[:, 427:3500])

                im_copy = im.copy()
                _, _, im_without_bg = circle_area_otsu(im_copy)
                #cv2.imshow("im_without_bg", im_without_bg)
                #cv2.waitKey(0)
                lw = find_edge_lw(im_without_bg)
                rw = find_edge_rw(im_without_bg)
                #print(lw, rw)
                im = im[:, lw:rw]
                #cv2.imshow("crop", im)
                #cv2.waitKey(0)
                im_copy = im.copy()
                (y_c, x_c), im_gray, otsu_im = circle_center_otsu(im_copy)
                #print(y_c, x_c)
                cropped = im_gray[(y_c-half_size):(y_c+half_size), (x_c-half_size):(x_c+half_size)]
                #cropped = padding(cropped,half_size)

                #cv2.imshow("cropped", cropped)
                #cv2.waitKey(0)

                cropped = transform.resize(cropped, (768, 768))
                #print(cropped.shape)
                io.imsave(NOBG_SAVE_ALL / (str(id_cnt) + im_p.suffix), cropped)


if __name__ == '__main__':
    preprocessing()