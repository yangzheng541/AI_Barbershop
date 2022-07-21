import json
import os
import time

import cv2.cv2 as cv2
import numpy as np
import paddlehub as hub

# 常量——路径
INPUT_PATH = 'input' + os.sep
DST_PATH = 'dst' + os.sep
DST_PRE_PATH = DST_PATH + os.sep + 'pre' + os.sep
DST_IMG_PATH = DST_PATH + os.sep + 'img' + os.sep
DST_LANDMARK_PATH = DST_PATH + os.sep + 'landmark' + os.sep
DST_MARK_PATH = DST_PATH + os.sep + 'mark' + os.sep
OUTPUT_PATH = 'output' + os.sep


# 常量——人脸特征点
COLOUR_CORRECT_BLUR_FRAC = 0.6
LEFT_EYE_POINTS = list(range(42, 48))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_BROW_POINTS = list(range(22, 27))
RIGHT_BROW_POINTS = list(range(17, 22))
NOSE_POINTS = list(range(27, 35))
MOUTH_POINTS = list(range(48, 61))
OVERLAY_POINTS = [
    LEFT_EYE_POINTS + RIGHT_EYE_POINTS + LEFT_BROW_POINTS + RIGHT_BROW_POINTS,
    NOSE_POINTS + MOUTH_POINTS,
]
FEATHER_AMOUNT = 11


def get_landmark(img):
    module = hub.Module(name="face_landmark_localization")
    result = module.keypoint_detection(images=[img])
    return result[0]['data'][0]


def transformation_from_points(points1, points2):
    points1 = points1.astype(np.float64)
    points2 = points2.astype(np.float64)

    c1 = np.mean(points1, axis=0)
    c2 = np.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2

    s1 = np.std(points1)
    s2 = np.std(points2)
    points1 /= s1
    points2 /= s2

    U, S, Vt = np.linalg.svd(points1.T * points2)
    R = (U * Vt).T

    return np.vstack([np.hstack(((s2 / s1) * R,
                                 c2.T - (s2 / s1) * R * c1.T)),
                      np.matrix([0., 0., 1.])])


def warp_im(im, M_, dshape):
    output_im = np.zeros(dshape, dtype=im.dtype)
    cv2.warpAffine(im,
                   M_[:2],
                   (dshape[1], dshape[0]),
                   dst=output_im,
                   borderMode=cv2.BORDER_TRANSPARENT,
                   flags=cv2.WARP_INVERSE_MAP)
    return output_im


def warp_im1(im, M_, dshape):
    output_im = np.zeros(dshape, dtype=im.dtype)
    cv2.warpAffine(im,
                   M_[:2],
                   (dshape[1], dshape[0]),
                   dst=output_im,
                   borderValue=(1, 1, 1),
                   flags=cv2.WARP_INVERSE_MAP)
    return output_im


def get_segmentation(img):
    human_parser = hub.Module(name="ace2p")
    segmentation = human_parser.segmentation(images=[img])
    return segmentation[0]['data']


def get_face_mask(im, landmarks):
    im = np.zeros(im.shape[:2], dtype=np.float64)

    for group in OVERLAY_POINTS:
        draw_convex_hull(im,
                         landmarks[group],
                         color=1)

    im = np.array([im, im, im]).transpose((1, 2, 0))

    im = (cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0) > 0) * 1.0
    im = cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0)

    return im


def correct_colours(im1, im2, landmarks1):
    blur_amount = COLOUR_CORRECT_BLUR_FRAC * np.linalg.norm(
        np.mean(landmarks1[LEFT_EYE_POINTS], axis=0) -
        np.mean(landmarks1[RIGHT_EYE_POINTS], axis=0))
    blur_amount = int(blur_amount)
    if blur_amount % 2 == 0:
        blur_amount += 1
    im1_blur = cv2.GaussianBlur(im1, (blur_amount, blur_amount), 0)
    im2_blur = cv2.GaussianBlur(im2, (blur_amount, blur_amount), 0)

    # Avoid divide-by-zero errors.
    im2_blur = im2_blur + 128 * (im2_blur <= 1.0)

    return (im2.astype(np.float64) * im1_blur.astype(np.float64) /
            im2_blur.astype(np.float64))


def draw_convex_hull(im, points, color):
    points = points.astype(np.float32)
    points = cv2.convexHull(points)
    cv2.fillConvexPoly(im, points.astype(np.int), color=color)


def get_im1s(im1_name):
    im1 = cv2.imread(INPUT_PATH + im1_name)
    im1_mark = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    s = get_segmentation(im1)
    for h in range(len(s)):
        for w in range(len(s[0])):
            if s[h][w] != 2:
                im1_mark[h][w] = 0
    im1_landmark = np.mat(get_landmark(im1))
    return im1, im1_mark, im1_landmark


def get_im2s(im2_name):
    if not os.path.isfile(DST_LANDMARK_PATH + im2_name[:im2_name.rindex('.'):] + '.json'):
        im2 = cv2.imread(DST_PRE_PATH + im2_name)
        im2_landmark = get_landmark(im2)
        with open(DST_LANDMARK_PATH + im2_name[:im2_name.rindex('.'):] + '.json', 'w') as f:  # 注意：若输入重名，则缓存将被替换
            f.write(json.dumps({'data': im2_landmark}, indent=4))
    else:
        with open(DST_LANDMARK_PATH + im2_name[:im2_name.rindex('.'):] + '.json', 'r') as f:
            im2_landmark = json.load(f)['data']
    im2_landmark = np.mat(im2_landmark)

    if not (os.path.isfile(DST_MARK_PATH + im2_name) and os.path.isfile(DST_IMG_PATH + im2_name)):
        im2 = cv2.imread(DST_PRE_PATH + im2_name)
        im2_mark = im2.copy()
        s = get_segmentation(im2)
        for h in range(len(s)):
            for w in range(len(s[0])):
                if s[h][w] == 2:
                    im2_mark[h][w] = (0, 0, 0)
                else:
                    im2[h][w] = (0, 0, 0)
                    im2_mark[h][w] = (1, 1, 1)
        cv2.imwrite(DST_IMG_PATH + im2_name, im2)
        cv2.imwrite(DST_MARK_PATH + im2_name, im2_mark)
    else:
        im2_mark = cv2.imread(DST_MARK_PATH + im2_name)
        im2 = cv2.imread(DST_IMG_PATH + im2_name)

    return im2, im2_mark, im2_landmark


def check_img(im1_name, im2_name, mode):
    if not os.path.isfile(INPUT_PATH + im1_name):
        print('用户输入的待换图片{}不存在'.format(im1_name))
        return None
    if mode == 1:
        if (not os.path.isfile(DST_MARK_PATH + im2_name) or not
            os.path.isfile(DST_IMG_PATH + im2_name) or not
            os.path.isfile(DST_LANDMARK_PATH + im2_name)) \
                and not os.path.isfile(DST_PRE_PATH + im2_name):
            print('用户输入的目标图片{}不存在'.format(im2_name))
            return None
    elif mode == 2:
        if not os.path.isfile(DST_LANDMARK_PATH + im2_name) and not os.path.isfile(DST_PRE_PATH + im2_name):
            print('用户输入的目标图片{}不存在'.format(im2_name))
            return None
    return True


def introduction():
    print('---------------命令简介---------------')
    print('N/n--模式1（将目标的发型换到自己脸上）')
    print('Y/y--模式2（将自己的脸型融合到他人的脸上）')
    print('E/e--退出本系统')
    print('------------------------------------')


def welcome():
    print('------------欢迎访问一键换发型系统------------')
    print('初次访问，如对命令尚不熟悉，可输入?查询命令使用方法')


def mode1():
    # 输入图片名并校验图片
    print('【提示】请确认你的待换发型头像置于input文件夹中；目标发型头像置于pre文件夹中（均需为jpg文件，需要带后缀）')
    im1_name = input('输入待换发型头像图片名：')
    im2_name = input('输入目标发型头像图片名：')
    if check_img(im1_name, im2_name, mode=1) is None:
        return

    # 获得待换头像及目标发型头像 ==> 遮罩、标识点
    im1, im1_mark, im1_landmark = get_im1s(im1_name)
    im2, im2_mark, im2_landmark = get_im2s(im2_name)

    # 仿射变换，将目标头像/遮罩与输入头像对准
    M = transformation_from_points(im1_landmark, im2_landmark)
    warp_img = warp_im(im2, M, im1.shape)
    warp_img_mark = warp_im1(im2_mark, M, im1.shape)

    # 融化背景 + 图像融合
    inpaint_img = cv2.inpaint(im1, im1_mark, 3, cv2.INPAINT_NS)
    end_img = warp_img + (warp_img_mark * inpaint_img)
    end_img = cv2.GaussianBlur(end_img, (3, 3), 0)
    end_img_name = str(time.time()) + '.jpg'
    cv2.imwrite(OUTPUT_PATH + end_img_name, end_img)
    print('转换成功，最终文件保存在output文件夹中，文件名为{}，请继续下一张吧！'.format(end_img_name))


def mode2():
    # 读取文件
    print('【提示】请确认你的待换发型头像置于input文件夹中；目标发型头像置于pre文件夹中（均需为jpg文件，需要带后缀）')
    im1_name = input('输入待换发型头像图片名：')  # 待换脸头像/目标发型的头像
    im2_name = input('输入目标发型头像图片名：')  # 源头像/需换发型的头像
    if check_img(im1_name, im2_name, mode=2) is None:
        return
    im1 = cv2.imread(INPUT_PATH + im1_name)
    im2 = cv2.imread(DST_PRE_PATH + im2_name)

    # 分别得到68个特征点的坐标
    landmarks1 = np.mat(get_landmark(im1))
    if not os.path.isfile(DST_LANDMARK_PATH + im2_name[:im2_name.rindex('.'):] + '.json'):
        landmarks2 = get_landmark(im2)
        with open(DST_LANDMARK_PATH + im2_name[:im2_name.rindex('.'):] + '.json', 'w') as f:  # 注意：若输入重名，则缓存将被替换
            f.write(json.dumps({'data': landmarks2}, indent=4))
    else:
        with open(DST_LANDMARK_PATH + im2_name[:im2_name.rindex('.'):] + '.json', 'r') as f:
            landmarks2 = json.load(f)['data']
    landmarks2 = np.mat(landmarks2)

    # 得到仿射变换矩阵
    M = transformation_from_points(landmarks2, landmarks1)

    # 得到不同蒙版
    mask = get_face_mask(landmarks1, landmarks1)  # im1应显示的蒙版
    warped_mask = warp_im(mask, M, im2.shape)  # im2应显示的蒙版
    combined_mask = np.max([get_face_mask(im2, landmarks2), warped_mask], axis=0)  # im1/im2共同显示的蒙版

    # 得到变化及矫正颜色后的im2
    warped_im1 = warp_im(im1, M, im2.shape)
    warped_corrected_im1 = correct_colours(im2, warped_im1, landmarks2)

    # 得到结果
    output_im = im2 * (1.0 - combined_mask) + warped_corrected_im1 * combined_mask
    cv2.imwrite(OUTPUT_PATH + str(time.time()) + '.jpg', output_im)
    print('转换成功，请继续下一张吧！')


def mode3():
    print('恭喜你！激发了隐藏功能噢~ | 直男为女友的必备神器①')
    print('------这是一个换口红色号的功能------')
    print('请确认你的所换色号的图片位于dst/pre目录下后，以RGB格式，逗号分隔的输入你想换的口红色号。')
    print('例：128, 0, 128')
    img_name = input('请输入待换图片名：')
    color = input('色号：').split(',')
    img = cv2.imread(DST_PRE_PATH + img_name)

    # 68个人脸识别点检测
    img_landmark = get_landmark(img)
    points = np.array([[int(x), int(y)] for x, y in img_landmark])

    # 对嘴唇部分提取
    lip_mask = cv2.fillPoly(
        np.zeros_like(img),
        [points[48:61]],
        (255, 255, 255)
    )

    # 嘴唇区域上色并与原图融合
    lip_color = np.zeros_like(lip_mask)
    lip_color[:] = color  # ---------------换口红颜色
    lip_color = cv2.bitwise_and(lip_mask, lip_color)
    lip_color = cv2.GaussianBlur(lip_color, (7, 7), 10)
    img_result = cv2.addWeighted(img, 1, lip_color, 0.4, 0)
    cv2.imwrite(OUTPUT_PATH + str(time.time()) + '.jpg', img_result)


if __name__ == '__main__':
    # 读入图片
    welcome()
    while True:
        cmd = input('命令输入：')
        if cmd.capitalize() == 'N':
            mode1()
        elif cmd.capitalize() == 'Y':
            mode2()
        elif cmd.capitalize() == 'E':
            break
        elif cmd.capitalize() == 'D':
            mode3()
        elif cmd == '?':
            introduction()
        else:
            print('您的输入有误，请重新输入')
