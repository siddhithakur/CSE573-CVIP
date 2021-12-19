###############
##Design the function "calibrate" to  return
# (1) intrinsic_params: should be a list with four elements: [f_x, f_y, o_x, o_y], where f_x and f_y is focal length, o_x and o_y is offset;
# (2) is_constant: should be bool data type. False if the intrinsic parameters differed from world coordinates.
#                                            True if the intrinsic parameters are invariable.
#It is ok to add other functions if you need
###############
import numpy as np
from cv2 import imread, cvtColor, COLOR_BGR2GRAY, TERM_CRITERIA_EPS, TERM_CRITERIA_MAX_ITER, \
    findChessboardCorners, cornerSubPix, drawChessboardCorners

def calibrate(imgname):
    #......
    result=[]
    np.set_printoptions(suppress=True)
    criteria = (TERM_CRITERIA_EPS +
                TERM_CRITERIA_MAX_ITER, 30, 0.001)
    nx = 4
    ny = 9
    world_co = [[40, 0, 40], [40, 0, 30], [40, 0, 20], [40, 0, 10],
                [30, 0, 40], [30, 0, 30], [30, 0, 20], [30, 0, 10],
                [20, 0, 40], [20, 0, 30], [20, 0, 20], [20, 0, 10],
                [10, 0, 40], [10, 0, 30], [10, 0, 20], [10, 0, 10],
                [0, 0, 40], [0, 0, 30], [0, 0, 20], [0, 0, 10],
                [0, 10, 40], [0, 10, 30], [0, 10, 20], [0, 10, 10],
                [0, 20, 40], [0, 20, 30], [0, 20, 20], [0, 20, 10],
                [0, 30, 40], [0, 30, 30], [0, 30, 20], [0, 30, 10],
                [0, 40, 40], [0, 40, 30], [0, 40, 20], [0, 40, 10]]
    img = imread(imgname)
    gray = cvtColor(img, COLOR_BGR2GRAY)
    ret, corners = findChessboardCorners(gray, (nx, ny))
    if ret == True:
        # Draw and display the corners
        corners = cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        drawChessboardCorners(img, (nx, ny), corners, ret)
        #image = plt.imshow(img)
    object_m = []
    test_world = []
    test_img = []
    i = 0
    for m in range(0, 72, 2):

        if i == 16 or i == 18 or i == 19 or i == 17:
            if i > 35:
                break
            #print(corners[i][0])
            i += 1
            continue
        else:
            if i > 35:
                break
            test_world.append(world_co[i])
            test_img.append(corners[i])
            object_m.append(
                [world_co[i][0], world_co[i][1], world_co[i][2], 1, 0, 0, 0, 0, -corners[i][0][0] * world_co[i][0],
                 -corners[i][0][0] * world_co[i][1], -corners[i][0][0] * world_co[i][2], -corners[i][0][0]])
            object_m.append(
                [0, 0, 0, 0, world_co[i][0], world_co[i][1], world_co[i][2], 1, -corners[i][0][1] * world_co[i][0],
                 -corners[i][0][1] * world_co[i][1], -corners[i][0][1] * world_co[i][2], -corners[i][0][1]])
            # m+=2
            #print(corners[i][0])
            #print(i)
            i += 1

    #np_array = np.array(object_m, dtype=int)

    u, s, v = np.linalg.svd(object_m)

    np_array = np.array(v[-1, :]).reshape(3, 4)

    m3 = np.matrix(np_array[2, :3])

    normalize1 = np_array / np.linalg.norm(m3)

    m1=np.matrix(normalize1[0,:3])
    m2 = np.matrix(normalize1[1, :3])
    m3 = np.matrix(normalize1[2, :3])
    m4 = np.matrix([normalize1[0,3], normalize1[1,3], normalize1[2, 3]])

    o_x = m1 * np.transpose(m3)
    o_y = m2 * np.transpose(m3)
    f_x = np.sqrt((m1 * np.transpose(m1)) - o_x * o_x)
    f_y = np.sqrt((m2 * np.transpose(m2)) - o_y * o_y)

    result.extend([float(f_x),float(f_y),float(o_x),float(o_y)])

    return result, True


if __name__ == "__main__":
    intrinsic_params, is_constant = calibrate('checkboard.png')
    print(intrinsic_params)
    print(is_constant)