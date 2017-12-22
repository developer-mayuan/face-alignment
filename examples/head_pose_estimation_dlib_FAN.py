import sys, os, math, argparse
import face_alignment
import numpy as np
import cv2
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from skimage import io

P3D_RIGHT_SIDE = np.float32([-100.0, -77.5, -5.0])  # 0
P3D_GONION_RIGHT = np.float32([-110.0, -77.5, -85.0])  # 4
P3D_MENTON = np.float32([0.0, 0.0, -122.7])  # 8
P3D_GONION_LEFT = np.float32([-110.0, 77.5, -85.0])  # 12
P3D_LEFT_SIDE = np.float32([-100.0, 77.5, -5.0])  # 16
P3D_FRONTAL_BREADTH_RIGHT = np.float32([-20.0, -56.1, 10.0])  # 17
P3D_FRONTAL_BREADTH_LEFT = np.float32([-20.0, 56.1, 10.0])  # 26
P3D_SELLION = np.float32([0.0, 0.0, 0.0])  # 27
P3D_NOSE = np.float32([21.1, 0.0, -48.0])  # 30
P3D_SUB_NOSE = np.float32([5.0, 0.0, -52.0])  # 33
P3D_RIGHT_EYE = np.float32([-20.0, -65.5, -5.0])  # 36
P3D_RIGHT_TEAR = np.float32([-10.0, -40.5, -5.0])  # 39
P3D_LEFT_TEAR = np.float32([-10.0, 40.5, -5.0])  # 42
P3D_LEFT_EYE = np.float32([-20.0, 65.5, -5.0])  # 45
# P3D_LIP_RIGHT = np.float32([-20.0, 65.5,-5.0]) #48
# P3D_LIP_LEFT = np.float32([-20.0, 65.5,-5.0]) #54
P3D_STOMION = np.float32([10.0, 0.0, -75.0])  # 62

TRACKED_POINTS = (0, 4, 8, 12, 16, 17, 26, 27, 30, 33, 36, 39, 42, 45, 62)

# Run the 3D face alignment on a test image, without CUDA.
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D,
                                  enable_cuda=False, flip_input=False)


# todo: add camera calibration process here.

# todo: load standard head model.

def return_landmarks_from_prediction(preds, points_to_return=None):
    """

    :param preds:
    :param points_to_return:
    :return:
    """

    if not points_to_return:
        landmarks_2D = np.column_stack((preds[:, 0], preds[:, 1]))
    else:
        landmarks_2D = np.column_stack((preds[points_to_return, 0],
                                        preds[points_to_return, 1]))

    return landmarks_2D


def parse_args():
    """

    :return:
    """
    parser = argparse.ArgumentParser(
        description='Head pose estimation using the Hopenet network.')
    parser.add_argument('--video', dest='video_path', help='Path of video')
    parser.add_argument('--output_string', dest='output_string',
                        help='String appended to output file')
    parser.add_argument('--n_frames', dest='n_frames',
                        help='Number of frames', type=int)
    parser.add_argument('--fps', dest='fps',
                        help='Frames per second of source video',
                        type=float, default=30.)

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    camera_matrix = np.float32(
        [[1691.76110953477, 0.0, 640.601537464294],
         [0.0, 1740.30146114039, 445.570081380801],
         [0.0, 0.0, 1.0]])

    print("Estimated camera matrix: \n" + str(camera_matrix) + "\n")

    camera_distortion = np.float32([0.0, 0.0, 0.0, 0.0, 0.0])

    landmarks_3D = np.float32([P3D_RIGHT_SIDE,
                               P3D_GONION_RIGHT,
                               P3D_MENTON,
                               P3D_GONION_LEFT,
                               P3D_LEFT_SIDE,
                               P3D_FRONTAL_BREADTH_RIGHT,
                               P3D_FRONTAL_BREADTH_LEFT,
                               P3D_SELLION,
                               P3D_NOSE,
                               P3D_SUB_NOSE,
                               P3D_RIGHT_EYE,
                               P3D_RIGHT_TEAR,
                               P3D_LEFT_TEAR,
                               P3D_LEFT_EYE,
                               P3D_STOMION])

    # store the pose registration result.
    bias_rvec = np.float32([0, 0, 0])
    bias_rvec_matrix = cv2.Rodrigues(bias_rvec)[0]
    calibrated_fg = False

    video_path = args.video_path
    output_dir = 'output/video'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not os.path.exists(video_path):
        sys.exit('Video does not exist.')

    video = cv2.VideoCapture(video_path)

    if not args.n_frames:
        args.n_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter('output/video/%s.avi' % args.output_string,
                          fourcc, args.fps, (width, height))
    txt_out = open('output/video/%s.txt' % args.output_string, 'w')

    frame_num = 1

    while frame_num < args.n_frames:
        ret, frame = video.read()
        if ret is False:
            break

        try:
            preds = fa.get_landmarks(frame)[-1]
        except TypeError:
            print("Cannot detect face for frame Index ." + str(frame_num))
            out.write(frame)
            txt_out.write(str(frame_num) + '\%f\t%f\t%f\n' % (0, 0, 0))
            frame_num += 1
            continue

        # todo: visualize the 2D point using cv2
        for i in range(0, 68):
            cv2.circle(frame, (preds[i, 0], preds[i, 1]), 2, (0, 0, 255),
                       -1)

        landmarks_2D = return_landmarks_from_prediction(preds,
                                                        points_to_return=TRACKED_POINTS)
        retval, rvec, tvec = cv2.solvePnP(landmarks_3D,
                                          landmarks_2D,
                                          camera_matrix,
                                          camera_distortion)

        if not calibrated_fg:
            bias_rvec = rvec
            bias_rvec_matrix = cv2.Rodrigues(bias_rvec)[0]
            calibrated_fg = True

        raw_rvec_matrix = cv2.Rodrigues(rvec)[0]
        rvec_matrix = np.dot(np.linalg.inv(bias_rvec_matrix),
                             raw_rvec_matrix)
        head_pose = np.hstack((rvec_matrix, tvec))
        R = np.vstack((head_pose, [0, 0, 0, 1]))

        sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        singular = sy < 1e-6

        if not singular:
            x = math.atan2(R[2, 1], R[2, 2])
            y = math.atan2(-R[2, 0], sy)
            z = math.atan2(R[1, 0], R[0, 0])
        else:
            x = math.atan2(-R[1, 2], R[1, 1])
            y = math.atan2(-R[2, 0], sy)
            z = 0

        euler_angles = np.array([x, y, z]) / np.pi * 180

        roll_predicted = euler_angles[0]
        pitch_predicted = euler_angles[1]
        yaw_predicted = euler_angles[2]

        print("Frame Index: %d Landmark-based Estimated "
              "[roll, pitch, yaw] ..... [%f\t%f\t%f]\n" %
              (frame_num, roll_predicted, pitch_predicted, yaw_predicted))

        txt_out.write(str(frame_num) + '\t%f\t%f\t%f\n' % (
            yaw_predicted, pitch_predicted, roll_predicted))

        # Now we project the 3D points into the image plane
        # Creating a 3-axis to be used as reference in the image.
        axis = np.float32([[50, 0, 0],
                           [0, 50, 0],
                           [0, 0, 50]])
        imgpts, jac = cv2.projectPoints(axis, rvec, tvec, camera_matrix,
                                        camera_distortion)

        sellion_xy = (landmarks_2D[7][0], landmarks_2D[7][1])
        cv2.line(frame, sellion_xy, tuple(imgpts[1].ravel()),
                 (0, 255, 0), 3)  # GREEN
        cv2.line(frame, sellion_xy, tuple(imgpts[2].ravel()),
                 (255, 0, 0), 3)  # BLUE
        cv2.line(frame, sellion_xy, tuple(imgpts[0].ravel()),
                 (0, 0, 255), 3)  # RED

        # cv2.imshow('Video', frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

        out.write(frame)
        frame_num += 1

    out.release()
    video.release()
    txt_out.close()


if __name__ == '__main__':
    main()
