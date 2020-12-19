import pathlib
import argparse
import cv2

from util import morph_image

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--mode', choices=['training', 'test', 'predict'])
parser.add_argument(
    '-n', '--name', help="enter name of folder", required=True, type=str)
parser.add_argument('-g', '--grayscale',
                    help="convert images to greyscale", action="store_true")
args = parser.parse_args()


def generate_images(name, grayscale=False, mode='training'):
    """
    Generate images from webcam
    """
    cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

    count = 1
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            continue

        img = frame
        if grayscale:
            img = morph_image(frame)

        # draw rectangle
        cv2.rectangle(img, (8, 8), (322, 322), (255, 0, 0), 1)
        cv2.imshow('Image', img)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            crop = img[10:320, 10:320]
            resized = cv2.resize(crop, (64, 64))
            pathlib.Path(
                f'./data/{mode}/{name}').mkdir(parents=True, exist_ok=True)
            cv2.imwrite(f'./data/{mode}/{name}/{name}_{count}.jpg', resized)
            count += 1
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    generate_images(args.name, args.grayscale, args.mode)
