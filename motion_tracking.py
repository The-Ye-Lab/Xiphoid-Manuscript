import cv2
import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt

def detect_mouse(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (35, 35), 0)
    _, thresh = cv2.threshold(blurred, 30, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) != 0:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        return x + w // 2, y + h // 2
    else:
        return None

def save_coordinates_to_csv(coordinates, output_file):
    df = pd.DataFrame(coordinates, columns=['frame', 'x', 'y'])
    df.to_csv(output_file, index=False)

def create_heat_map(coordinates, output_file):
    x_values = [coord[1] for coord in coordinates]
    y_values = [coord[2] for coord in coordinates]

    heatmap, _, _ = np.histogram2d(x_values, y_values, bins=(100, 100))
    plt.imshow(heatmap.T, origin='lower', cmap='hot')
    plt.colorbar()
    plt.savefig(output_file)

def main():
    video_file = 'mouse_video.mp4'
    output_csv = 'mouse_coordinates.csv'
    output_heatmap = 'mouse_heatmap.png'

    cap = cv2.VideoCapture(video_file)

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    coordinates = []

    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break

        mouse_center = detect_mouse(frame)

        if mouse_center is not None:
            x, y = mouse_center
            coordinates.append((i, x, y))

    cap.release()

    save_coordinates_to_csv(coordinates, output_csv)
    create_heat_map(coordinates, output_heatmap)

if __name__ == '__main__':
    main()
