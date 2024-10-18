from ultralytics import YOLO
import pyautogui
from PIL import Image
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import time

model = YOLO("weights/best.pt")
model.to("cuda")

x_center = 1920/2
y_center = 1080/2
closest_gem_distance = float('inf')
safe_dist = 200
roam_dist = 500

output_folder = "detections"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

def classify_direction(x1, y1):
    if x1 < x_center and y1 < y_center:
        return "topleft"
    elif x1 > x_center and y1 < y_center:
        return "topright"
    elif x1 < x_center and y1 > y_center:
        return "bottomleft"
    else:
        return "bottomright"

def move(boxes, classes):
    direction_count = {"topleft": 0, "topright": 0, "bottomleft": 0, "bottomright": 0}
    danger = 0
    roam = True
    choice = ["still", 0]

    for box, cls in zip(boxes, classes):
        x1, y1, x2, y2 = box
        x_center_box = (x1 + x2) / 2
        y_center_box = (y1 + y2) / 2
        box_center_coords = (x_center_box, y_center_box)
        enemy_distance = find_distance(x_center_box, y_center_box, x_center, y_center)

        if names[cls].startswith("e_"):
            if enemy_distance < roam_dist:
                roam = False
                pyautogui.keyUp("a")

            if enemy_distance < safe_dist:
                runaway(box_center_coords, None) 
                danger += 1
                continue

            direction = classify_direction(x1, y1)
            direction_count[direction] += 1

            if direction_count[direction] > choice[1]:
                choice = [direction, direction_count[direction]]

        elif names[cls] == "levelup": 
            pyautogui.keyDown("enter")

    if danger == 0:
        runaway(None, choice[0])
    elif danger > 5:
        move_circularly()
    elif roam:
        roam()


def roam():
    pyautogui.keyUp("s")
    pyautogui.keyUp("w")
    pyautogui.keyUp("d")
    pyautogui.keyDown("a")

def move_circularly(duration = 2, sleep_time = 0.1):
    directions = [
        ("w","a"),
        ("w","d"),
        ("s","a"),
        ("s","d"),
    ]
    start_time = time.time()
    while time.time() < start_time + duration:
        for direction in directions:
            pyautogui.keyDown(direction[0])
            pyautogui.keyDown(direction[1])
            time.sleep(sleep_time)
            pyautogui.keyUp(direction[0])
            pyautogui.keyUp(direction[1])
            time.sleep(sleep_time)
        
        
def move_gem(coords):
    x, y = coords
    pyautogui.keyUp("d" if x < x_center else "a")
    pyautogui.keyDown("a" if x < x_center else "d")

    pyautogui.keyUp("s" if y < y_center else "w")
    pyautogui.keyDown("w" if y < y_center else "s")


def runaway(coords=None, choice=None):
    if choice == None:
        x, y = coords
        pyautogui.keyUp("d" if x > x_center else "a")
        pyautogui.keyDown("a" if x > x_center else "d")

        pyautogui.keyUp("s" if y > y_center else "w")
        pyautogui.keyDown("w" if y > y_center else "s")

    elif choice == "still":
        pyautogui.keyUp("s")
        pyautogui.keyUp("w")
        pyautogui.keyUp("d")
        pyautogui.keyUp("a")

    else:
        pyautogui.keyUp("d" if choice.endswith("left") else "a")
        pyautogui.keyDown("a" if choice.endswith("left") else "d")

        pyautogui.keyUp("s" if choice.startswith("top") else "w")
        pyautogui.keyDown("w" if choice.startswith("top")else "s")

def find_distance(x1, y1, x2, y2):
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

#screenshot_count = 0

while True:
    screenshot = pyautogui.screenshot()
    screenshot = Image.frombytes("RGB", screenshot.size, screenshot.tobytes())
    result = model.predict(source=screenshot, show=False, conf=0.55)

    """fig, ax = plt.subplots(1)
    ax.imshow(screenshot)"""

    boxes = result[0].boxes.xyxy.tolist()
    classes = result[0].boxes.cls.tolist()
    names = result[0].names
    confidences = result[0].boxes.conf.tolist()

    """for box, cls, conf in zip(boxes, classes, confidences):
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

        label = f"{names[cls]} {conf:.2f}"
        plt.text(x1, y1, label, color="white", fontsize=2)

        if names[cls] == "gem":
            gem_distance = find_distance(x_center_box, y_center_box, x_center, y_center)
            if gem_distance < closest_gem_distance:
                closest_gem_distance = gem_distance
                closest_gem_coords = box_center_coords
            move_gem(closest_gem_coords)"""
    
    move(boxes, classes)

    """save_path = os.path.join(output_folder, f"screenshot_{screenshot_count}.png")
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    screenshot_count += 1

    plt.close(fig)  """         
            




