import cv2
import numpy as np

IMG_PATH = "C:\\Users\\User\\Desktop\\William\\SAS\\ultralytics\\area_detection/test2.jpg"
img_orig = cv2.imread(IMG_PATH)
if img_orig is None:
    raise FileNotFoundError(f"Couldn't read: {IMG_PATH}")

# --- scale for display (fit-to-window) ---
max_w, max_h = 1280, 800          # tweak if you want a bigger/smaller preview
H, W = img_orig.shape[:2]
scale = min(max_w / W, max_h / H, 1.0)  # only downscale, never upscale
if scale < 1.0:
    disp_size = (int(W * scale), int(H * scale))
    img_disp = cv2.resize(img_orig, disp_size, interpolation=cv2.INTER_AREA)
else:
    img_disp = img_orig.copy()
disp_clone = img_disp.copy()

zones, current_zone = [], []

def redraw_display():
    """Redraw all closed zones on the display image."""
    global img_disp
    img_disp = disp_clone.copy()
    for z in zones:
        pts_disp = (np.array(z, np.float32) * scale).astype(np.int32)
        cv2.polylines(img_disp, [pts_disp], True, (255, 0, 0), max(1, int(2 * scale)))

def click_event(event, x, y, flags, param):
    global current_zone, zones, img_disp
    if event == cv2.EVENT_LBUTTONDOWN:
        # map display click -> original coords
        ox, oy = int(round(x / scale)), int(round(y / scale))
        current_zone.append((ox, oy))
        # draw a dot on the display image
        cv2.circle(img_disp, (x, y), max(3, int(5 * scale)), (0, 255, 0), -1)

    elif event == cv2.EVENT_RBUTTONDOWN:
        if len(current_zone) >= 3:
            zones.append(current_zone[:])
            current_zone.clear()
            redraw_display()

WIN = "Select Area (L-click add, R-click close, Enter finish, C clear)"
cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
cv2.setMouseCallback(WIN, click_event)

while True:
    cv2.imshow(WIN, img_disp)
    key = cv2.waitKey(1) & 0xFF

    if key in (13, 10):   # Enter -> finish
        break
    elif key == 27:       # Esc -> abort
        zones = []
        break
    elif key in (ord('c'), ord('C')):  # clear and continue
        zones = []
        current_zone.clear()
        img_disp = disp_clone.copy()

cv2.destroyAllWindows()

if zones:
    print("Selected zones (original image coordinates):")
    for i, zone in enumerate(zones, 1):
        print(f"Zone {i}: {zone}")
else:
    print("No zones selected.")
