import cv2
from ultralytics.utils.plotting import Annotator
from ultralytics import solutions
from ultralytics.solutions.roi import RegionOfInterest
import numpy as np
import torch
from ultralytics.solutions.mask import MaskedRegion



# Load a frame for processing
# For testing purposes, you can replace the path with your own image or video frame.
cap =cv2.VideoCapture(0)
# cap =cv2.VideoCapture("./test3.mp4")  # Use 0 for webcam or replace with a video file path
if not cap.isOpened():
    raise IOError("Error opening video stream or file")
ret, im0 = cap.read()
img_disp = im0.copy()
# Initialize the Region of Interest (ROI) manager
roi_manager = solutions.AreaInvasion(
        image=im0,
        show=True,  # display the output
        model="yolo11n.pt",  # i.e. yolo11s.pt, yolo11m.pt
        records=1,  # total detections count to send an email
        classes=[0],  # i.e. [0] for person, [0, 1] for person and car, etc.
    )

# Operate the ROI
WIN = "Select Area (L-click add, R-click close, Enter finish, C clear)"
cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
cv2.setMouseCallback(WIN, roi_manager.click_event, param=img_disp)

while True:
    cv2.imshow(WIN, img_disp)
    key = cv2.waitKey(1) & 0xFF

    if key in (13, 10):   # Enter -> finish
        break
    elif key == 27:       # Esc -> abort
        roi_manager.zones = []
        break
    elif key in (ord('c'), ord('C')):  # clear and continue
        roi_manager.zones = []
        roi_manager.current_zone.clear()
        img_disp = im0.copy()

cv2.destroyAllWindows()
roi_manager.print_zones()
# # im0 = cv2.imread("C:\\Users\\User\\Desktop\\William\\SAS\\ultralytics\\area_detection\\people.png")
# # if im0 is None:
# #     raise FileNotFoundError("Image not found or path is incorrect.")

# securityalarm = solutions.SecurityAlarm(
#     show=True,  # display the output
#     model="yolo11n.pt",  # i.e. yolo11s.pt, yolo11m.pt
#     records=1,  # total detections count to send an email
#     classes=[0],  # i.e. [0] for person, [0, 1] for person and car, etc.
# )

# results = securityalarm(im0)
# # annotator = Annotator(im0, line_width=10)
# # annotator.box_label(box=[10, 20, 5472, 3600], label="chihuahua", color=(0, 255, 0))
# # plot_im = annotator.result()

# # cv2.namedWindow("s", cv2.WINDOW_NORMAL)  # Allow window resizing
# # cv2.resizeWindow("s", 800, 600)          # Set window size to 800x600
# # cv2.imshow("s", results.plot_im)  # Display the output frame
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# def build_user_mask(zones, img_shape):
#     """
#     Build a mask from the defined zones.

#     Args:
#         zones (list): List of defined zones as lists of points.
#         img_shape (tuple): Shape of the image (height, width).

#     Returns:
#         np.ndarray: Mask with the same shape as the image, where defined zones are filled.
#     """
#     mask = np.zeros(img_shape[:2], dtype=np.uint8)
#     for zone in zones:
#         pts = np.array(zone, np.int32).reshape((-1, 1, 2))  # Reshape for cv2.fillPoly
#         cv2.fillPoly(mask, [pts], 255)
#     return mask

def box_tensor_to_xyxy(box_tensor):
    """
    Convert a tensor of bounding boxes to xyxy format.

    Args:
        box_tensor (torch.Tensor): Tensor of shape (N, 4) where N is the number of boxes.

    Returns:
        np.ndarray: Array of shape (N, 4) with bounding boxes in xyxy format.
    """
    return box_tensor.cpu().numpy() if isinstance(box_tensor, torch.Tensor) else box_tensor
"""
def overlap_area(user_mask_bool, box_xyxy):
    
    Calculate the overlap area between a user-defined mask and bounding boxes.

    Args:
        user_mask_bool (np.ndarray): Boolean mask where True indicates the area of interest.
        box_xyxy (np.ndarray): Bounding boxes in xyxy format.

    Returns:
        bool: True if there is any overlap, False otherwise.
    
    if not user_mask_bool.any():
        return False  # No area of interest defined

    for box in box_xyxy:
        x1, y1, x2, y2 = map(float, box)
        if np.any(user_mask_bool[y1:y2, x1:x2]):
            return True  # Overlap found

    return False  # No overlap found

def prepare_mask_u8(mask_img, frame_shape):
    H, W = frame_shape[:2]
    if mask_img.ndim == 3:
        mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
    if mask_img.dtype == bool:
        m = (mask_img.astype(np.uint8)) * 255
    elif mask_img.dtype == np.uint8:
        # normalize to 0/255 if it might contain 0/1
        m = (mask_img > 0).astype(np.uint8) * 255
    else:
        m = (mask_img > 0).astype(np.uint8) * 255
    if (m.shape[0] != H) or (m.shape[1] != W):
        m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)  # keep edges crisp
    return m
"""

# Example usage
if roi_manager.zones:
    #user_mask = roi_manager.build_mask(im0.shape)
    #user_mask_bool = user_mask.astype(bool)
    #mask_u8 = prepare_mask_u8(user_mask_bool, im0.shape)
    #mask_bool = mask_u8.astype(bool)
    # red_img = np.zeros_like(im0)
    # red_img[:] = (0, 0, 255)  # Red color
    # alpha = 0.20  # Transparency factor

    stream = cv2.VideoCapture(0)
    # stream = cv2.VideoCapture("./test3.mp4")
    assert stream.isOpened(), "Error reading video file"
    # Video writer
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
    video_writer = cv2.VideoWriter("security_output_0820.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    # from_email = "abc@gmail.com"  # the sender email address
    # password = "---- ---- ---- ----"  # 16-digits password generated via: https://myaccount.google.com/apppasswords
    # to_email = "xyz@gmail.com"  # the receiver email address

    # Initialize security alarm object
    # securityalarm = solutions.AreaInvasion(
    #     image=im0,
    #     show=True,  # display the output
    #     model="yolo11n.pt",  # i.e. yolo11s.pt, yolo11m.pt
    #     records=1,  # total detections count to send an email
    #     classes=[0],  # i.e. [0] for person, [0, 1] for person and car, etc.
    # )

    # Process video
    while cap.isOpened():
        success, frame = cap.read()

        if not success:
            print("Video frame is empty or video processing has been successfully completed.")
            break

        results = roi_manager(frame)
        out_frame = results.plot_im

        
        blended = cv2.addWeighted(out_frame, 1 - roi_manager.alpha, roi_manager.overlay, roi_manager.alpha, 0)

        out_vis = out_frame.copy()
        cv2.copyTo(blended, roi_manager.mask, out_vis)


    # print(results)  # access the output
    # cv2.imshow("Security Alarm", results.plot_im)  # display the output frame
        cv2.imshow("Security Alarm", out_vis)
        if cv2.waitKey(1) & 0xFF == ord("q"):
                cv2.destroyAllWindows()  # Closes current frame window

        video_writer.write(out_frame)  # write the processed frame.

cap.release()
video_writer.release()
cv2.destroyAllWindows()  # destroy all opened windows