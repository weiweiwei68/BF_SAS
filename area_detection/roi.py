import cv2
import numpy as np

class RegionOfInterest:
    """
    Class to define and manage a region of interest (ROI) in an image or video frame.

    Attributes:
        zones (list): List of defined zones as lists of points.
        current_zone (list): Points of the zone currently being defined.
    """

    def __init__(self):
        self.zones = []
        self.current_zone = []

    def add_point(self, point):
        """Add a point to the current zone."""
        self.current_zone.append(point)

    def close_zone(self):
        """Close the current zone and add it to the list of zones."""
        if len(self.current_zone) >= 3:
            self.zones.append(self.current_zone[:])
            self.current_zone.clear()

    def clear_zones(self):
        """Clear all defined zones."""
        self.zones = []
        self.current_zone.clear()

    def redraw_display(self, img_disp):
        """Redraw all closed zones on the display image."""
        for z in self.zones:
            pts_disp = (np.array(z, np.float32)).astype(np.int32)
            cv2.polylines(img_disp, [pts_disp], True, (255, 0, 0), max(1, int(2)))
        return img_disp
    
    def click_event(self, event, x, y, flags, param):
        """Handle mouse click events to add points or close zones."""
        img_disp = param
        if event == cv2.EVENT_LBUTTONDOWN:
            # Map display click -> original coords
            ox, oy = int(x), int(y)
            self.add_point((ox, oy))
            # Draw a dot on the display image
            cv2.circle(img_disp, (x, y), 5, (0, 255, 0), -1)

        elif event == cv2.EVENT_RBUTTONDOWN:
            self.close_zone()
            for z in self.zones:
                pts_disp = (np.array(z, np.float32)).astype(np.int32)
                cv2.polylines(img_disp, [pts_disp], True, (255, 0, 0), max(1, int(2)))
        
        return img_disp
    
    def print_zones(self):
        """Print the defined zones."""
        if self.zones:
            print("Selected zones (original image coordinates):")
            for i, zone in enumerate(self.zones, 1):
                print(f"Zone {i}: {zone}")
        else:
            print("No zones selected.")