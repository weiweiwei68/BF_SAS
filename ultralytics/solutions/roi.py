import cv2
import numpy as np

class RegionOfInterest:
    """
    Class to define and manage a region of interest (ROI) in an image or video frame.

    Attributes:
        zones (list): List of defined zones as lists of points.
        current_zone (list): Points of the zone currently being defined.
    """

    def __init__(self, image, **kwargs) -> None:
        """
        Initialize the RegionOfInterest class with parameters for ROI creation, including mouse event handling, points management, zones handling, image mask generation, and so on.

        Args:
            image (np.ndarray): The input image or frame to define the ROI on.
            **kwargs: Additional keyword arguments for ROI properties.
            overlay (tuple): Color for ROI overlay (default is red).
            alpha (float): Transparency for overlay (default is 0.3).
        
        Examples:
            >>> roi = RegionOfInterest(image, overlay=(0, 0, 255), alpha=0.3)
        """
        self.zones = []
        self.current_zone = []
        self.imageHeight = image.shape[0]
        self.imageWidth = image.shape[1]

        # ROI Properties
        self.mask = None       # 0/255 version of the mask
        self.mask_bool = None  # Boolean version of the mask
        self.overlay = np.zeros_like(image)
        self.overlay[:] = kwargs.get('overlay', (0, 0, 255))  # Color for ROI overlay (red)
        self.alpha = kwargs.get('alpha', 0.3)  # Transparency for overlay

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

    # TODO: finer
    def build_mask(self):
        """
        Build a binary mask from the defined zones.

        Returns:
            np.ndarray: Binary mask with the same shape as the image, where defined zones are filled.
        """
        self.mask = np.zeros((self.imageHeight, self.imageWidth), dtype=np.uint8)
        for zone in self.zones:
            pts = np.array(zone, np.int32).reshape((-1, 1, 2))  # Reshape for cv2.fillPoly
            cv2.fillPoly(self.mask, [pts], 255)
        self.mask_bool = self.mask.astype(bool)