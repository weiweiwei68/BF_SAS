from typing import Any
import torch
import numpy as np

from ultralytics.solutions.solutions import BaseSolution, SolutionAnnotator, SolutionResults
from ultralytics.utils import LOGGER
from ultralytics.utils.plotting import colors
from ultralytics.solutions.roi import RegionOfInterest

class AreaInvasion(BaseSolution, RegionOfInterest):
    """
    A class to manage area invasion detection functionalities for real-time monitoring.

    This class extends the BaseSolution class and provides features to monitor objects entering a defined area,
    send email notifications when specific thresholds are exceeded for total detections, and annotate the output frame for
    visualization.

    Attributes:
        email_sent (bool): Flag to track if an email has already been sent for the current event.
        records (int): Threshold for the number of detected objects to trigger an alert.
        server (smtplib.SMTP): SMTP server connection for sending email alerts.
        to_email (str): Recipient's email address for alerts.
        from_email (str): Sender's email address for alerts.

    Methods:
        authenticate: Set up email server authentication for sending alerts.
        send_email: Send an email notification with details and an image attachment.
        process: Monitor the frame, process detections, and trigger alerts if thresholds are crossed.

    Examples:
        >>> area_invasion = AreaInvasion()
        >>> area_invasion.authenticate("
    """

    def __init__(self, image, **kwargs):
        """
        Initialize the AreaInvasion class with parameters for real-time object monitoring.
        
        Args:
            **kwargs (Any): Additional keyword arguments passed to the parent class.
        """
        BaseSolution.__init__(self, **kwargs)
        RegionOfInterest.__init__(self, image, **kwargs)
        self.email_sent = False
        self.records = self.CFG["records"]
        self.server = None
        self.to_email = ""
        self.from_email = ""

    def authenticate(self, from_email: str, password: str, to_email: str) -> None:
        pass

    def box_tensor_to_xyxy(self, box_tensor):
        """
        Convert a tensor of bounding boxes to xyxy format.

        Args:
            box_tensor (torch.Tensor): Tensor of shape (N, 4) where N is the number of boxes.

        Returns:
            np.ndarray: Array of shape (N, 4) with bounding boxes in xyxy format.
        """
        return box_tensor.cpu().numpy() if isinstance(box_tensor, torch.Tensor) else box_tensor
    
    def is_overlapping(self, box_xyxy):
        """
        Check if any bounding box overlaps with the user-defined mask.

        Args:
            box_xyxy (np.ndarray): Bounding boxes in xyxy format.

        Returns:
            bool: True if any box overlaps with the mask, False otherwise.
        """
        if self.mask_bool is None or not self.mask_bool.any():
            LOGGER.warning("Mask not built. Call build_mask() before checking for overlaps.")
            return False
        
        x1, y1, x2, y2 = map(int, box_xyxy)
        if np.any(self.mask_bool[y1:y2, x1:x2]):
            return True  # Overlap found

        return False # No overlap
        
    
    def process(self, im0) -> SolutionResults:
        """
        Monitor the frame, process detections, and trigger alerts if invasion is triggered.

        Args:
            im0 (np.ndarray): The input image or frame to be processed and annotated.

        Returns:
            (SolutionResults): The results object containing detection information and annotated frame.
        """
        self.build_mask()
        self.extract_tracks(im0)  # Extract tracks from the frame
        annotator = SolutionAnnotator(im0, line_width=self.line_width, example="area_invasion")
        for box in self.boxes:
            # Draw bounding boxes and labels on the frame
            if self.is_overlapping(self.box_tensor_to_xyxy(box)):
                color = (0, 0, 255) # Red for invasion
            else:
                color = colors(0, True)
            annotator.box_label(box=box, label=self.names[0], color=color)

        plot_im = annotator.result()
        #self.display_output(plot_im)  # Display output with base class function

        # Return a SolutionResults object
        return SolutionResults(plot_im=plot_im, total_tracks=len(self.track_ids), email_sent=self.email_sent)