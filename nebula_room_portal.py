"""
Rainbow Galaxy Portal

author: Elsa Culler
event: Spectra Art Space Immersive Exhibit May 2022

Interactive projector art project. Viewers will see their
mirror image with a delay/fade and 'pixie dust' effect

Project uses the following hardware:
  * IMX219-83 Stereo camera
  * Raspberry Pi 4 Compute Module and breakout board
  * HDMI projector
"""
import os

import cv2
import numpy as np

class Camera:
    """
    A class to manage the camera connection and images

    Iterates through stereo images on request and returns
    depth images.
    """

    gstreamer_pipeline = (
        'libcamerasrc camera-name="/base/soc/i2c0mux/i2c@1/ov5647@36" '
        '! video/x-raw,width=640,height=480,framerate=30/1'
        '! videoconvert '
        '! appsink')
    model_path = 'tf-segmentation-model.tflite'
    gamma = 1.0
    threshold_percent = 0.1
    prev_alpha = 0.5
    

    def __init__(self, camera_ids):
        """Establish camera connection"""
        self.stream = cv2.VideoCapture(
            self.gstreamer_pipeline, 
            cv2.CAP_GSTREAMER) 
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 2)

        self.gamma_table = (
            np.array(
                [((i / 255.0) ** (1.0 / self.gamma)) * 255
                 for i in np.arange(0, 256)])
            .astype("uint8"))
        self.hue = 0
        self._raw_image = None
        self._colorized_image = None

        # Initialize lag
        self.previous_image = None
        self.previous_display = None
        self.raw_image
        self.composite_image

    @property
    def raw_image(self):
        """Read in the latest image"""
        # Return from read() is flag, array    
        if self.previous_image is None:
            self.previous_image = self.stream.read()[1]
        else:
            self.previous_image = self._raw_image
        self._raw_image =  self.stream.read()[1]
        return self._raw_image
    
    @property
    def gamma_corrected_image(self):
        """Fix washed out images"""
        return cv2.LUT(self.raw_image, self.gamma_table)
        
    @property
    def normalized_image(self):
        """Normalize raw image"""
        return self.gamma_corrected_image.astype("float32") / 255
    
    @property
    def diff_image(self):
        return cv2.absdiff(self.gamma_corrected_image, self.previous_image)
        
    @property
    def mask_image(self):
        diff_grey = cv2.cvtColor(self.diff_image, cv2.COLOR_RGB2GRAY)
        diff_threshold = np.max(diff_grey) * self.threshold_percent
        _, mask = cv2.threshold(
            diff_grey, diff_threshold, 255, cv2.THRESH_BINARY)
        return mask

    @property
    def colorized_image(self):
        self.hue += 2
        rgb_image = cv2.cvtColor(self.mask_image, cv2.COLOR_GRAY2RGB)
        hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
        hsv_image[(rgb_image!=0).all(-1)] = (self.hue, 200, 200)
        rgb_shifted_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)
        return rgb_shifted_image

    @property
    def composite_image(self):
        if self.previous_display is None:
            self.previous_display = self.colorized_image
        composite = cv2.addWeighted(
            self.previous_display, 0.5, self.colorized_image, 0.5, 0)
        self.previous_display = composite
        return composite

    @property
    def mirror_image(self):
        """Flip image around y-axis"""
        return cv2.flip(self.composite_image, 1)

    #@property
    #def cropped_image(self):
    #    img = self.mirror_image
    #    img_copy = img.copy()
    #    cv2.copyMakeBorder(
    #        img_copy, img, 
    #        10, 10, 10, 10, 
    #        cv2.BORDER_CONSTANT, None, [255, 255, 255]) 
    #    return img

    @property
    def im_size(self):
        """The size of the read image"""
        return self.stream.get(3), self.stream.get(4)

    def imshow(self, name):
        """Display image in the named window"""
        cv2.imshow(name, self.mirror_image)

    def stop(self):
        """Close the camera connections"""
        self.stream.release()

class WindowForProjector:
    """Configure a fullscreen cv2 window"""

    def __init__(self, im_size, name="portal"):
        self.name = name
        cv2.namedWindow(self.name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(
            self.name,
            cv2.WND_PROP_FULLSCREEN,
            cv2.WINDOW_FULLSCREEN)

    def close(self):
        cv2.destroyAllWindows()

if __name__=="__main__":
    # Connect with camera
    camera = Camera((0, 1))

    # Configure the projection window
    window = WindowForProjector(camera.im_size)

    # Run until shutdown
    while True:
        # Project current image
        camera.imshow(window.name)
 
        # Clean exit - press q
        if cv2.waitKey(1) & 0xFF == ord('q'):
            camera.stop()
            window.close()
            break
