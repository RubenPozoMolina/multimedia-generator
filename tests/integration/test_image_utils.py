from utils.image_utils import ImageUtils, logger


class TestImageUtils:

    def test_image_utils(self):
        image_utils = ImageUtils()
        for i in range(20):
            logger.info(f"Generating test image {i+1} of 20")
            image_utils.text_to_image("A test image")