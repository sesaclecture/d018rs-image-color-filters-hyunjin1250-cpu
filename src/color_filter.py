import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt


class Filters:
    # TODO: Image kernels


    Kernels = {"Original" : np.reshape([0,0,0,0,1,0,0,0,0], (3, 3)),
               "blur" : np.array(np.ones((3,3)), dtype=np.float32)/9 ,
               "Gaussian blur" : np.reshape([1,2,1,2,4,2,1,2,1], (3,3))/16,
               "sharpen" : np.reshape([0,-1,0,-1,5,-1,0,-1,0], (3,3)),
               "sobel (x)" : np.reshape([-1,0,1,-2,0,2,-1,0,1], (3,3)),
               "sobel (y)" : np.reshape([-1,-2,-1,0,0,0,1,2,1], (3,3)),
               "edge Detection" : np.reshape([-1,-1,-1,-1,8,-1,-1,-1,-1], (3,3)),
               "emboss" : np.reshape([-2,-1,0,-1,1,1,0,1,2], (3,3))
    }

    def __init__(self, kernels=Kernels):
        self.kernels = kernels

    def apply_filter(self, image, filter_name) -> np.array:
        kernel = self.kernels[filter_name]
        return cv2.filter2D(image, -1, kernel)

    def get_filter_names(self) -> list[str]:
        return list(self.kernels.keys())


if __name__ == "__main__":
    # Show usage
    if len(sys.argv) < 2:
        print(f"{sys.argv[0]} <image_file>")
        exit(1)

    # Filters
    f = Filters()

    # Load image as RGB
    img = cv2.imread(sys.argv[1])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Apply kernels
    plt.figure()
    for i, filter_name in enumerate(f.get_filter_names()):
        filtered = f.apply_filter(img, filter_name)

        plt.subplot(3, 3, i+1)
        plt.imshow(filtered)
        plt.title(filter_name)
        plt.axis("off")
    plt.tight_layout()
    plt.show()
