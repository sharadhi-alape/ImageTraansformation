import cv2
import os
from PIL import Image, ImageOps, ImageFilter
import numpy

#Dynamic Threshold Setting for Edge Detection
def get_auto_edge(image, sigma = 0.33):

    v = numpy.median(image)
    l = int(max(0, (1.0 - sigma) * v))
    u = int(min(255, (1.0 + sigma) * v))
    return (l, u)


#****User Input Required for the Images***
input_path = input('Provide the path to the file containing the images. Use the "/" separator without "/" at the end.\n')
input_path = input_path + '/'

#print(input_path)


arr = os.listdir(input_path)
matches_1 = [match for match in arr if "_input" in match]

#If there is an exclusive folder with all of the input images, replace the previous command with the following one
#matches_1 = arr

#***User Input required for a file to contain the Output Images***
op_file_name = input('Provide the name of the output file you want the images in.\n')
loc = os.path.join(input_path, op_file_name)
os.mkdir(loc)
print("Output images are available in '% s'" % loc)
#Image Processing for all of the input images
for i in range(0, len(matches_1)):
    image_path = input_path+matches_1[i]
    #print(image_path)
    img_cv = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    img_pil = Image.open(image_path)

    #Conversion of the original image to grayscale
    img_gray = ImageOps.grayscale(img_pil)

    #Conversion of the grayscale image to the colors blue and orange
    img_changed_from_gray = ImageOps.colorize(img_gray, black="blue", white="orange")

    #Conversion of the PIL image to opencv image
    opencv_image = cv2.cvtColor(numpy.array(img_changed_from_gray), cv2.COLOR_RGB2BGR)

    #Conversion of the colored image to grayscale for Edge Detection
    opencv_image_gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)

    #Blurring the image for better edge detection
    opencv_image_gray = cv2.medianBlur(opencv_image_gray, 3)

    #Obtaining the limits for the pixel differences for edge detection
    t1, t2 = get_auto_edge(opencv_image_gray)

    # Canny Edge Detection
    edges = cv2.Canny(image=opencv_image_gray, threshold1 = t1, threshold2 = t2)

    #Converting the grayscale image to BGR format
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    #Combining the transformed image with the detected edges
    dst = cv2.addWeighted(opencv_image,0.8,edges,0.3,0)

    #Naming the output file
    ind_op = matches_1[i].replace("_input", "_output")
    #Alternate name for output files
    #ind_op = Output_" + i
    ind_op_path = loc  + '/' + ind_op

    #Copying the transformed image to the system
    cv2.imwrite(ind_op_path, dst)
