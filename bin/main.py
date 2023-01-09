


""" Ce Fichier est le main """

########### Import libraries #############
from tkinter import N
import warnings
import cv2
from points_d_interets import * 
from skimage.io import imread
import scipy.signal as sig
import os
from skimage.transform import rotate
import pathlib

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 

path2 = str(pathlib.Path(__file__).parent.resolve())
actual_path = path2 [: -4]

cible_path = '\pics\P1.jpg'
cible_path2 = '\pics\P2.jpg'
path = actual_path + cible_path

PATH2 = actual_path + cible_path2
n = 3


# read the images 

#P1 = imread('C:/Users/33660/Desktop/Etudes/Master_2/SEMESTRE_1/TRAITEMENT_DES_IMAGES/TP/TP1/Point_d_interets/pics/P1.jpg')
P1 = imread(path)
P2 = imread(actual_path + cible_path2)
# cv2.imshow ('P2', P2)
# cv2.waitKey(0)

I_rotate = rotate(np.copy(P1), 45)

PI = Points_d_interets (P1)    
PI_2 = Points_d_interets (P2)        
# PI_2.plot_image(P2, 'image_2')                                    # Instanciate the class              
# Ixx, Iyy, Ixy = PI.gradient()                                              # compute gradient  
k_values = np.array(np.arange (0.04 , 0.06 , 0.002))

# image_harris, C2 = PI.harris_detector('réctangle')                             # Harris detector by rectangular window
#image_harris_gauss, C2 = PI.harris_detector('Gaussiène')                        # Harris detector by gaussian window  
# harris_cv2, nbr = PI.harris_by_cv2(0.05)                                    # Harris already implemented by cv2                  


P_rotate = Points_d_interets(I_rotate)
#image_harris_rotate, C3 = P_rotate.harris_detector('réctangle')                             # Harris detector by rectangular window
# image_harris_gauss_rotate, C4 = P_rotate.harris_detector('Gaussiène')                        # Harris detector by gaussian window  
image_fast, Cf = PI.fast_detector(9)

# # ----------------- Plots -----------------#
# PI.plot_image(image_harris,'image_harris_réctangle_window')
#PI.plot_image(image_harris_gauss, 'harris_gaussiene')
# PI.plot_image(harris_cv2, 'harris_cv2')
# PI.plot_k_impact(k_values)
#PI.compare_methods (image_harris, image_harris_gauss, harris_cv2, "harris_réctangle", "harris_gaussiènne", "harris_cv2")

#PI.compare_methods (image_harris, image_harris_rotate, image_harris_gauss_rotate, "image_originale", "harris_réctangle", "harris_gaussiène")
# image_fastcv2 , nbr = PI.cv2_fast_detector()
#PI.plot_image(image_fastcv2,"fast_detector_cv2")
PI.plot_image(image_fast, 'fast_detector')
PI.suppression_of_non_maximas_fast(Cf)
#pi_cord, nbr, harris_maximas = PI.suppression_of_non_maximas(C2)
# PI.plot_harris_suppression_non_maximas ()
descriptor = PI.simple_descriptor (n) 
descriptor2 = PI_2.simple_descriptor(n)
#print(descriptor2.shape)

# pid_cord2 = np.array(PI_2.suppression_of_non_maximas(PI_2.harris_detector('Gaussiène')[1])[0])
pid_cord2, img_fast_suppression, nbr_fast = np.array(PI_2. suppression_of_non_maximas_fast (PI_2.fast_detector()[1]))
# pid2, img_fast_suppression2, nbr_fast2 = np.array(PI. suppression_of_non_maximas_fast (Cf))


# PI.plot_image (img_fast_suppression2, f"Nombres de points d'interets avec suppression = {nbr_fast2}")

points_of_matching, p1, p2 = PI.matching_blocs(descriptor, descriptor2, pid_cord2)

# print (points_of_matching.shape)

# PI.plot_bloc_matching(P2,descriptor, descriptor2, pid_cord2)


 #------------------ Homography--------------#
#h, status = cv2.findHomography(p1, p2, cv2.RANSAC, 5.0)
#print("h", h)
#h1 = PI.find_homography(p1,p2)
#print("h1", h1)
#im_out = cv2.warpPerspective(P2, h1, (P2.shape[1],P2.shape[0]))
#cv2.imshow("Warped Image", im_out)
#cv2.waitKey(0)
