


""" 
Ce script représente la définition de la classe qui contient tous les méthodes utilisés pour résoudre le TP
"""

############################# Importation des bibliothèques #######################

import numpy as np
import cv2
from scipy import ndimage
import matplotlib.pyplot as plt
import sys
import scipy.signal as sig
from skimage.color import rgb2gray
from scipy.spatial import distance
from scipy.ndimage import convolve
from math import sqrt
#from main import pid_cord2


class Points_d_interets :

    def __init__ (self, IMAGE) : 
        """ taking an RGB image opened by opencv library and convert it to a gray image """
        self.I = np.float32(rgb2gray(IMAGE))       # convert to Gray image    
        self.img = np.copy(IMAGE)                             # create a copy of our image , it will be used to put points of inerest on the image                     
        

    def gradient (self) : 

        Ix  = np.gradient(self.I, axis = 1)
        Iy = np.gradient(self.I, axis = 0)
        Ixx = Ix**2                              # Ix**2
        Iyy = Iy**2                               # Iy**2   
        Ixy = Ix * Iy
        return Ixx, Iyy, Ixy


    def gauss(self, l=3, sig=3.):
        """\
        creates gaussian kernel with side length `l` and a sigma of `sig`
        """
        ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
        gauss = np.exp(-0.5 * np.square(ax) / np.square(sig))
        kernel = np.outer(gauss, gauss)
        return np.array(kernel / np.sum(kernel))


    def harris_detector (self, window) :
        """ l'idée est de donner à cette méthode le nom de la fentre souhaitée puis elle appliquera la détéction des coins """
        
        Ixx, Iyy, Ixy = self.gradient()
        #img_result = np.copy(self.img)

        if window == 'Gaussiène' : 
            img_result = np.copy(self.img)
            mask = self.gauss()
            Sxx = sig.convolve2d(Ixx,mask , mode = 'same')
            Syy = sig.convolve2d(Iyy,mask, mode = 'same')
            Sxy = sig.convolve2d(Ixy,mask, mode = 'same')
            detM = Sxx * Syy - Sxy**2
            traceM = Sxx + Syy
            C = detM - 0.05 * (traceM ** 2)
            
            for row , response in enumerate (C) : 
                for col , r in enumerate (response) : 
                    if r > 1e-5 :            # it is a coin 
                       img_result[row,col] = [255, 0,0]           # set the point of interest to red color
                       
                    else  : 
                        pass                                       # Other usecases are used to detect edges ( c > 0) or homogenous region (c = 0)

            return img_result, C
            
        elif window == "réctangle" : 
            img_result = np.copy(self.img)
            mask = np.ones((3,3), dtype="uint8")

            Sxx = sig.convolve2d(Ixx, mask, mode = 'same')
            Syy = sig.convolve2d(Iyy, mask, mode ='same')
            Sxy = sig.convolve2d(Ixy, mask, mode ='same')
            det = (Sxx * Syy) - (Sxy**2)
            trace = Sxx + Syy
            C = det - 0.05*(trace**2)
           
            for row , response in enumerate (C) : 
                for col , r in enumerate (response) : 
                    if r > 1e-3 :            # it is a coin 
                        img_result[row,col] = [255, 0,0]           # set the point of interest to red color
            else  : 
                pass                                               # Other usecases are used to detect edges ( c > 0) or homogenous region (c = 0)  

            return img_result, C
        
        else : 
            print("Vous êtes trompé de fentres, merci de choisir Gaussiène ou réctangle")
            sys.exit()   


    def harris_by_cv2 (self,k) : 

        img_result = np.copy(self.img)
        dst = cv2.cornerHarris(self.I,2,3,k)
        number_corners = np.sum(dst>0.01*dst.max())                   # compute the number of corners detected by Harris
        #result is dilated for marking the corners, not important
        dst = cv2.dilate(dst,None)
        # Threshold for an optimal value, it may vary depending on the image.
        img_result [dst>0.01*dst.max()]=[0,0,255]
        return img_result, number_corners
    

    def harris_by_cv2_k (self,k_values) : 
        """  k_values : an array of multiple k values
             return a matrix of images 
             return the number of corner detected for any k values

        """
        tab_of_figure = [[]]
        number_corners = []
        for k in k_values : 
            tab_of_figure.append(self.harris_by_cv2(k)[0])
            number_corners.append(self.harris_by_cv2(k)[1])

        return np.array((tab_of_figure)) , np.array(number_corners)
    

    def plot_image (self, image_to_show,title) : 
        # This function is used to show  images 
        plt.figure (3)
        plt.title(title)
        plt.imshow(image_to_show)
        plt.show()


    def plot_k_impact (self, k_values) : 
        """ k_values : an array of k values
            tab_of_images : a matrix of images to plot 
        """

        #for i in range (1,len(table_of_images) - 1) :
        #    plt.imshow(table_of_images[i])
        #    plt.title(f"k = {tab[i]} ")
        #    plt.show()

        plt.figure (2)
        plt.plot(k_values, self.harris_by_cv2_k(k_values)[1])
        plt.title ("Le nombre de points d'intéret détéctées pour chaque valeur de k")
        plt.xlabel ('k')
        plt.ylabel("nbr de points d'intéret")
        plt.grid()
        plt.show()


    def compare_methods (self, image1, image2, image3, title1, title2, title3) : 

        fig = plt.figure( figsize = (10,8))
        fig.add_subplot(1,3,1)
        plt.imshow(image1)
        plt.axis('off')
        plt.title(title1)
        #plt.show()

        fig.add_subplot(1,3,2)
        plt.imshow(image2)
        plt.axis('off')
        plt.title(title2)
        #plt.show()

        fig.add_subplot(1,3,3)
        plt.imshow(image3)
        plt.axis('off')
        plt.title(title3)
        plt.show()

    
    def suppression_of_non_maximas (self, C) : 
        img_result = np.copy(self.img)
        n, m = C.shape[0], C.shape[1]
        nbr = 0
        tab = []
        for i in range (1,n-1) : 
            for j in range (1,m-1) :
                #if C[i,j] < C[i-1, j-1] or C[i,j] < C[i+1, j+1] or C[i,j] < C[i+1, j-1] or C[i,j] < C[i-1, j+1] or C[i,j] < C[i, j+1] or C[i,j] < C[i, j-1] or C[i,j] < C[i+1, j] or C[i,j] < C[i-1, j] : 
                if C[i,j] < max([C[i-1, j-1], C[i+1, j+1], C[i+1, j-1], C[i-1, j+1], C[i, j+1], C[i, j-1], C[i+1, j], C[i-1, j]]) :
                    C[i,j] = 0
        

        for row , response in enumerate (C) : 
            for col , r in enumerate (response) : 
                if r > 0.00005 :            # it is a coin 
                    img_result[row,col] = [255, 0,0]           # set the point of interest to red color
                    nbr = nbr + 1
                    tab.append((row,col))
        
        return tab, nbr, img_result
        

    def plot_harris_suppression_non_maximas (self) : 

        C = self.harris_detector("Gaussiène")[1]
        fig = plt.figure( figsize = (8,5))
        # fig.add_subplot(1,2,1)
        plt.imshow(self.suppression_of_non_maximas(self.harris_detector("Gaussiène")[1])[2])
        plt.axis('off')
        plt.title(f"nombre de points d'intérets = {self.suppression_of_non_maximas(C)[1]} - Suppression des non maximas HARRIS")
        
        # fig.add_subplot(1,2,2)
        plt.figure (2)
        plt.imshow(self.harris_by_cv2(0.05)[0])
        plt.axis('off')
        plt.title(f"nombre de points d'intérets = {self.harris_by_cv2(0.05)[1]} - Harris" )
        plt.show()
        
        
   

    def cv2_fast_detector(self,t = 50, n = 2):
        copy = np.copy(self.img)
        fast = cv2.FastFeatureDetector_create(t,False,n)
        # find and draw the keypoints
        kp = fast.detect(copy,None)
        img2 = cv2.drawKeypoints(copy, kp, None, color=(255,0,0))
        
        return cv2.drawKeypoints(copy, kp, None, color=(255,0,0)), len(kp)


    def fast_detector(self, n = 9, t = 60 ):

        t = t/255                                   # Seuil
        img_gray =  np.copy(self.I)
        im = np.copy(self.img)
        height = img_gray.shape[0]    
        width = img_gray.shape[1]
        C = np.zeros((height,width))
        # coordonnées relatives des voisins
        dx = np.array([-3,-3,-2,-1,0,1,2,3,3,3,2,1,0,-1,-2,-3])
        dy = np.array([0,1,2,3,3,3,2,1,0,-1,-2,-3,-3,-3,-2,-1])
        I = [0]*16
        x = np.arange(3,height-3,1)
        y = np.arange(3,width-3,1)
        # grille des coordonnées
        yy , xx = np.meshgrid(y,x)
        xx, yy = xx.flatten().reshape(-1,1), yy.flatten().reshape(-1,1)
        # coordonnées linéarisées
        idxN = xx * width + yy
        idxV = dx*width + dy
        idx = idxN + idxV
        imgN = img_gray.flatten()[idx]
        imgP = (img_gray[3:-3, 3:-3].flatten()).reshape(-1,1)
        d1,d2= np.where(imgN > imgP + t,1,0), np.where(imgN < imgP - t,1,0)
        # mask pour convolution
        mask = np.ones((1,n))
        # nbr de pixel successifs 
        d1C = convolve(d1, mask, mode="wrap")
        d2C = convolve(d2, mask, mode="wrap")
        # 1 si nous avons n pixel successifs qui vérifient la condition
        d1 = np.where(d1C==n,1,0)
        d1 = np.sum(d1,axis = 1)
        d2 = np.where(d2C==n,1,0)
        d2 = np.sum(d2,axis = 1)
        # extraire les points d'interets
        for i in range (len(d1)):
            if d1[i]>0 or d2[i]> 0:  
                rx = (np.ones((16,1))*xx[i]+dx).astype(dtype = 'int64')[0]
                ry = (np.ones((16,1))*yy[i]+dy).astype(dtype = 'int64')[0]
                for j in range (16):
                    I[j] = img_gray[rx[j],ry[j]]
                C[xx[i],yy[i]] = abs(np.sum(img_gray[xx[i],yy[i]] - I))   
                
                
                im[xx[i],yy[i]] = [255,0,0]
                        
        
        return im, C


     
    def suppression_of_non_maximas_fast (self, C) : 
        img_result = np.copy(self.img)
        n, m = C.shape[0], C.shape[1]
        nbr = 0
        for i in range (1,n-1) : 
            for j in range (1,m-1) :
                if C[i,j] < max([C[i-1, j-1], C[i+1, j+1], C[i+1, j-1], C[i-1, j+1], C[i, j+1], C[i, j-1], C[i+1, j], C[i-1, j]]) :
                    C[i,j] = 0
        
        cord = []
        for row , response in enumerate (C) : 
            for col , r in enumerate (response) : 
                if r > 3.7 :            # it is a coin 
                    img_result[row,col] = [255, 0,0]           # set the point of interest to red color
                    nbr = nbr + 1
                    cord.append((row,col)) 
        
        return cord,img_result, nbr

    
    def simple_descriptor (self,n) : 

        img_copy = np.copy(self.I)
        height , width = img_copy.shape[0], img_copy.shape[1]
        neighboors = []
        
        #pdi_cord = np.array(self.suppression_of_non_maximas (self.harris_detector ('Gaussiène')[1])[0])
        pdi_cord = np.array(self. suppression_of_non_maximas_fast (self.fast_detector(9,60)[1])[0])
        for i in range (len(pdi_cord)) : 
            ligne , col = pdi_cord[i]
            h_min , h_max, w_min, w_max = (ligne - n) , (ligne + n + 1) , (col - n ), (n + col + 1)
            if (ligne - n > 0) and (ligne + n < height-1) and (col - n > 0) and (col + n < width-1) : 
                neighboors.append(img_copy[h_min : h_max , w_min : w_max ].flatten('K'))

            else : 
                 pass

        simple_descriptor = np.array(neighboors)
        return simple_descriptor.T




    def matching_blocs (self , descriptor1, descriptor2, pid_cord2) : 

        points_of_matching = []
        p1 = []
        p2 = []
        pid_cord = np.array(self. suppression_of_non_maximas_fast (self.fast_detector()[1])[0])

        self.D = []
        
        for i in range (descriptor1.shape[1]) : 
            dist = []
            for j in range (descriptor2.shape[1]):
                #dist.append(np.linalg.norm(descriptor1[:,i] - descriptor2[:,j]))
                dist.append(distance.euclidean(descriptor1[:,i] , descriptor2[:,j]))

            dist = np.array(dist,dtype=np.float32)
            pos_min = np.argmin(dist)
            d1 = dist[pos_min]

            dist[pos_min] = np.inf
            
            pos_min2 = np.argmin(dist)
            d2 = dist[pos_min2]
            
            
            if d1/d2 < 0.45 : 
                self.D.append(d1/d2)
                points_of_matching.append((pid_cord[i], pid_cord2[pos_min]))
                p1.append(pid_cord[i])
                p2.append(pid_cord2[pos_min])
  
        return np.array(points_of_matching), np.array(p1), np.array(p2)



    def plot_bloc_matching (self, P2, descriptor1, descriptor2, pid_cord2) : 

        img_originalle = np.copy(self.img)
        P3 = np.copy((P2))
        points_of_matching, p1, p2 = self.matching_blocs(descriptor1, descriptor2, pid_cord2)
        output_img = np.concatenate((img_originalle, P3), axis = 1)
        offset = [img_originalle.shape[1],0]
        img_result = np.copy(output_img)
        plt.figure (figsize=(10,12))
        for n in range (len(points_of_matching)) : 
            cord_p1 = np.array(points_of_matching[n][0])[::-1] 
            cord_p2 = np.array(points_of_matching[n][1])[::-1]
            cv2.circle (img_result, cord_p1, 3, (255,0,0), 3)
            cv2.circle (img_result, cord_p2 + offset, 3, (255,0,0), 3)
            cv2.line(img_result, (int(cord_p1[0]),int(cord_p1[1]) ), (int(cord_p2[0]) + offset[0], int(cord_p2[1])), (0, 255, 255), 1)
        plt.imshow(img_result)
        plt.show()
        
    def Mat_A(self,points_source, points_dist):
        num_points = points_source.shape[0]
        A = []
        for i in range(num_points):
            xs, ys = points_source[i][0], points_source[i][1]
            xt, yt = points_dist[i][0], points_dist[i][1]
            Ai =  np.array([
                [xs,ys,1,0, 0, 0, -xt*xs, -xt*ys, -xt],
                [0, 0, 0,xs,ys,1, -yt*xs, -yt*ys, -yt]])
            A.append(Ai)
            
        return np.concatenate(A, axis=0)
    
    def find_homography(self,points_source, points_target):
        A  = self.Mat_A(points_source, points_target)
        u, s, vh = np.linalg.svd(A, full_matrices=True)
        # Solution to H is the last column of V, or last row of V transpose
        L = vh[-1,:]/vh[-1,-1]
        homography = L.reshape((3,3))
        return homography
    
    def panorama(self, image):
        imgs = [self.img, image]
        stitchy=cv2.Stitcher.create()
        (dummy,output)=stitchy.stitch(imgs)
        plt.imshow(output)
        plt.show()
    
    def normalisation(self,points_source, points_target):
        A_norm = []
        #n = len(points_source[:][0])
        n = 5
        xs, ys = points_source[:][0], points_source[:][1]
        xt, yt = points_target[:][0], points_target[:][1]
        moy_xs, moy_ys = np.mean(points_source, axis = 0)
        moy_xt, moy_yt = np.mean(points_target, axis = 0)
        sum1 = np.sum((xs-moy_xs)**2 + (ys-moy_ys)**2)
        sum2 = np.sum((xt-moy_xt)**2 + (yt-moy_yt)**2)
        s1 = int((sqrt(2)*n)*np.sqrt(sum1))
        s2 = int((sqrt(2)*n)*np.sqrt(sum2))
        # T1 = np.array(s1*[[1,0,-moy_xs],[0,1,-moy_ys],[0,0,1/s1]])
        # T2 = np.array(s2*[[1,0,-moy_xt],[0,1,-moy_yt],[0,0,1/s2]])
        T1 = s1* np.vstack(([1,0,-moy_xs], [0,1,-moy_ys], [0,0,1/s1]))
        T2 = s2 * np.vstack(([1,0,-moy_xt], [0,1,-moy_yt], [0,0,1/s2]))
        ps, pt = np.vstack((points_source.T, np.ones((1,n)))), np.vstack((points_target.T, np.ones((1,n))))
        
        ps = T1.dot(ps)
        pt = T2.dot(pt)
        A_norm = np.vstack(([np.zeros((1,3)), ps[:,0].T, ps[1,0]*pt[:,0].T], [pt[:,0], np.zeros((1,3)), -ps[0,0]*pt[:,0].T]))
        for i in range(1,n):
            Ai = np.vstack(([np.zeros((1,3)), ps[:,i].T, ps[1,i]*pt[:,i].T], [pt[:,i], np.zeros((1,3)), -ps[0,i]*pt[:,i].T]))
            A_norm = np.vstack((A_norm, Ai))

        u, s, vh = np.linalg.svd(A_norm, full_matrices=True)
        L = vh[-1,:]/vh[-1,-1]
        homography = L.reshape((3,3))
        return homography
        
                


        
 










