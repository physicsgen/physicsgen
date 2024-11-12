import cv2
import os
import pandas as pd
import argparse
import numpy as np
from matplotlib import pyplot as plt
import math

def CheckGround(sample, IMAGE_SIZE, DEBUG=False):
    # The aim is to recognise the ground using line detection implemented with a Canny filter
    # To ensure that the correct line is selected as ground surface in the entire image, 
    # the threshold of the line detection must first be selected correctly and the lines sorted using logic. 
    
    # Copy of the input sample for debug figures
    if DEBUG: In_DebugImg = sample.copy()

    # Define colors interval for the colors belonging to the ground 
    lower_white = np.array([200, 200, 200])
    upper_gray = np.array([255, 255, 255])

    # Create mask containing the white and grey areas of the input image
    mask = cv2.inRange(sample, lower_white, upper_gray)
    if DEBUG: cv2.imshow("ground mask", ~mask)

    SearchTrial = 0
    if IMAGE_SIZE == 64:
        Start_LineThreshold = 29
    elif IMAGE_SIZE == 128:
        Start_LineThreshold = 73
    elif IMAGE_SIZE == 256:
        Start_LineThreshold = 120
    Ground_NotFound = True
    while (SearchTrial < 3 and Ground_NotFound):
        # Three search passes are permitted, in which the threshold for the line detection is reduced progressively 
        # if no ground surface could be detected in the previous pass.

        ##-- Edge detection with Canny filter
        edges = cv2.Canny(~mask, 0, 50)
        if DEBUG: cv2.imshow("Canny filter", edges)

        ## Hough line transformation: Line detection
        # The result of this transformation is a slope angle and the distance from the crossing of the Y-axis 
        # to the upper edge of the image for each line. If necessary, the threshold for the line detection is reduced here.
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=Start_LineThreshold - SearchTrial*20)

        
        ##-- Search for the line representing the ground surface
        Idx_GroundLine = -1
        Line_dist = -100
        count_line = 1
        if lines is not None and len(lines) > 0:
            for line in lines:
                rho, theta = line[0]
                theta_grad = theta*180/math.pi
                if theta != 0 and (theta_grad < 87.8 or theta_grad > 92.2):
                    # An angle corresponding to a slope of minimum +/- 2,2 degrees is required so that the line 
                    # is considered plausible for the ground surface.
                    if theta_grad - 90 > 0 and theta_grad - 90 < 27:
                        # Additionally the angle is limited to the interval in which the ground inclination lies
                        if Line_dist < rho:
                            # Finally, the line with the greatest distance to the upper edge of the image is searched for
                            Line_dist = rho
                            Idx_GroundLine = count_line - 1
                
                if DEBUG: print(f"Line {count_line}: Distance: {rho}, Angle: {theta_grad}")
                count_line += 1


            ##-- Start next search run or refine the search
            if Idx_GroundLine == -1:
                # No line matching the criteria was found upper. The next search run is therefore started.
                Ground_NotFound = True
                if DEBUG: print("No ground surface recognised")
            else:
                # A possible ground line was found: a final check is carried out to exclude any error.
                # Sometimes there are images with two almost identical ground lines. In this case, the steepest ground line 
                # must always be used, as this necessarily follows the ground contour, as no "plausible" line can penetrate 
                # the ground. A tolerance band of 4px is therefore placed around the largest distance in which the steepest line 
                # is taken.
                Ground_NotFound = False
                if DEBUG: print("Idx ground line", Idx_GroundLine)

                # Line with plausible angle and biggest distance to the upper edge of the image
                BestRho, BestTheta = lines[Idx_GroundLine][0]
                BestTheta_grad = BestTheta * 180 / math.pi

                # Search for a ground line with a similar distance, but which would be steeper.
                Idx_line = 0
                for line in lines:
                    rho, theta = line[0]
                    theta_grad = theta * 180 / math.pi
                    if rho <= BestRho and rho >= BestRho - 4:
                        # The distance of the possible lines cannot be more than 4px smaller than the biggest measured distance
                        if theta_grad > BestTheta_grad:
                            # If a line has a bigger angle in this tolerance band of 4px, it is considered a new floor surface
                            Idx_GroundLine = Idx_line
                            if DEBUG: print("Better ground line found: ", Idx_GroundLine)
                    
                    Idx_line += 1
        
        SearchTrial += 1    # 3 search runs are possible
    

    ##-- Get ground slope from final detected line --##
    if Idx_GroundLine == -1:
        # No ground slope can be determined 
        Ground_Incli = -1000
        BestRho = 1000; BestTheta = 1000
        print("No ground slope can be measured")
    else:
        BestRho, BestTheta = lines[Idx_GroundLine][0]
        Ground_Incli = BestTheta * 180 / math.pi - 90
        print("Ground slope: ", Ground_Incli)


    ##-- Draw found ground lines
    count_line = 0       
    print("Number of search runs:", SearchTrial)             
    if DEBUG and not(Ground_NotFound):
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            theta_grad = theta*180/math.pi
            if Idx_GroundLine == count_line:
                cv2.line(In_DebugImg, (x1, y1), (x2, y2), (0, 0, 255), 2)
            else:
                if theta != 0 and (theta_grad < 89 or theta_grad > 91):
                    cv2.line(In_DebugImg, (x1, y1), (x2, y2), (255, 0, 0), 1)
                elif DEBUG:
                    cv2.line(In_DebugImg, (x1, y1), (x2, y2), (255, 0, 255), 1)
            count_line += 1

        # Show the result lines 
        cv2.imshow("Ground line", In_DebugImg)

    return Ground_Incli, BestRho, BestTheta 

def CheckBall(sample, IMAGE_SIZE, DEBUG=False, Ground_Rho=1000, Ground_Theta=1000):
    ###--- Detection of the ball (position, rotation, roundness) ---###
    CombinedMask = sample.copy()

    ##-- Detect blue half --##
    # blue color: G0 B255 R0
    blue_low = np.array([120, 0, 0])
    blue_high = np.array([255, 150, 100])
    blue_mask = cv2.inRange(sample, blue_low, blue_high)
    # blurred = cv2.GaussianBlur(blue_mask, (1, 1), 0)

    # Identify contours in the blue mask
    BlueContours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if DEBUG: print("Number of blue contours:", len(BlueContours))
    if DEBUG and False: cv2.drawContours(sample, BlueContours, -1, (0, 255, 0), 1)    #green color
    if DEBUG: cv2.imshow("Blaue Maske", blue_mask)

    if DEBUG:
        # Sum up all points of the blue mask
        blue_px_pos = []
        height, width = blue_mask.shape
        for y in range(height):
            for x in range(width):
                if blue_mask[y, x] > 0:
                    blue_px_pos.append((x, y))
        print("Number of blue pixel:", len(blue_px_pos))

    ##-- Detect black half --##

    # black color
    black_low = np.array([0, 0, 0])
    black_high = np.array([100, 110, 90])
    black_mask = cv2.inRange(sample, black_low, black_high)

    # Identify contours in the black mask
    BlackContours, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if DEBUG: print("Number of black contours:", len(BlackContours))
    if DEBUG: cv2.drawContours(sample, BlackContours, -1, (0, 0, 255), 1)    #red color
    if DEBUG: cv2.imshow("Black mask:", black_mask)

    if DEBUG: cv2.imshow("Input sample", sample)


    ##-- Filter all contours if more than 1 contour found --##
    NoBallCheckPossible = False
    # If several contours have been found:
    # filter out the ball contours from the blue and black mask contours which habe been found
    if BlueContours:
        # If several contours have been detected, keep the contour that has the most points
        Ball_BlueContour_Nb = 0
        Contour_MaxPts = 0
        counter = 0
        for c in BlueContours:
            contour_coord = c.ravel()
            if Contour_MaxPts < len(contour_coord[::2]):
                Contour_MaxPts = len(contour_coord[::2])
                Ball_BlueContour_Nb = counter
            counter += 1
    else:
        # No contour found in the blue mask: quit the ball evaluation with error values
        NoBallCheckPossible = True
    if BlackContours:
        # If several contours have been detected, keep the contour that has the most points
        Ball_BlackContour_Nb = 0
        Contour_MaxPts = 0
        counter = 0
        for c in BlackContours:
            contour_coord = c.ravel()
            if Contour_MaxPts < len(contour_coord[::2]):
                Contour_MaxPts = len(contour_coord[::2])
                Ball_BlackContour_Nb = counter
            counter += 1
    else:
        # No contour found in the black mask: quit the ball evaluation with error values
        NoBallCheckPossible = True


    if not(NoBallCheckPossible):
        ###------------------------------###
        ###--- Identify ball rotation ---###
        ###------------------------------###

        # Compare the two separation lines of the ball halves with each other and determine the slope: this provides the rotation of the ball.
        KonturHaelfte1 = BlueContours[Ball_BlueContour_Nb]
        KonturHaelfte2 = BlackContours[Ball_BlackContour_Nb]
        if DEBUG: print("Number of points blue contour:", len(KonturHaelfte1))
        if DEBUG: print("Number of points black contour:", len(KonturHaelfte2))

        ##-- Define center of the black ball half --##
        XCoords_black = []
        YCoords_black = []
        XMin_black = 1000
        YMin_black = 0
        XMax_black = 0
        YMax_black = 0
        for t in KonturHaelfte2:
            XCoords_black.append(t[0][0])
            YCoords_black.append(t[0][1])
            if XMin_black > t[0][0]:
                XMin_black = t[0][0]
                YMin_black = t[0][1]
            if XMax_black < t[0][0]:
                XMax_black = t[0][0]
                YMax_black = t[0][1]
        XMean_black = np.mean(XCoords_black)
        YMean_black = np.mean(YCoords_black)
        if DEBUG: cv2.circle(CombinedMask, (int(XMean_black), int(YMean_black)), 3, (0, 255, 255), -1)  #color: yellow
        if DEBUG: print("Center of black ball half:", XMean_black, YMean_black)


        # The line that passes through the centre of the ball is filtered out of the semi-circle contours 
        # by keeping only the points of both contours that have a distance (Euclidean distance) of less than 1, 2 or 3.
        MaxAllowed_Dist = 1   # maximal allowed distance between a point on the blue and black contour

        while MaxAllowed_Dist < 3:
            Pts_Blue = []
            Pts_Black = []
            for punkt1 in KonturHaelfte1:
                for punkt2 in KonturHaelfte2:
                    x1, y1 = punkt1[0]
                    x2, y2 = punkt2[0]

                    distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

                    if distance <= MaxAllowed_Dist:
                        Pts_Blue.append((x1, y1))
                        #if DEBUG: cv2.circle(sample, (int(x1), int(y1)), 1, (255, 0, 0), -1)    #color: cyan
                        Pts_Black.append((x2, y2))
                        #if DEBUG: cv2.circle(sample, (int(x2), int(y2)), 1, (0, 0, 255), -1)    #color: green
            if len(Pts_Blue) > 3 and len(Pts_Black) > 3:
                # If at least 4 common points are found between the blue and black contours for a defined maximum distance, 
                # the maximum distance is not increased any further in order to avoid adding any irrelevant points.
                break
            else:
                # If no common points are found between blue and black during the first run with a distance of 1, 
                # make another run with a distance of 2 and then of 3 if necessary.
                MaxAllowed_Dist += 1
        if DEBUG: print("Number of blue points next to black:", len(Pts_Blue))
        if DEBUG: print("Number of black points next to blue:", len(Pts_Black))


        MaxDiff_BlueBlack = 8   # max allowed difference between the identified rotation of the blue and black halves
        if len(Pts_Blue) >= 3 and len(Pts_Black) >= 3:
            Err_ColorHalf = False
            MaxDist_PtLinie = 2.3
            # If enough common points are found between the blue and black halves of the ball, 
            # fit lines along the color separation of the ball halves to determine the ball rotation

            ###--- Blue ball half ---###
            
            ##-- Fit line through the points of the blue ball half belonging to the color separation line --#
            vx_blue, vy_blue, x_blue, y_blue = cv2.fitLine(np.array(Pts_Blue), cv2.DIST_L2, 0, 0.01, 0.01)
            # vx and vy are direction vectors ; x and y the coordinates of a point on the fitted line

            ##-- Determine the line slope along the blue ball half --##
            BlueRot_slope = vy_blue / vx_blue
            if DEBUG: print("Blue line slope: ", BlueRot_slope[0])
            if BlueRot_slope == 0:
                BallRot_blue = 0
            else: 
                BallRot_blue = math.atan2(BlueRot_slope[0], 1) * 180 / math.pi
            if DEBUG: print("Ball rotation from blue points: ", BallRot_blue)
            
            ##-- Measure distances of all points belonging to the color separation line with the fitted line itself to check the max distance --##
            DirVect = np.array([vx_blue[0], vy_blue[0]])
            LinePt = np.array([x_blue[0], y_blue[0]])            

            for p in Pts_Blue:
                Pt_toCheck = np.array(p)
                Dist_Pt_Line = np.linalg.norm(np.cross(Pt_toCheck - LinePt, DirVect)) / np.linalg.norm(DirVect)
                #if DEBUG: print("Distance blue points - line: ", Dist_Pt_Line)
                if Dist_Pt_Line > MaxDist_PtLinie:
                    # A too large a distance between the fitted separation line of both halves of the ball and a point belonging to this separation line. 
                    # This can be an indication that the image does not have two clean colour halves.
                    if DEBUG: cv2.circle(sample, (int(p[0]), int(p[1])), 3, (255, 0, 0), -1)
                    if DEBUG: print("Point too far away from the separation line:", p, "Distance:", Dist_Pt_Line)
                    Err_ColorHalf = True
                    break


            ###--- Black ball half ---###
            
            ##-- Fit line through the points of the black ball half belonging to the color separation line --#
            vx_black, vy_black, x_black, y_black = cv2.fitLine(np.array(Pts_Black), cv2.DIST_L2, 0, 0.01, 0.01)
            # vx and vy are direction vectors ; x and y the coordinates of a point on the fitted line

            ##-- Determine the line slope along the black ball half --##
            slope = vy_black / vx_black
            if DEBUG: print("Black line slope: ", slope[0])
            if slope == 0:
                BallRot_black = 0
            else: 
                BallRot_black = math.atan2(slope[0], 1) * 180 / math.pi
            if DEBUG: print("Ball rotation from black points: ", BallRot_black)

            ##-- Measure distances of all points belonging to the color separation line with the fitted line itself to check the max distance --##
            DirVect = np.array([vx_black[0], vy_black[0]])
            LinePt = np.array([x_black[0], y_black[0]])

            for p in Pts_Black:
                Pt_toCheck = np.array(p)
                Dist_Pt_Line = np.linalg.norm(np.cross(Pt_toCheck - LinePt, DirVect)) / np.linalg.norm(DirVect)
                #if DEBUG: print("Distance black points - line: ", Dist_Pt_Line)
                if Dist_Pt_Line > MaxDist_PtLinie:
                    # A too large a distance between the fitted separation line of both halves of the ball and a point belonging to this separation line. 
                    # This can be an indication that the image does not have two clean colour halves.
                    if DEBUG: cv2.circle(sample, (int(p[0]), int(p[1])), 3, (0, 255, 0), -1)
                    if DEBUG: print("Point too far away from the separation line:", p, "Distance:", Dist_Pt_Line)
                    Err_ColorHalf = True
                    break

            ##-- Position of the black ball half relative to the separation line --##
            # A median line is calculated from the black and blue rotation lines which are helping to determine the ball rotation
            Rot_DirVect = np.array([vx_blue[0], vy_blue[0]])
            Rot_LinePt = np.array([x_blue[0], y_blue[0]])

            RotLine_StartPt = tuple(np.int32(Rot_LinePt - 100*Rot_DirVect))
            RotLine_EndPt = tuple(np.int32(Rot_LinePt + 100*Rot_DirVect))
            RotLine_slope = vy_blue / vx_blue
            RotLine_intercept = y_blue - (RotLine_slope * x_blue)
            cv2.line(sample, RotLine_StartPt, RotLine_EndPt, (255, 0, 0), 1)
            print("Starting point of the line:", RotLine_StartPt)
            print("End point of the line:", RotLine_EndPt)
            
            # Evaluation whether the points of the black contour are above or below the separation line
            BlackPts_underRotLine = 0
            BlackPts_overRotLine = 0
            for p in Pts_Black:
                Pt_toCheck = np.array(p)
                if p[1] > RotLine_slope[0] * p[0] + RotLine_intercept[0]:
                    BlackPts_underRotLine += 1
                else:
                    BlackPts_overRotLine += 1
         

            ###--- Define ball rotation ---###
            # The angle is calculated from the median separation line.
            # The requirement is that all the points of the fitted line are close to the separation line and 
            # that the calculated slopes of the blue and black halves do not deviate too much from each other.
            if Err_ColorHalf:
                # some points of the fitted line are too far away from the separation line: unclear color separation
                BallRotAngle = -9999
                print("Ball rotation not identified due to incorrect colour separation")
            else:
                if BlackPts_underRotLine > BlackPts_overRotLine:
                    # Starting and end point of the line used to calculate the rotation angle
                    BallRotAngle = -math.atan2((RotLine_EndPt[1]-RotLine_StartPt[1]), (RotLine_EndPt[0]-RotLine_StartPt[0])) * 180 / math.pi                    
                    print("Black ball half under rotation line.")
                    
                else:
                    # Starting and end point of the line are swapped to calculate the angle
                    BallRotAngle = -math.atan2((RotLine_StartPt[1]-RotLine_EndPt[1]), (RotLine_StartPt[0]-RotLine_EndPt[0])) * 180 / math.pi
                    print("Black ball half over rotation line.")

                if abs(BallRot_blue - BallRot_black) > MaxDiff_BlueBlack:
                    # too excessive slope difference between the blue and black ball separation line
                    # error value for the ball rotation
                    BallRotAngle = -5000
                    print("Ball rotation not determined due to an excessive slope difference between blue and black")
        
        else:
            ###--- Differentiation between rotation 0° or 90° or error ---###
            Err_Rot = False
            # If there are not enough points of the blue and black contours close to each other, the ball rotation may be 0° or 90°. 
            # To distinguish this from an error case, the distances between the contour points are calculated. 
            # If a large distance is found, it corresponds to the median separation line, where the contour does not follow the 
            # roundness of the ball. In this case, these two furthest points are taken to calculate the ball rotation.

            print("Fewer than 3 neighboring points between the blue and black halves of the ball were found.")
            
            ##-- Calculation of the distances between the contour points --##
            # For the blue ball half
            DistContPts = []
            MaxDist_tmp = 0
            for i in range(len(KonturHaelfte1)):
                # Compute distance between the contour points
                if i < (len(KonturHaelfte1) - 1):
                    dist = math.sqrt((KonturHaelfte1[i+1][0][0] - KonturHaelfte1[i][0][0])**2 + (KonturHaelfte1[i+1][0][1] - KonturHaelfte1[i][0][1])**2 )
                    DistContPts.append(dist)
                else:
                    dist = math.sqrt((KonturHaelfte1[0][0][0] - KonturHaelfte1[i][0][0])**2 + (KonturHaelfte1[0][0][1] - KonturHaelfte1[i][0][1])**2 )
                    DistContPts.append(dist)
                if dist > MaxDist_tmp:
                    MaxDist_tmp = dist
                    if i < (len(KonturHaelfte1) - 1):
                        MinPt = KonturHaelfte1[i][0]
                        MaxPt = KonturHaelfte1[i+1][0]
                    else:
                        MinPt = KonturHaelfte1[i][0]
                        MaxPt = KonturHaelfte1[0][0]
            DistContPts_tmp = sorted(DistContPts)
            if DEBUG: print("Distance list between blue points:", DistContPts_tmp)

            if DistContPts_tmp[-1] > 20 and DistContPts_tmp[-2] < 20:
                # Two points can be found that clearly have a larger distance from each other compared to the other points.
                if DEBUG: print("1st point:", MinPt)
                if DEBUG: print("2nd point: ", MaxPt)
                
                # if these two points are existing, calculate the ball rotation
                y_delta = MaxPt[1] - MinPt[1]
                x_delta = MaxPt[0] - MinPt[0]
                if y_delta == 0:
                    BallRot_blue = 0
                else: 
                    BallRot_blue = -math.atan2(y_delta, x_delta) * 180 / math.pi
                if DEBUG: print("Ball rotation from blue points: ", BallRot_blue)
            else:
                # two points corresponding to the required criteria not found: error for the ball rotation
                Err_Rot = True
            

            # For the black ball half
            if not(Err_Rot):
                DistContPts = []
                MaxDist_tmp = 0
                for i in range(len(KonturHaelfte2)):
                    # Compute distance between the contour points
                    if i < (len(KonturHaelfte2) - 1):
                        dist = math.sqrt((KonturHaelfte2[i+1][0][0] - KonturHaelfte2[i][0][0])**2 + (KonturHaelfte2[i+1][0][1] - KonturHaelfte2[i][0][1])**2 )
                        DistContPts.append(dist)
                    else:
                        dist = math.sqrt((KonturHaelfte2[0][0][0] - KonturHaelfte2[i][0][0])**2 + (KonturHaelfte2[0][0][1] - KonturHaelfte2[i][0][1])**2 )
                        DistContPts.append(dist)
                    if dist > MaxDist_tmp:
                        MaxDist_tmp = dist
                        if i < (len(KonturHaelfte2) - 1):
                            MinPt = KonturHaelfte2[i][0]
                            MaxPt = KonturHaelfte2[i+1][0]
                        else:
                            MinPt = KonturHaelfte2[i][0]
                            MaxPt = KonturHaelfte2[0][0]
                DistContPts_tmp = sorted(DistContPts)
                if DEBUG: print("Distance list between black points:", DistContPts_tmp)

                if DistContPts_tmp[-1] > 20 and DistContPts_tmp[-2] < 20:
                    # Two points can be found that clearly have a larger distance from each other compared to the other points.
                    if DEBUG: print("1. Punkt: ", MinPt)
                    if DEBUG: print("2. Punkt: ", MaxPt)
                    
                    # if these two points are existing, calculate the ball rotation
                    y_delta = MaxPt[1] - MinPt[1]
                    x_delta = MaxPt[0] - MinPt[0]
                    if y_delta == 0:
                        BallRot_black = 0
                    else: 
                        BallRot_black = -math.atan2(y_delta, x_delta) * 180 / math.pi
                    if DEBUG: print("Ball rotation from blue points: ", BallRot_black)
                else:
                    # two points corresponding to the required criteria not found: error for the ball rotation
                    Err_Rot = True
            

            ##-- Comparison of the rotation from the black and blue halves of the ball --##
            if not(Err_Rot) and abs(BallRot_blue)<=180 and abs(BallRot_black)<=180:
                if np.sign(BallRot_black) != np.sign(BallRot_blue) and \
                    (89 < abs(BallRot_blue) and abs(BallRot_blue) < 91) and \
                    (89 < abs(BallRot_black) and abs(BallRot_black) < 91):
                    # If the ball angle is almost exactly 90° or -90°, it is possible that the angle of one half is positive and the other is negative. 
                    # In this case, it is known that the rotation has an absolute value of 90°. 
                    # The angle sign will be determined later based on the position of the black half of the ball.
                    BallRotAngle = 90
                    if DEBUG: print("Ball rotation without any sign: ", BallRotAngle)
                elif abs(BallRot_blue - BallRot_black) > MaxDiff_BlueBlack:
                    # too excessive slope difference between the blue and black ball separation line
                    BallRotAngle = -5000
                    print("Ball rotation not determined due to an excessive slope difference between blue and black")
                else:
                    # If there is no rotation around +/-90°, the actual rotation is calculated as the average of both ball halves.
                    BallRotAngle = (BallRot_blue + BallRot_black) / 2
                    print("Ball rotation mean value between blue and black: ", BallRotAngle)
            else:
                # Error case: if the determined rotation values of both ball halves are not valid, return an error value
                BallRotAngle = -1001
                print("Ball rotation could not be determined.")



        ###----------------------------------------###
        ###--- Analyze whole ball: center point ---###
        ###----------------------------------------###
                
        ##-- Create whole ball mask: merge both ball halves --##
        ball_mask = cv2.add(blue_mask, black_mask)
        if DEBUG: cv2.imshow("Entire ball mask:", ball_mask)

        blurred = cv2.GaussianBlur(ball_mask, (1, 1), 0) 
        contours, _ = cv2.findContours(blurred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if DEBUG: print("Number of detected ball contours:", len(contours))

        
        ##-- Identify the amount of balls --##
        # If no or more than exactly one ball is found, no evaluation is carried out
        if DEBUG: circle_img = sample.copy()
        minRad_detect = math.floor(12 / (256 / IMAGE_SIZE))
        maxRad_detect = math.ceil(18 / (256 / IMAGE_SIZE))
        minDist_detect = math.floor(18 / (256 / IMAGE_SIZE))
        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=minDist_detect, param1=40, param2=10, minRadius=minRad_detect, maxRadius=maxRad_detect)

        # Check if circles are found on sample
        circle_count = 0
        if circles is not None:
            # Convert the coordinates and radius of the circles in integers
            circles = np.round(circles[0, :]).astype("int")

            # Draw circles
            for (x, y, r) in circles:
                if DEBUG: cv2.circle(circle_img, (x, y), r, (0, 255, 0), 2)
                circle_count += 1
            if DEBUG: cv2.imshow("Identified balls", circle_img)
        
        print(f"{circle_count} balls identified")


        if circle_count == 1:  # Continue sample evaluation when exactly one ball was found
            ##-- Selection of useful contours --##
            if contours:
                # If several contours are detected, keep the two contours that have the most points.
                BallContour_MaxPts_First_Nb = 0
                BallContour_MaxPts_Second_Nb = 0
                BallContour_MaxPts_First_Idx = 0
                BallContour_MaxPts_Second_Idx = 0
                counter = 0
                for c in contours:
                    contour_coord = c.ravel()
                    if counter == 0:
                        # first iteration of the for loop
                        BallContour_MaxPts_First_Nb = len(contour_coord[::2])
                        BallContour_MaxPts_First_Idx = counter
                    elif counter !=0 and BallContour_MaxPts_First_Nb < len(contour_coord[::2]):
                        # A longer contour has been found compared to the last one saved
                        # Write the number of points and the index of the largest contour into the second largest one.
                        BallContour_MaxPts_Second_Nb = BallContour_MaxPts_First_Nb
                        BallContour_MaxPts_Second_Idx = BallContour_MaxPts_First_Idx
                        # Write the new maximum number of points into the longest contour (with index).
                        BallContour_MaxPts_First_Nb = len(contour_coord[::2])
                        BallContour_MaxPts_First_Idx = counter
                        
                    counter += 1
                
                if DEBUG: print(f"Longest contour: {BallContour_MaxPts_First_Nb} points")
                if DEBUG: print(f"Second longest contour: {BallContour_MaxPts_Second_Nb} points")
            

            ##-- Create cleaned ball mask --##
            # Fill at most the two largest ball contours (from the uncleaned mask)
            ball_mask_cleaned = np.zeros(sample.shape[:2], dtype=np.uint8)
            
            # In every case, the largest contour is filled
            cv2.drawContours(ball_mask_cleaned, [contours[BallContour_MaxPts_First_Idx]], 0, 255, thickness=cv2.FILLED)
            # If there is a second contour that has nearly the same number of points, suggesting that the ball 
            # might have been detected in two parts, the second contour will also be included in the cleaned mask.
            if (BallContour_MaxPts_First_Nb <= BallContour_MaxPts_Second_Nb + 10) and BallContour_MaxPts_Second_Nb != 0:
                cv2.drawContours(ball_mask_cleaned, [contours[BallContour_MaxPts_Second_Idx]], 0, 255, thickness=cv2.FILLED)
            
            if DEBUG: cv2.imshow("Cleaned ball mask", ball_mask_cleaned)


            ##-- Determine center of the ball --##
            # Therefore all points of the compiled ball mask are examined. The center of the ball is derived from 
            # the smallest and largest X and Y coordinates found among these points.
            
            ball_px_pos = []
            height, width = ball_mask_cleaned.shape
            for y in range(height):
                for x in range(width):
                    # Save coordinates from all the mask points
                    if ball_mask_cleaned[y, x] > 0:
                        ball_px_pos.append((x, y))
            if DEBUG: print("Number of ball pixel: ", len(ball_px_pos))

            # Find smallest and largest X- or Y-coordinate
            XMax = 0; XMin = IMAGE_SIZE
            YMax = 0; YMin = IMAGE_SIZE
            for e in ball_px_pos:
                if e[0] >= XMax:
                    XMax = e[0]
                if e[0] <= XMin:
                    XMin = e[0]
                if e[1] >= YMax:
                    YMax = e[1]
                if e[1] <= YMin:
                    YMin = e[1]

            # Calculate the center of the ball by taking the average of the minimum and maximum coordinates.
            XMean = (XMax-XMin)/2 + XMin
            YMean = (YMax-YMin)/2 + YMin
            print("Center of the ball:", XMean, YMean)
            if DEBUG: print("Height difference between ball center and black center:", YMean-YMean_black)
            if DEBUG: cv2.circle(CombinedMask, (int(XMean), int(YMean)), 3, (0, 0, 255), -1)  #color: red
            if DEBUG: cv2.drawContours(CombinedMask, contours, -1, (0, 255, 0), 1)    #color: green

            # Calculate the distance between the ball and the ground surface based on the ball's center point.
            BallRad = 15 / (256 / IMAGE_SIZE)
            if (Ground_Rho != 1000) and (Ground_Theta != 1000):
                # Calculate the distance of the ball only when the equation of the ground surface is known.
                # Determine the ground height instead of the X-coordinate of the center point.
                YGround = (Ground_Rho - XMean*np.cos(Ground_Theta)) / np.sin(Ground_Theta)
                GroundDist = YGround - (YMean+BallRad)
                print("Distance between ball and ground:", GroundDist)
            else:
                GroundDist = -1000
            

            ###-------------------------------------------------------------###
            ###--- Define the sign of the ball rotation for the 90° case ---###
            ###-------------------------------------------------------------###

            if abs(BallRotAngle) < 180:
                # Ball rotation already computed no error case...
                if np.sign(BallRot_black) != np.sign(BallRot_blue) and \
                    (89 < abs(BallRot_blue) and abs(BallRot_blue) < 91) and \
                    (89 < abs(BallRot_black) and abs(BallRot_black) < 91):
                    # If the ball angle is almost exactly 90° or -90°, it may happen that one half of the angle is positive and the other is negative. 
                    # In this case, the sign of the rotation should be determined based on the X-position (right or left) of the black area, 
                    # rather than its Y-position (top or bottom).
                    if XMean_black > XMean:
                        # black ball half is on the right, rotation angle is positive
                        BallRotAngle = abs(BallRotAngle)
                    elif XMean_black < XMean:
                        # black ball half is on the left, rotation angle is negative
                        BallRotAngle = -abs(BallRotAngle)
                    else:
                        # In this case the ball's center and the center of the black half would be superposed in X-direction. This is not possible.
                        BallRotAngle = -1002
            print("Ball rotation: ", BallRotAngle)


            ###------------------------------------###
            ###--- Determine the ball roundness ---###
            ###------------------------------------###
            
            # Find contours of the cleaned mask    
            contours, _ = cv2.findContours(ball_mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if DEBUG: print(f"Number of contours on the cleaned mask: {len(contours)}")

            # Only one contour should be found here.
            if len(contours) == 1:
                # Extract coordinates of the contour
                contour_coordinates = contours[0].ravel()
                x_coordinates = contour_coordinates[::2]        #Split x-coordinates of the array
                y_coordinates = contour_coordinates[1::2]       #Split y-coordinates of the array
                if DEBUG: print("Number of points in the cleaned mask: ", len(x_coordinates))
                
                ##-- Center of the ball: control method --##
                # The center of the ball can alternatively be determined by averaging all point coordinates of the ball contour. 
                # However, this method is less precise because if the contour has more points on one side of the ball, 
                # averaging all points will shift the center towards that side. This second method can serve as a verification method.
                XMean_fromContour = np.mean(x_coordinates)
                YMean_fromContour = np.mean(y_coordinates)
                if DEBUG: print("Circle center from contour:", XMean_fromContour, YMean_fromContour)


                ##-- Check roundness of the ball --##

                # Calculate a radius for each contour point relative to the ball center
                RadiusList = []
                for x, y in zip(x_coordinates, y_coordinates):
                    Dist_Pt_Center = math.sqrt((XMean - x)**2 + (YMean - y)**2)
                    RadiusList.append(Dist_Pt_Center)
                    
                    # Draw all points of the detected contour
                    if DEBUG: cv2.circle(CombinedMask, (int(x), int(y)), 1, (0, 0, 255), -1)  #color: red

                # Define min- / max- / average radius and standard deviation
                BRad_Avrg = sum(RadiusList) / len(RadiusList)
                BRad_Max = max(RadiusList)
                BRad_Min = min(RadiusList)
                BRad_StdDev = np.std(np.array(RadiusList))
                print("RadMean: ", BRad_Avrg, " RadMin: ", BRad_Min, " RadMax: ", BRad_Max, "RadStdDev: ", BRad_StdDev)  


                ##-- Plausibility check: roundness and ball detection --##

                # Based on threshold values, it is determined if the ball detection process has been successful. It is checked whether there
                # is not a too large distance between the two computed ball center points and whether the standard deviation of the radius is not too large.
                BCenter_ok = True
                BRad_ok = True
                BRound_ok = True
                if abs(XMean - XMean_fromContour) > 5 or abs(YMean - YMean_fromContour) > 5:
                    BCenter_ok = False
                    GroundDist = -1000
                if abs(BRad_Avrg - 15) > 2:
                    BRad_ok = False
                if abs(BRad_StdDev) > 2:
                    BRound_ok = False
                
            else:
                ##-- Error: position, roundness --##
                # No evaluation of the roundness possible because several contours were found in the cleaned ball mask
                BRad_Avrg = -1000; BRad_StdDev = -1000
                BRad_ok = False; BRound_ok = False
                # No plausibility check possible
                BCenter_ok = False
                GroundDist = -1000

            if DEBUG: cv2.imshow("Ball with its center", CombinedMask)
        
        else:
            # Not exactly one ball detected - Error values
            BallRotAngle = -1000
            XMean = -1000; YMean = -1000; GroundDist = -1000
            BRad_Avrg = -1000; BRad_StdDev = -1000
            BCenter_ok = False
            BRad_ok = False; BRound_ok = False

    else:
        ##-- Error case: no ball found --##
        # No ball evaluation possible because no blue or black contour were found: return error values
        BallRotAngle = -1000
        XMean = -1000; YMean = -1000; GroundDist = -1000
        BRad_Avrg = -1000; BRad_StdDev = -1000
        BCenter_ok = False
        BRad_ok = False; BRound_ok = False
        circle_count = -1000
        

    if DEBUG: cv2.imshow('Detected ball', sample)


    return BallRotAngle, XMean, YMean, GroundDist, BRad_Avrg, BRad_StdDev, BCenter_ok, BRad_ok, BRound_ok, circle_count

def FindBallCenter(sample, IMAGE_SIZE, DEBUG=False):
    ###--------------------------------------------###
    ###--- Detection of the center of the ball ---###
    ###--------------------------------------------###
    CombinedMask = sample.copy()

    ##-- Detection of blue half --##
    blue_low = np.array([120, 0, 0])
    blue_high = np.array([255, 150, 100])
    blue_mask = cv2.inRange(sample, blue_low, blue_high)

    # Find contours on the blue mask
    BlueContours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    ##-- Detection of the black half --##
    black_low = np.array([0, 0, 0])
    black_high = np.array([100, 110, 90])
    black_mask = cv2.inRange(sample, black_low, black_high)

    # Find contours on the black mask
    BlackContours, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    ##-- If more than 1 contour found, filter contours --##
    NoBallCheckPossible = False
    if not BlueContours:
        NoBallCheckPossible = True
    if not BlackContours:
        NoBallCheckPossible = True


    if not(NoBallCheckPossible):
        # At least 1 blue and 1 black contour was found in each mask

        ##-- Create whole ball mask: add both halves of the ball together --##
        ball_mask = cv2.add(blue_mask, black_mask)
        if DEBUG: cv2.imshow("Ganze Ball-Maske:", ball_mask)

        blurred = cv2.GaussianBlur(ball_mask, (1, 1), 0)
        BallContours, _ = cv2.findContours(blurred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)     # countours of whole ball 
        if DEBUG: print("Anzahl an Ball-Konturen:", len(BallContours))


        ##-- Selection of the relevant contours --##
        if BallContours:
            # If several contours have been detected, keep the two contour with the most points
            BallContour_MaxPts_First_Nb = 0
            BallContour_MaxPts_Second_Nb = 0
            BallContour_MaxPts_First_Idx = 0
            BallContour_MaxPts_Second_Idx = 0
            counter = 0
            for c in BallContours:
                contour_coord = c.ravel()
                if counter == 0:
                    # first iteration of the for loop
                    BallContour_MaxPts_First_Nb = len(contour_coord[::2])
                    BallContour_MaxPts_First_Idx = counter
                elif counter !=0 and BallContour_MaxPts_First_Nb < len(contour_coord[::2]):
                    # A larger contour was found than the last saved contour.
                    # Save the number of points and the index of the largest contour in the second largest contour.
                    BallContour_MaxPts_Second_Nb = BallContour_MaxPts_First_Nb
                    BallContour_MaxPts_Second_Idx = BallContour_MaxPts_First_Idx
                    # Save the new max number of points (and the index) in the largest contour. 
                    BallContour_MaxPts_First_Nb = len(contour_coord[::2])
                    BallContour_MaxPts_First_Idx = counter
                    
                counter += 1
            
            if DEBUG: print(f"Longest contour: {BallContour_MaxPts_First_Nb} points")
            if DEBUG: print(f"Second longest contour: {BallContour_MaxPts_Second_Nb} points")
        

        ##-- Create cleaned ball mask (without any artefacts etc) --##
        # Here, a maximum of the two largest ball contours (from the uncleaned mask) are taken and filled in
        ball_mask_cleaned = np.zeros(sample.shape[:2], dtype=np.uint8)
        
        # In each case, the largest contour is taken and filled in 
        cv2.drawContours(ball_mask_cleaned, [BallContours[BallContour_MaxPts_First_Idx]], 0, 255, thickness=cv2.FILLED)
        # If there is a second contour that has almost the same number of points which suggests that the ball 
        # may have been recognised in two parts, the second contour is also added to the final mask.
        if (BallContour_MaxPts_First_Nb <= BallContour_MaxPts_Second_Nb + 10) and BallContour_MaxPts_Second_Nb != 0:
            cv2.drawContours(ball_mask_cleaned, [BallContours[BallContour_MaxPts_Second_Idx]], 0, 255, thickness=cv2.FILLED)
        
        if DEBUG: cv2.imshow("Cleaned ball mask", ball_mask_cleaned)


        ##-- Detection of the ball center --##
        # All points of the assembled ball mask are analysed. The center of the ball is derived 
        # from the smallest and largest X and Y coordinates. 
        
        ball_px_pos = []
        height, width = ball_mask_cleaned.shape
        for y in range(height):
            for x in range(width):
                # Save all coordinates of the ball mask
                if ball_mask_cleaned[y, x] > 0:
                    ball_px_pos.append((x, y))
        if DEBUG: print("Number of ball pixel: ", len(ball_px_pos))

        # Search for smallest and largest X and Y coordinates
        XMax = 0; XMin = IMAGE_SIZE
        YMax = 0; YMin = IMAGE_SIZE
        for e in ball_px_pos:
            if e[0] >= XMax:
                XMax = e[0]
            if e[0] <= XMin:
                XMin = e[0]
            if e[1] >= YMax:
                YMax = e[1]
            if e[1] <= YMin:
                YMin = e[1]

        # Calculate the center of the ball using the average of the min and max coordinates
        XMean = (XMax-XMin)/2 + XMin
        YMean = (YMax-YMin)/2 + YMin
        BCenter_ok = True
        print("Ball center:", XMean, YMean)
        if DEBUG: cv2.circle(CombinedMask, (int(XMean), int(YMean)), 3, (0, 0, 255), -1)  #red color
        if DEBUG: cv2.drawContours(CombinedMask, BallContours, -1, (0, 255, 0), 1)        #green color
        

    else:
        ##-- Error case: no ball detected --##
        # No ball evaluation possible because no blue or black contour was found: only error values
        XMean = -1000; YMean = -1000
        BCenter_ok = False
        

    if DEBUG: cv2.imshow('Detected ball', sample)


    return XMean, YMean, BCenter_ok


###------------------------------###
#     Main - Start evaluation      #
###------------------------------###

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/true")        # path of the folder containing the true samples including csv file for parametrization
    parser.add_argument("--pred_dir", type=str, default="data/pred")        # path of the folder containing the predicted samples which have to be evaluated
    parser.add_argument("--output", type=str, default="evaluation.csv")     # path and name of the file to save the evaluation results
    parser.add_argument("--ball_case", type=bool, default=False)            # 0: rolling ball | 1: bouncing ball
    args = parser.parse_args()

    data_dir = args.data_dir
    pred_dir = args.pred_dir
    output = args.output
    ball_case = args.ball_case

    IMAGE_SIZE = 256

    # Import parameters of true label (from csv file)
    TrueCSV_path = data_dir + "/" + "Dataset_ImageParams.csv"
    df_ImgParams = pd.read_csv(TrueCSV_path)
    # Create list of predicted samples of the indicated folder
    list_PredImgs = [f for f in os.listdir(pred_dir) if f.endswith((".png"))]

    # Create DataFrame to save the results of the evaluation
    columns = ["TruePict", "PredPict", "StartHeight", "StartTime", "TimeIntervall", "GroundIncli_theo", "GroundIncli_meas", "GroundIncli_Err", "BallsNb", "BRot_true", "BRot_meas", "BRot_Err", "BCenterX", "BCenterY", "BCenterX_Err", "BCenterY_Err", "GroundDist", "Balls_XDist", "BRad", "BRound_Err"]
    DF_EvalPredImg = pd.DataFrame(columns=columns)
    BallsImg_NotOnGround = []
    BallsImg_NotRollingDown = []

    counter = 0 
    for i in list_PredImgs:
        y_pred_path = pred_dir + "/" + i
        y_true_path = data_dir + "/DoubleImg_" + str(counter) + ".jpg"
        print(f"Image {i} is being analysed...")


        ###--- Open and convert images ---###
        TrueImgOriginal = cv2.imread(y_true_path)
        StartImgOriginal = TrueImgOriginal[:, :512]
        StartImg = cv2.resize(StartImgOriginal, (IMAGE_SIZE, IMAGE_SIZE))
        TrueImgOriginal = TrueImgOriginal[:, 512:]
        TrueImg = cv2.resize(TrueImgOriginal, (IMAGE_SIZE, IMAGE_SIZE))
        PredImg = cv2.imread(y_pred_path)

        ###--- Get parameters of the start and target sample (from csv) ---###
        StartTime = df_ImgParams.InputTime[counter]
        TimeIntervall = df_ImgParams.TargetTime[counter] - df_ImgParams.InputTime[counter]
        if ball_case: StartHeight = df_ImgParams.StartHeight[counter]
        else: StartHeight = 0


        ###--- Evaluate ground slope of both images ---###
        GroundIncli_true = abs(df_ImgParams.GroundIncli[counter])
        GroundIncli_pred, GroundRho_pred, GroundTheta_pred = CheckGround(PredImg, IMAGE_SIZE, DEBUG=False)

        ###--- Evaluate ball position and shape of both images ---###
        BRot_pred, XMean_pred, YMean_pred, GroundDist_pred, BRad_Avrg_pred, BRad_StdDev_pred, BCenter_ok_pred, BRad_ok_pred, BRound_ok_pred, Balls_Nb_pred = CheckBall(PredImg, IMAGE_SIZE, DEBUG=False, Ground_Rho=GroundRho_pred, Ground_Theta=GroundTheta_pred)
        BRot_true, XMean_true, YMean_true, GroundDist_true, BRad_Avrg_true, BRad_StdDev_true, BCenter_ok_true, BRad_ok_true, BRound_ok_true, Balls_Nb_true = CheckBall(TrueImg, IMAGE_SIZE, DEBUG=False)
        # Additionally determine the ball position on the start image (only if there is exactly 1 ball in the predicted image)
        if Balls_Nb_pred == 1: 
            XMean_start, YMean_start, BCenter_ok_start = FindBallCenter(StartImg, IMAGE_SIZE, DEBUG=False)


        ###--- Save results of evaluation in DataFrame ---###

        TrueImg_Name = "truth_" + str(counter)
        PredImg_Name = "gen_" + str(counter)

        # Ball rotation
        # If the ball rotation could not be determined, the value is set to -1000 so that the images are not included in the average value of the roundness error during an evaluation.
        # If there is no clean colour separation between the two halves of the ball in the image, a value of -9999 is set for the ball rotation.
        # If the distance between the blue and black colour separation lines of the ball is too big, a value of -5000 is set for the ball rotation.
        if BRot_pred == -1000 or BRot_true == -1000:
            BRot_Err = -1000
        elif BRot_pred == -1001 or BRot_true == -1001:
            BRot_Err = -1001
        elif BRot_pred == -1002 or BRot_true == -1002:
            BRot_Err = -1002
        elif BRot_pred == -5000:
            BRot_Err = -5000
        elif BRot_true == -5000:
            BRot_Err = -5001
        elif BRot_pred == -9999:
            BRot_Err = -9999
        else:
            # No error in the rotation detection of the ball
            BRot_Err = BRot_true - BRot_pred
            if abs(BRot_Err) > 180:
                BRot_Err = 360 - abs(BRot_Err)

        # Center of the ball
        if BCenter_ok_pred and BCenter_ok_true:
            # Ball center detected correctly
            BCenterX_Err = XMean_true - XMean_pred
            BCenterY_Err = YMean_true - YMean_pred
            # Check that the error is not too negative, which would mean that the predicted ball position is on the left of 
            # the "start ball" (along the x axis), which would be physically impossible as the ball always rolls downwards
            if BCenter_ok_start and (XMean_pred - XMean_start) < 0:
                BallsImg_NotRollingDown.append((PredImg_Name, XMean_start, XMean_true, XMean_pred))
                Balls_XDist = XMean_pred - XMean_start
            elif BCenter_ok_start:
                Balls_XDist = XMean_pred - XMean_start
            else:
                Balls_XDist = -1000
        elif not(BCenter_ok_pred) and XMean_pred != -1000 and YMean_pred != -1000:
            # Ball center detected for the predicted sample but evaluated as not plausibel
            BCenterX_Err = -5000
            BCenterY_Err = -5000
            Balls_XDist = -1000
        elif not(BCenter_ok_true) and XMean_true != -1000 and YMean_true != -1000:
            # Ball center detected for the true sample but evaluated as not plausibel
            BCenterX_Err = -5001
            BCenterY_Err = -5001
            Balls_XDist = -1000
        else:
            BCenterX_Err = -1000
            BCenterY_Err = -1000
            Balls_XDist = -1000

        # Ball radius error case
        if not(BRad_ok_pred):
            BRad_Avrg_pred = -1000
        # Ball roundness error case
        if not(BRound_ok_pred):
            BRad_StdDev_pred = -1000

        NewData = {"TruePict": TrueImg_Name, "PredPict": PredImg_Name, "StartHeight": StartHeight, "StartTime": StartTime, "TimeIntervall": TimeIntervall, 
                "GroundIncli_theo": GroundIncli_true, "GroundIncli_meas": GroundIncli_pred, "GroundIncli_Err": GroundIncli_true-GroundIncli_pred,
                "BallsNb": Balls_Nb_pred, "BRot_true": BRot_true, "BRot_meas": BRot_pred, "BRot_Err": BRot_Err, 
                "BCenterX": XMean_pred, "BCenterY": YMean_pred, "BCenterX_Err": BCenterX_Err, "BCenterY_Err": BCenterY_Err, 
                "GroundDist": GroundDist_pred, "Balls_XDist": Balls_XDist,
                "BRad": BRad_Avrg_pred, "BRound_Err": BRad_StdDev_pred}
        
        DF_EvalPredImg.loc[len(DF_EvalPredImg)] = NewData

        # Evaluation if ball on ground surface
        if abs(GroundDist_pred) > 3 and not(ball_case):
            BallsImg_NotOnGround.append((PredImg_Name, GroundDist_pred))


        counter += 1

    # print("Following balls not on the ground:", BallsImg_NotOnGround)
    # print("Following balls not rolling downwards:", BallsImg_NotRollingDown)

    # Save the Dataframe to csv file
    DF_EvalPredImg.to_csv(output, index=False)
