from skimage import morphology, data, color
import cv2 as cv
import numpy as np
import math
import random
import copy
import cv2
import matplotlib.pyplot as plt


def plot_test(input_img):

    img_rgb = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGBA)
    plt.imshow(img_rgb)
    plt.axis('off')  # Hide axis ticks
    plt.title('RGB Image')
    plt.show()

def information_extraction(src, k1_control_field_corner, k2_control_field_third_bezier):
    """
    [[list_cor], [list_ske], [list_bezier_use_information]]
     /////[list_bezier_use_information] = [[pt_cor1, pt_cor2],[pt_third_bezier],[control region of the two corner points],[control region of the third control point]]
    """

    # Calculate distance between two points
    def two_points_distance(point1, point2):
        p1 = point1[0] - point2[0]
        p2 = point1[1] - point2[1]
        distance = math.hypot(p1, p2)

        return distance

    # Calculate distance from a point to a line formed by two points
    def get_distance_from_point_to_line(point, line_point1, line_point2):
        # If the two line points are the same, return the distance between the point and the line point
        if line_point1 == line_point2:
            point_array = np.array(point)
            point1_array = np.array(line_point1)
            return np.linalg.norm(point_array - point1_array)
        # Calculate the three parameters of a straight line
        A = line_point2[1] - line_point1[1]
        B = line_point1[0] - line_point2[0]
        C = (line_point1[1] - line_point2[1]) * line_point1[0] + \
            (line_point2[0] - line_point1[0]) * line_point1[1]
        # Calculate the distance using the distance formula for
        distance = np.abs(A * point[0] + B * point[1] + C) / (np.sqrt(A ** 2 + B ** 2))
        return distance

    # skeleton index
    def skeleton_index(skeleton_src):
        index = []
        row, col = skeleton_src.shape
        for a in range(0, row):
            for b in range(0, col):
                if skeleton_src[a][b] == 1:
                    index.append((b, a))

        return index

    # Seeking multiple branch points and endpoints
    def corner_index(list_ske):
        list_corner = []
        for point in list_ske:
            num_branches = 0
            for a in range(-1, 2):
                for b in range(-1, 2):
                    if(point[0] + a, point[1] + b) in list_ske:
                        num_branches = num_branches + 1
            if num_branches != 3:
                list_corner.append(point)

        return list_corner

    # Remove all vertices from the skeleton, leaving only the skeleton.
    def skeleton_clean(list_ske, list_corner):
        for pt in list_corner:
            list_ske.remove(pt)

        return list_ske

    # Corner point redundancy removal
    def list_corner_clean(image_original, list_corner):
        # First, perform connected component detection on the corner points.
        image_original = image_original.astype(np.uint8) * 255
        image_original_copy = np.zeros_like(image_original)
        for pt in list_corner:
            cv.circle(image_original_copy, pt, 0, (255, 255, 255), -1)

        # Connected domain detection
        nums_label, labels = cv.connectedComponents(image_original_copy)

        # Classify the points in the connected domain into a list.
        list_total = []
        # First, determine how many sublists you want to create.
        for a in range(0, nums_label - 1):
            list_total.append([])
        # Iterate through all labels
        row, col = labels.shape
        for a in range(0, row):
            for b in range(0, col):
                if labels[a][b] > 0:
                    list_total[(labels[a][b] - 1)].append((b, a))

        # Next, directly select the center vertices of each subconnected domain.
        list_corner_new = []
        for lst in list_total:
            if len(lst) == 1:
                list_corner_new.append(lst[0])
            else:
                list_sort_a = []
                list_sort_b = []
                list_distance = []
                for pt in lst:
                    list_sort_a.append(pt[0])
                    list_sort_b.append(pt[1])
                list_sort_a.sort()
                list_sort_b.sort()
                mid_a = (list_sort_a[0] + list_sort_a[-1]) / 2
                mid_b = (list_sort_b[0] + list_sort_b[-1]) / 2
                for pt in lst:
                    list_distance.append(two_points_distance((mid_a, mid_b), pt))
                list_corner_new.append(lst[list_distance.index(min(list_distance))])

        return list_corner_new

    # Skeleton connectivity domain detection
    def skeleton_connected(image_original, ske_index):

        # Skeleton diagram after removing redundancy
        image_original = image_original.astype(np.uint8) * 255
        image_original_copy = np.zeros_like(image_original)
        for pt in ske_index:
            cv.circle(image_original_copy, pt, 0, (255, 255, 255), 0)

        # Connected domain detection
        nums_label, labels = cv.connectedComponents(image_original_copy)

        # Classify the points in the connected domain into a list.
        list_total = []
        # First, determine how many sublists you want to create.
        for a in range(0, nums_label-1):
            list_total.append([])
        # Iterate through all labels
        row, col = labels.shape
        for a in range(0, row):
            for b in range(0, col):
                if labels[a][b] > 0:
                    list_total[(labels[a][b]-1)].append((b, a))

        return list_total

    # ---------------------------
    # Skeleton corner matching - fixed endpoints
    # ---------------------------
    def ske_cor_match(list_ske, list_cor):
        """
        Match skeleton segments to corner points.
        Returns a list of items, each shaped as:
          [ [end_pt1, end_pt2], [cor_pt1, cor_pt2], [segment_points] ]
        """
        list_total = []

        for seg in list_ske:
            list_integration = []
            list_endpoint = []
            list_corner = []

            # Segment with >= 2 points
            if len(seg) >= 2:
                # Find endpoints: count neighbors in 8-neighborhood (endpoint => count == 2: self + 1 neighbor)
                for pt in seg:
                    count_branches = 0
                    for a in range(-1, 2):
                        for b in range(-1, 2):
                            if (pt[0] + a, pt[1] + b) in seg:
                                count_branches += 1
                    if count_branches == 2:
                        list_endpoint.append(pt)

                # No endpoints found: keep the segment and choose pseudo-endpoints (farthest pair)
                if len(list_endpoint) == 0:
                    print("No Endpoints on segment.. keeping segment with pseudo-endpoints")
                    p1, p2, maxd = seg[0], seg[-1], -1.0
                    for i in range(len(seg)):
                        for j in range(i + 1, len(seg)):
                            d = two_points_distance(seg[i], seg[j])
                            if d > maxd:
                                maxd = d
                                p1, p2 = seg[i], seg[j]
                    list_endpoint = [p1, p2]
                elif len(list_endpoint) == 1:
                    # If only one endpoint was found, duplicate it
                    list_endpoint = [list_endpoint[0], list_endpoint[0]]

                # Match nearest corner to each endpoint (if no corners, use the endpoint itself)
                list_integration.append(list_endpoint)
                for pt_end in list_endpoint:
                    if list_cor:
                        dists = [two_points_distance(pt_end, c) for c in list_cor]
                        min_idx = dists.index(min(dists))
                        list_corner.append(list_cor[min_idx])
                    else:
                        list_corner.append(pt_end)

                list_integration.append(list_corner)
                list_integration.append(seg)
                list_total.append(list_integration)

            # Segment with exactly 1 point
            elif len(seg) == 1:
                pt = seg[0]
                list_endpoint = [pt, pt]
                list_integration.append(list_endpoint)

                if list_cor:
                    dists = [two_points_distance(pt, c) for c in list_cor]
                    d_sorted = sorted(((d, i) for i, d in enumerate(dists)))
                    idx1 = d_sorted[0][1]
                    idx2 = d_sorted[1][1] if len(d_sorted) > 1 else idx1
                    list_integration.append([list_cor[idx1], list_cor[idx2]])
                else:
                    list_integration.append([pt, pt])

                list_integration.append(seg)
                list_total.append(list_integration)

        # Remove already used corners (ignore duplicates safely)
        used = []
        for it in list_total:
            for pt in it[1]:
                used.append(pt)
        for pt in set(used):
            if pt in list_cor:
                list_cor.remove(pt)

        # Keep isolated corners as degenerate segments
        if len(list_cor) != 0:
            for pt in list_cor:
                list_integration = []
                list_endpoint = [pt, pt]
                list_integration.append(list_endpoint)      # endpoints
                list_integration.append(list_endpoint)      # corners (same)
                list_integration.append([pt])               # segment points
                list_total.append(list_integration)

        return list_total

    # ---------------------------
    # Debug: visualize a problematic segment  
    # ---------------------------
    def show_segment_issue(list_ske, seg):
        # Create visualization image
        x_coords = [pt[0] for segment in list_ske for pt in segment] + [pt[0] for pt in seg]
        y_coords = [pt[1] for segment in list_ske for pt in segment] + [pt[1] for pt in seg]
        width = max(x_coords) - min(x_coords) + 20
        height = max(y_coords) - min(y_coords) + 20
        debug_img = np.zeros((height, width, 3), dtype=np.uint8)
        # Draw all segments in blue
        for segment in list_ske:
            for pt in segment:
                cv.circle(debug_img, pt, 1, (255, 0, 0), -1)
        # Draw current segment in red
        for pt in seg:
            cv.circle(debug_img, pt, 1, (0, 0, 255), -1)
        cv.imshow("Debug - No Endpoints", debug_img)
        cv.waitKey(0)

    # Skeleton reordering
    def ske_rearrangement(list_total):
        """
         [[cor1_1,cor1_2],[ske_list1]]
        """
        for a in list_total:
            try:
                point_temp = a[0][0]
            except:
                print()
            list_ske_new = []
            while point_temp != a[0][1]:
                list_ske_new.append(point_temp)
                a[2].remove(point_temp)
                list_temp = []
                # Check the 8 areas of this point.
                for x in range(-1, 2):
                    for y in range(-1, 2):
                        if (point_temp[0]+x, point_temp[1]+y) in a[2]:
                            list_temp.append((point_temp[0]+x, point_temp[1]+y))
                point_temp = list_temp[0]
            list_ske_new.append(a[0][1])
            # Reconstruct the original list_from_ske_cor_match
            del(a[2])
            del(a[0])
            a.append(list_ske_new)

        return list_total

    # Corner point supplementation
    def cor_addition(list_total):
        """
        [[list_cor], [list_ske], [list_bezier_use_information]]
            ///////[list_bezier_use_information] = [[list_cor], [list_ske]]
        """
        # Loop through the skeleton list
        for index_list_total in range(len(list_total)):

            # Data preparation
            list_bezier_information, list_bezier_cor, list_bezier_ske, list_integration = [], [], [], []
            child_list_total = list_total[index_list_total]
            len_ske = len(child_list_total[1])
            # A small number of skeletons are not processed directly.
            if len_ske <= 4:
                list_no_addition = copy.copy(child_list_total)
                list_bezier_information.append(list_no_addition)
                list_total[index_list_total].append(list_bezier_information)
                continue  # The following will not be executed; proceed directly to the next main loop.

            # Normal skeletal condition
            list_cor = child_list_total[0]
            list_ske = child_list_total[1]
            # First, enter the first corner point into list_bezier_cor.
            list_bezier_cor.append(list_cor[0])
            flag_first_pt = 1  # Set the initial flag and determine whether to start from the first point.
            flag_need_new_direction = 1
            pt_cor = list_cor[0]  # Define corner points Tentatively set as the first corner point
            flag_judge_s_change = 0  # Define a flag to determine whether the S-type skeleton has changed.
            target_arc_x, target_arc_y = 0, 0  # Initialize arc skeleton direction
            flag_judge_integration = 0  # Initialization integration determination tag
            list_judge_three_direction = []  # Initialize a list of three reasonable directions.
            x, y = 0, 0  # Initialization direction
            index_ske = 0

            # Cycle from the skeleton
            for index_ske in range(len_ske-3):
                # First, determine whether a new direction needs to be defined
                if flag_need_new_direction == 1:  # Indicates a new direction is needed
                    flag_need_new_direction = 0  # Reset direction; no new direction needed

                    # Determine whether it's the first corner point
                    if flag_first_pt == 1:   # Indicates it's the first corner point
                        flag_first_pt = 0 # Reset; no longer the first point, happens only once
                        pt_cor = list_cor[0]
                        # Check whether the corner point and skeleton point are adjacent
                        flag_judge_near = 0 # Initialize the 'near' flag
                        for a in range(-1, 2):
                            for b in range(-1, 2):
                                if list_ske[index_ske][0] + a == pt_cor[0] and list_ske[index_ske][1] + b == pt_cor[1]:
                                    flag_judge_near = 1  # Indicates the corner point and skeleton are adjacent
                                    x, y = list_ske[0][0] - pt_cor[0], list_ske[0][1] - pt_cor[1]
                        if flag_judge_near == 0:  # Indicates the corner point and skeleton are not adjacent; directly use direction between first and second skeleton points
                            x, y = list_ske[1][0] - list_ske[0][0], list_ske[1][1] - list_ske[0][1]
                    # Indicates it's not the first corner point; directly use direction from corner to skeleton
                    else:
                        x, y = list_ske[index_ske][0] - pt_cor[0], list_ske[index_ske][1] - pt_cor[1]

                    # A new direction is defined, so the target direction must also be updated
                    target_arc_x, target_arc_y = -x, -y
                    # Set which three directions are valid for flag_judge_s_change; total of 8 cases
                    if (x, y) == (-1, -1):
                        list_judge_three_direction = [(-1, -1), (-1, 0), (0, -1)]
                    elif (x, y) == (-1, 0):
                        list_judge_three_direction = [(-1, 0), (-1, 1), (-1, -1)]
                    elif (x, y) == (-1, 1):
                        list_judge_three_direction = [(-1, 1), (-1, 0), (0, 1)]
                    elif (x, y) == (0, 1):
                        list_judge_three_direction = [(0, 1), (-1, 1), (1, 1)]
                    elif (x, y) == (1, 1):
                        list_judge_three_direction = [(1, 1), (0, 1), (1, 0)]
                    elif (x, y) == (1, 0):
                        list_judge_three_direction = [(1, 0), (1, 1), (1, -1)]
                    elif (x, y) == (1, -1):
                        list_judge_three_direction = [(1, -1), (1, 0), (0, -1)]
                    elif (x, y) == (0, -1):
                        list_judge_three_direction = [(0, -1), (-1, -1), (1, -1)]

                # After determining whether a new direction is needed, come here (regardless of whether the determination is successful or not, you must go through this step).
                # Use the front and rear skeleton directions of this cycle for the determination.
                direction_ske_x = list_ske[index_ske+1][0] - list_ske[index_ske][0]
                direction_ske_y = list_ske[index_ske+1][1] - list_ske[index_ske][1]

                # Determine whether there are S-shaped or arc-shaped strokes, and change the integration tag accordingly.
                # Determine whether there is a circular arc-shaped skeleton.
                if target_arc_x == direction_ske_x and target_arc_y == direction_ske_y:
                    flag_judge_integration = 1
                # Determine whether there is an S-shaped skeleton.
                elif flag_judge_s_change == 1:
                    if direction_ske_x == x and direction_ske_y == y:
                        flag_judge_integration = 1
                elif flag_judge_s_change == 0:
                    if (direction_ske_x, direction_ske_y) not in list_judge_three_direction:
                        flag_judge_s_change = 1

                # Use the integration flag to determine whether to integrate
                if flag_judge_integration == 0: # Indicates no integration, but this point should be added to the new skeleton
                    list_bezier_ske.append(list_ske[index_ske])
                elif flag_judge_integration == 1:  # Indicates integration is needed; don't forget to initialize
                    # Integrate
                    list_bezier_cor.append(list_ske[index_ske])
                    list_integration.append(list_bezier_cor), list_integration.append(list_bezier_ske)
                    list_bezier_information.append(list_integration)
                    # Initialize
                    list_bezier_cor, list_bezier_ske, list_integration = [], [], []
                    pt_cor = list_ske[index_ske]  # Corner point needs to be redefined
                    list_bezier_cor.append(pt_cor)
                    flag_judge_s_change, flag_judge_integration, flag_need_new_direction = 0, 0, 1

            # The loop has finished, but the last integration is still missing because some skeleton points remain
            list_bezier_cor.append(list_cor[1])
            list_bezier_ske.extend(list_ske[index_ske + 1:])
            list_integration.append(list_bezier_cor), list_integration.append(list_bezier_ske)
            list_bezier_information.append(list_integration)
            list_total[index_list_total].append(list_bezier_information)

        return list_total

    # Find the most prominent point of the skeleton and the third Bézier curve point.
    def third_bezier_curve_point(list_total):
        """
        [[list_cor], [list_ske], [list_bezier_use_information]]
            ///////[list_bezier_use_information] = [[pt_cor1, pt_cor2], [pt_third_bezier]]
        """
        for index_list_total in range(len(list_total)):
            list_bezier_information = list_total[index_list_total][2]
            for index_child_bezier in range(len(list_bezier_information)):
                list_child_bezier = list_bezier_information[index_child_bezier]
                list_ske, list_distance_ske = list_child_bezier[1], []
                pt_cor1, pt_cor2 = list_child_bezier[0][0], list_child_bezier[0][1]
                # Find the most prominent point of the skeleton.
                for pt_ske in list_ske:
                    list_distance_ske.append(get_distance_from_point_to_line(pt_ske, pt_cor1, pt_cor2))
                pt_bulge = list_ske[list_distance_ske.index(max(list_distance_ske))]  # 最突出点
                # The skeleton is now useless, so delete it.
                del(list_total[index_list_total][2][index_child_bezier][1])
                # Third Bézier control point
                pt_center_x = 0.5 * float(pt_cor1[0]) + 0.5 * float(pt_cor2[0])
                pt_center_y = 0.5 * float(pt_cor1[1]) + 0.5 * float(pt_cor2[1])
                pt_third_bezier_x = int((float(pt_bulge[0]) - pt_center_x) * 2 + pt_center_x)
                pt_third_bezier_y = int((float(pt_bulge[1]) - pt_center_y) * 2 + pt_center_y)
                pt_third_bezier = [(pt_third_bezier_x, pt_third_bezier_y)]  # Third control point
                # 打进数据
                list_total[index_list_total][2][index_child_bezier].append(pt_third_bezier)

        return list_total

    # Control domain computing
    def control_field(list_total, k1, k2):  # (k_corner point, k_third control point)
        """
        [[list_cor], [list_ske], [list_bezier_use_information]]
            ///////[list_bezier_use_information] = [[pt_cor1, pt_cor2],[pt_third_bezier],[Two corner point control domain], [Third control point control domain]
        """

        # First, you need a set of all vertices.
        list_cor_all = []
        for p in list_total:
            for q in p[2]:
                for pt in q[0]:
                    list_cor_all.append(pt)
        list_cor_all = list(set(list_cor_all))

        # Main loop starts
        for index_list_total in range(len(list_total)):
            list_bezier_information = list_total[index_list_total][2]
            for index_child_bezier in range(len(list_bezier_information)):
                list_child_bezier = list_bezier_information[index_child_bezier]
                list_cor, pt_third_bezier = list_child_bezier[0], list_child_bezier[1][0]
                pt_cor1, pt_cor2 = list_cor[0], list_cor[1]
                list_control_field_corner, list_control_field_third_bezier = [], []

                # First, calculate the corner control domain.
                for pt_cor in list_cor:
                    list_temp = copy.copy(list_cor_all)   # Copy the list of all corner points
                    list_temp.remove(pt_cor)  # Remove the currently iterated corner point from the list
                    list_distance =  []  # Create a list to store distances between corner points
                    for pt_temp in list_temp:
                        list_distance.append(two_points_distance(pt_temp, pt_cor))
                    pt_corner_min_distance = list_temp[list_distance.index(min(list_distance))]   # Get the corner point closest to pt_cor
                    # Control domain size
                    width_control_field = float(abs(pt_cor[0] - pt_corner_min_distance[0]) * k1)
                    height_control_field = float(abs(pt_cor[1] - pt_corner_min_distance[1]) * k1)
                    # Data entered into list_control_field_corner
                    list_control_field_corner.append([width_control_field, height_control_field])

                # Calculate the control domain of the third control point
                len_rectangular = float(get_distance_from_point_to_line(pt_third_bezier, pt_cor1, pt_cor2))
                list_control_field_third_bezier.append(float(len_rectangular * k2))

                # enter data
                list_total[index_list_total][2][index_child_bezier].append(list_control_field_corner)
                list_total[index_list_total][2][index_child_bezier].append(list_control_field_third_bezier)

        return list_total

    # Draw it out to see how it looks.
    def draw_temp(skeleton_src, list_total):
        # First adjust the format
        skeleton_image = skeleton_src.astype(np.uint8) * 255
        skeleton_image = np.zeros_like(skeleton_image)
        skeleton_image = cv.cvtColor(skeleton_image, cv.COLOR_GRAY2BGR)
        list_color = [(60, 230, 150), (230, 150, 150), (255, 80, 10), (60, 60, 60), (10, 70, 250),
                      (50, 255, 190)]
        # Draw corner points
        for k in list_total:
            # Draw skeleton
            # Randomly select a color
            col = random.randint(0, 5)
            for pt in k[1]:
                cv.circle(skeleton_image, pt, 0, list_color[col], cv.FILLED)
            for g in k[2]:
                # Draw corner points
                for pt in g[0]:
                    cv.circle(skeleton_image, pt, 0, (0, 255, 0), cv.FILLED)
                # Draw third Bezier control point
                pt_third_bezier = g[1][0]
                cv.circle(skeleton_image, pt_third_bezier, 0, (255, 255, 255), cv.FILLED)
                # Draw control domains
                pt_cor1, pt_cor2 = g[0][0], g[0][1]
                list_control_field_corner,  list_control_field_third_bezier = g[2], g[3]
                # Draw control domain of corner point 1
                cv.rectangle(img=skeleton_image,
                             pt1=(int(pt_cor1[0])-int(list_control_field_corner[0][0]),
                                  int(pt_cor1[1])+int(list_control_field_corner[0][1])),
                             pt2=(int(pt_cor1[0])+int(list_control_field_corner[0][0]),
                                  int(pt_cor1[1])-int(list_control_field_corner[0][1])),
                             color=(255, 255, 0), thickness=1)
                # Draw control domain of corner point 2
                cv.rectangle(img=skeleton_image,
                             pt1=(int(pt_cor2[0]) - int(list_control_field_corner[1][0]),
                                  int(pt_cor2[1]) + int(list_control_field_corner[1][1])),
                             pt2=(int(pt_cor2[0]) + int(list_control_field_corner[1][0]),
                                  int(pt_cor2[1]) - int(list_control_field_corner[1][1])),
                             color=(255, 255, 0), thickness=1)
                # Draw control domain of third control point
                cv.rectangle(img=skeleton_image,
                             pt1=(int(pt_third_bezier[0]) - int(list_control_field_third_bezier[0]),
                                  int(pt_third_bezier[1]) + int(list_control_field_third_bezier[0])),
                             pt2=(int(pt_third_bezier[0]) + int(list_control_field_third_bezier[0]),
                                  int(pt_third_bezier[1]) - int(list_control_field_third_bezier[0])),
                             color=(0, 255, 0), thickness=1)

        return skeleton_image

    # Convert handwritten text to skeleton
    img = src
    #plot_test(img)
    img = cv.blur(img, (3, 3))
    #plot_test(img)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # gray = 255 - gray
    gray2 = (gray < 240).astype(np.uint8) * 255
    gray = (gray < 240)

    #plot_test(gray2)

    skeleton_image = morphology.skeletonize(gray)
    skeleton = (skeleton_image).astype(np.uint8) * 255
    #plot_test(skeleton)
    # Calculate skeleton list
    list_skeleton = skeleton_index(skeleton_image)

    # Calculate all multi-branch points and endpoints
    list_corner_original = corner_index(list_skeleton)
    # Skeleton redundancy removal (Remove all vertices from the skeleton, leaving only the skeleton)
    list_skeleton = skeleton_clean(list_skeleton, list_corner_original)
    # Corner point redundancy removal
    list_corner_new = list_corner_clean(skeleton_image, list_corner_original)
    # Skeleton connectivity domain detection
    list_skeleton = skeleton_connected(skeleton_image, list_skeleton)
    # Skeleton connectivity domain detection
    list_match = ske_cor_match(list_skeleton, list_corner_new)

    # At this point, we have enough information. Next, we will add supplementary information for Bezier curve deformation.
    ################################
    ################################

    # Skeleton reordering
    list_match = ske_rearrangement(list_match)
    # Corner point supplementation
    list_match = cor_addition(list_match)
    # Third Bézier control point
    list_match = third_bezier_curve_point(list_match)
    # control domain
    image_information = control_field(list_match, k1_control_field_corner, k2_control_field_third_bezier)

    # Visualize the result
    #image_information = draw_temp(skeleton_image, list_match)

    return image_information


# if __name__ == '__main__':
#     for i in range(1, 101):
#         img = cv.imread("./pending/"+str(i)+".jpg", cv.IMREAD_COLOR)
#         image = information_extraction(img, 0.5, 0.4)
#         cv.imwrite("./After processing/"+str(i)+".png", image)
#         print("done:"+str(i)+"/100")
#
#         # img = cv.imread("./pending/57.jpg", cv.IMREAD_COLOR)
#         # image = information_extraction(img)
