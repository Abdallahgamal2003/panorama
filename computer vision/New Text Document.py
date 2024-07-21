import cv2
import numpy as np

def extract_SIFT(image):
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return keypoints, descriptors

def arrange_pairs(setA, setB):
    arranged_setB = []
    for pointA in setA:
        min_distance = float('inf')
        nearest_pointB = None
        xA, yA = pointA.pt
        for pointB in setB:
            xB, yB = pointB.pt
            dist = np.sqrt((xA - xB) * 2 + (yA - yB) * 2)
            if dist < min_distance:
                min_distance = dist
                nearest_pointB = pointB
        arranged_setB.append(nearest_pointB)
    return arranged_setB

def estimate_transformation(setA, setB):
    if len(setA) != len(setB) or len(setA) < 4:
        raise ValueError("Number of points in setA and setB must be equal and at least 4")
    A = []
    for i in range(len(setA)):
        x, y = setA[i].pt[0], setA[i].pt[1]
        u, v = setB[i].pt[0], setB[i].pt[1]
        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y, -u])
        A.append([0, 0, 0, x, y, 1, -v * x, -v * y, -v])
    _, _, V = np.linalg.svd(A)
    transformation_matrix = V[-1].reshape(3, 3)
    return transformation_matrix

def apply_transformation(T, set):
    transformed_set = []
    if T.shape == (3, 3):
        T = np.vstack([T, [0, 0, 1]])
    for point in set:
        point_homogeneous = np.array([point.pt[0], point.pt[1], 1])
        transformed_point = np.dot(T, point_homogeneous)
        transformed_set.append(cv2.KeyPoint(transformed_point[0], transformed_point[1], point.size))
    return transformed_set

def RANSAC(setA, setB, max_iterations, threshold):
    best_model = None
    best_inliers = []
    for _ in range(max_iterations):
        subsetA, subsetB = random_subset(setA, setB)
        model = estimate_transformation(subsetA, subsetB)
        inliers = find_inliers(setA, setB, model, threshold)
        if len(inliers) > len(best_inliers):
            best_model = model
            best_inliers = inliers
    return best_model, best_inliers

def random_subset(setA, setB, subset_size=4):
    indices = np.random.choice(len(setA), subset_size, replace=False)
    subsetA = [setA[i] for i in indices]
    subsetB = [setB[i] for i in indices]
    return subsetA, subsetB

def find_inliers(setA, setB, model, threshold):
    inliers = []
    for i in range(len(setA)):
        pointA = setA[i].pt
        pointB = setB[i].pt
        transformed_pointB = np.dot(model, np.array([pointB[0], pointB[1], 1]))
        transformed_pointB = (transformed_pointB[0] / transformed_pointB[2], transformed_pointB[1] / transformed_pointB[2])
        error = np.linalg.norm(np.array([pointA[0], pointA[1], 1]) - np.array([transformed_pointB[0], transformed_pointB[1], 1]))
        if error < threshold:
            inliers.append(i)
    return inliers

def create_panorama_image(imageA, imageB, transformation_matrix):
    hA, wA = imageA.shape[:2]
    hB, wB = imageB.shape[:2]
    transformation_matrix = np.array(transformation_matrix, dtype=np.float32)
    warped_imageB = cv2.warpPerspective(imageB, transformation_matrix, (wA + wB, hA))
    panorama_image = warped_imageB.copy()
    panorama_image[:hA, :wA] = imageA
    return panorama_image

# Example usage:
imageA = cv2.imread('../ImageA.jpg', cv2.IMREAD_GRAYSCALE)
imageB = cv2.imread('../ImageB.jpg', cv2.IMREAD_GRAYSCALE)
keypointsA, descriptorsA = extract_SIFT(imageA)
keypointsB, descriptorsB = extract_SIFT(imageB)
arrangedSetB = arrange_pairs(keypointsA, keypointsB)
print("Number of keypoints in setA:", len(keypointsA))
print("Number of keypoints in setB:", len(keypointsB))
transformation_matrix, inliers = RANSAC(keypointsA, keypointsB, max_iterations=1000, threshold=5)
print("Transformation matrix:")
print(transformation_matrix)
print("Number of inliers:", len(inliers))
panorama_image = create_panorama_image(imageA, imageB, transformation_matrix)
cv2.imshow("Panorama Image", panorama_image)
cv2.waitKey(0)
cv2.destroyAllWindows()