file_output_directory = ""
ground_truth_path = ""
result_path = ""

model_num = 0  # 0 for raw model, 1 for gradient model, 2 for mulGM model

if model_num == 0:
    # LiTS raw model
    file_output_directory = "Dataset/LiTS_data/Testing/"
    ground_truth_path = "Dataset/LiTS_data/Testing/img/ground_truth/test-segmentation-"
    result_path = "Dataset/LiTS_data/Testing/img/results/test-volume-"
elif model_num == 1:
    # LiTS gradient model
    file_output_directory = "Dataset/LiTS_data/GradientTesting/"
    ground_truth_path = "Dataset/LiTS_data/GradientTesting/img/ground_truth/test-segmentation-"
    result_path = "Dataset/LiTS_data/GradientTesting/img/results/test-volume-"
elif model_num == 2:
    # LiTS mulGM model
    file_output_directory = "Dataset/LiTS_data/MulGMTesting/"
    ground_truth_path = "Dataset/LiTS_data/MulGMTesting/img/ground_truth/test-segmentation-"
    result_path = "Dataset/LiTS_data/MulGMTesting/img/results/test-volume-"

# Initialize Output File Location
detected_organs_IoU_file = file_output_directory + "IoU with total detected organs.txt"
groundtruth_organs_IoU_file = file_output_directory + "IoU with original ground truth organs.txt"
organ_detected_rate_file = file_output_directory + "Organ Detected Rate.txt"

# Initialize Empty Organ Array to Store Each Organ's Accuracy in Each Image
liver = []
kidney_r = []
kidney_l = []
femur_r = []
femur_l = []
bladder = []
heart = []
spleen = []
pancreas = []
lung_r = []
lung_l = []

# Record Each Organ's Label According it's Label in Output File
organs_dict = {1: liver, 2: lung_r, 3: lung_l, 4: kidney_r, 5: kidney_l, 6: femur_r,
               7: femur_l, 8: bladder, 9: heart, 10: spleen, 11: pancreas}

# Record Each Organ's Number in Ground Truth
ground_truth_organ_num = {"liver": 0, "lung-r": 0, "lung-l": 0, "kidney-r": 0, "kidney-l": 0, "femur-r": 0,
                          "femur-l": 0, "bladder": 0, "heart": 0, "spleen": 0, "pancreas": 0}

# Get Organ's Name From Label
organs_name_dict = {1: "liver", 2: "lung-r", 3: "lung-l", 4: "kidney-r", 5: "kidney-l", 6: "femur-r",
                    7: "femur-l", 8: "bladder", 9: "heart", 10: "spleen", 11: "pancreas"}

# Define a 3D Bounding Box class
class BBox:
    def __init__(self, x, y, z):
        self.x0 = float(x[0])
        self.x1 = float(x[1])
        self.y0 = float(y[0])
        self.y1 = float(y[1])
        self.z0 = float(z[0])
        self.z1 = float(z[1])

    def getX(self):
        return abs(self.x1 - self.x0)

    def getY(self):
        return abs(self.y1 - self.y0)

    def getZ(self):
        return abs(self.z1 - self.z0)

# Calculate IoU of Two Bounding Box
def calculateIOU(bbox1, bbox2):
    v1 = bbox1.getX() * bbox1.getY() * bbox1.getZ()
    v2 = bbox2.getX() * bbox2.getY() * bbox2.getZ()
    cross_length = min(bbox1.x1, bbox2.x1) - max(bbox1.x0, bbox2.x0)
    if cross_length < 0:
        return 0
    cross_height = min(bbox1.y1, bbox2.y1) - max(bbox1.y0, bbox2.y0)
    if cross_height < 0:
        return 0
    cross_width = min(bbox1.z1, bbox2.z1) - max(bbox1.z0, bbox2.z0)
    if cross_width < 0:
        return 0
    crossv = cross_height * cross_width * cross_length
    return crossv / (v1 + v2 - crossv)

# Calculate Organs Valid IoU
def calculateAvg(list):
    if len(list) == 0:
        return 0
    else:
        return sum(list) / len(list)

# Count The Number of Specific Organ in Ground Truth
def cntGTOrgansNum(list):
    for num in range(len(list) - 1):
        cur_row = list[num].split(" ")
        ground_truth_organ_num[cur_row[0]] += 1

# Calculate True IoU
def calculateAvgWith0IoU(list):
    organ_label = 0
    for key, value in organs_dict.items():
        if value == list:
            organ_label = key
    organ = organs_name_dict[organ_label]
    cnt_organ = ground_truth_organ_num[organ]
    if cnt_organ == 0:
        return 0
    return sum(list) / cnt_organ

# Calculate An Organ Detection Rate
def calculateDetectRate(list):
    organ_label = 0
    for key, value in organs_dict.items():
        if value == list:
            organ_label = key
    organ = organs_name_dict[organ_label]
    cnt_organ = ground_truth_organ_num[organ]
    if cnt_organ == 0:
        return 0
    return len(list) / cnt_organ

ite = 70 # Number of Testing Images
for i in range(ite):
    with open(result_path + str(i) + ".mhd.pred.txt", "r")as f1, open(ground_truth_path + str(i) + ".txt") as f2:
        results = f1.read()
        ground_truth = f2.read()
        ground_truth_organs = ground_truth.split("\n")
        cntGTOrgansNum(ground_truth_organs)
        for organs in results.split("\n"):
            if not organs:
                continue
            ele_row = organs.split(" ")
            organ_label = int(ele_row[0]) # Get the organ label in prediction file
            organs_name = organs_name_dict[organ_label] # Get the organ name
            selected_row = ""
            for row in ground_truth_organs: # Find organ name in ground-truth file
                if organs_name in row:
                    selected_row = row
                    break
            if selected_row == "": # If cannot find, we ignore this organ prediction
                continue
            line1 = organs.split(" ")
            line2 = selected_row.split(" ")
            x = []
            y = []
            z = []
            prediction_bbox = BBox([line1[1], line1[4]], [line1[2], line1[5]], [line1[3], line1[6]])
            ground_truth_bbox = BBox([line2[2], line2[3]], [line2[4], line2[5]], [line2[6], line2[7].strip("\n")])
            iou = calculateIOU(prediction_bbox, ground_truth_bbox)
            organs_dict[organ_label].append(iou)

# Calculate Detected IoU for all Organs
liver_accuracy = calculateAvg(liver)
kidney_r_accuracy = calculateAvg(kidney_r)
kidney_l_accuracy = calculateAvg(kidney_l)
femur_r_accuracy = calculateAvg(femur_r)
femur_l_accuracy = calculateAvg(femur_l)
bladder_accuracy = calculateAvg(bladder)
heart_accuracy = calculateAvg(heart)
spleen_accuracy = calculateAvg(spleen)
pancreas_accuracy = calculateAvg(pancreas)
lung_r_accuracy = calculateAvg(lung_r)
lung_l_accuracy = calculateAvg(lung_l)

# Calculate True IoU for all Organs
liver_accuracy_gt = calculateAvgWith0IoU(liver)
kidney_r_accuracy_gt = calculateAvgWith0IoU(kidney_r)
kidney_l_accuracy_gt = calculateAvgWith0IoU(kidney_l)
femur_r_accuracy_gt = calculateAvgWith0IoU(femur_r)
femur_l_accuracy_gt = calculateAvgWith0IoU(femur_l)
bladder_accuracy_gt = calculateAvgWith0IoU(bladder)
heart_accuracy_gt = calculateAvgWith0IoU(heart)
spleen_accuracy_gt = calculateAvgWith0IoU(spleen)
pancreas_accuracy_gt = calculateAvgWith0IoU(pancreas)
lung_r_accuracy_gt = calculateAvgWith0IoU(lung_r)
lung_l_accuracy_gt = calculateAvgWith0IoU(lung_l)

# Calculate Organs Detection Rate for all Organs
liver_detection_rate = calculateDetectRate(liver)
kidney_r_detection_rate = calculateDetectRate(kidney_r)
kidney_l_detection_rate = calculateDetectRate(kidney_l)
femur_r_detection_rate = calculateDetectRate(femur_r)
femur_l_detection_rate = calculateDetectRate(femur_l)
bladder_detection_rate = calculateDetectRate(bladder)
heart_detection_rate = calculateDetectRate(heart)
spleen_detection_rate = calculateDetectRate(spleen)
pancreas_detection_rate = calculateDetectRate(pancreas)
lung_r_detection_rate = calculateDetectRate(lung_r)
lung_l_detection_rate = calculateDetectRate(lung_l)

detected_organs = 11 # Number of Detected Organs
# Print Accuracy with Detected Organs
print("IoU with Total Detected Organs in Results")
global_accuracy = (liver_accuracy + kidney_r_accuracy + kidney_l_accuracy + femur_r_accuracy +
                   femur_l_accuracy + bladder_accuracy + heart_accuracy + spleen_accuracy +
                   pancreas_accuracy + lung_r_accuracy + lung_l_accuracy) / detected_organs
print("liver accuracy: ", liver_accuracy)
print("kidney right accuracy: ", kidney_r_accuracy)
print("kidney left accuracy: ", kidney_l_accuracy)
print("femur right accuracy: ", femur_r_accuracy)
print("femur left accuracy: ", femur_l_accuracy)
print("bladder accuracy: ", bladder_accuracy)
print("heart accuracy: ", heart_accuracy)
print("spleen accuracy: ", spleen_accuracy)
print("pancreas accuracy: ", pancreas_accuracy)
print("lung right accuracy: ", lung_r_accuracy)
print("lung left accuracy: ", lung_l_accuracy)
print("global accuracy: ", global_accuracy)
print("")

# Print Accuracy with Ground Truth Organs
print("IoU with Ground Truth Organs")
global_accuracy_gt = (liver_accuracy_gt + kidney_r_accuracy_gt + kidney_l_accuracy_gt + femur_r_accuracy_gt
                      + femur_l_accuracy_gt + bladder_accuracy_gt + heart_accuracy_gt + spleen_accuracy_gt +
                      pancreas_accuracy_gt + lung_r_accuracy_gt + lung_l_accuracy_gt) / detected_organs
print("liver accuracy: ", liver_accuracy_gt)
print("kidney right accuracy: ", kidney_r_accuracy_gt)
print("kidney left accuracy: ", kidney_l_accuracy_gt)
print("femur right accuracy: ", femur_r_accuracy_gt)
print("femur left accuracy: ", femur_l_accuracy_gt)
print("bladder accuracy: ", bladder_accuracy_gt)
print("heart accuracy: ", heart_accuracy_gt)
print("spleen accuracy: ", spleen_accuracy_gt)
print("pancreas accuracy: ", pancreas_accuracy_gt)
print("lung right accuracy: ", lung_r_accuracy_gt)
print("lung left accuracy: ", lung_l_accuracy_gt)
print("global accuracy: ", global_accuracy_gt)
print("")

# Print Detection Rate
print("Organ Detection Rate")
global_detection_rate = (liver_detection_rate + kidney_r_detection_rate + kidney_l_detection_rate + femur_r_detection_rate
                         + femur_l_detection_rate + bladder_detection_rate + heart_detection_rate + spleen_detection_rate
                         + pancreas_detection_rate + lung_r_detection_rate + lung_l_detection_rate) / detected_organs
print("liver detection rate: ", liver_detection_rate)
print("kidney detection rate: ", kidney_r_detection_rate)
print("kidney detection rate: ", kidney_l_detection_rate)
print("femur detection rate: ", femur_r_detection_rate)
print("femur detection rate: ", femur_l_detection_rate)
print("bladder detection rate: ", bladder_detection_rate)
print("heart detection rate: ", heart_detection_rate)
print("spleen detection rate: ", spleen_detection_rate)
print("pancreas detection rate: ", pancreas_detection_rate)
print("lung right detection rate: ", lung_r_detection_rate)
print("lung left detection rate: ", lung_l_detection_rate)
print("global detection rate: ", global_detection_rate)

# Write all Results to Output Files
with open(detected_organs_IoU_file, "w+") as f:
    f.writelines("liver: " + str(liver_accuracy)+"\n")
    f.writelines("kidney right : " + str(kidney_r_accuracy) + "\n")
    f.writelines("kidney left: " + str(kidney_l_accuracy) + "\n")
    f.writelines("femur right: " + str(femur_r_accuracy) + "\n")
    f.writelines("femur left: " + str(femur_l_accuracy) + "\n")
    f.writelines("bladder: " + str(bladder_accuracy) + "\n")
    f.writelines("heart: " + str(heart_accuracy) + "\n")
    f.writelines("spleen: " + str(spleen_accuracy) + "\n")
    f.writelines("pancreas: " + str(pancreas_accuracy) + "\n")
    f.writelines("lung right: " + str(lung_r_accuracy) + "\n")
    f.writelines("lung left: " + str(lung_l_accuracy) + "\n")
    f.writelines("global: " + str(global_accuracy) + "\n")

with open(groundtruth_organs_IoU_file, "w+") as f:
    f.writelines("liver: " + str(liver_accuracy_gt)+"\n")
    f.writelines("kidney right : " + str(kidney_r_accuracy_gt) + "\n")
    f.writelines("kidney left: " + str(kidney_l_accuracy_gt) + "\n")
    f.writelines("femur right: " + str(femur_r_accuracy_gt) + "\n")
    f.writelines("femur left: " + str(femur_l_accuracy_gt) + "\n")
    f.writelines("bladder: " + str(bladder_accuracy_gt) + "\n")
    f.writelines("heart: " + str(heart_accuracy_gt) + "\n")
    f.writelines("spleen: " + str(spleen_accuracy_gt) + "\n")
    f.writelines("pancreas: " + str(pancreas_accuracy_gt) + "\n")
    f.writelines("lung right: " + str(lung_r_accuracy_gt) + "\n")
    f.writelines("lung left: " + str(lung_l_accuracy_gt) + "\n")
    f.writelines("global: " + str(global_accuracy_gt) + "\n")

with open(organ_detected_rate_file, "w+") as f:
    f.writelines("liver: " + str(liver_detection_rate)+"\n")
    f.writelines("kidney right : " + str(kidney_r_detection_rate) + "\n")
    f.writelines("kidney left: " + str(kidney_l_detection_rate) + "\n")
    f.writelines("femur right: " + str(femur_r_detection_rate) + "\n")
    f.writelines("femur left: " + str(femur_l_detection_rate) + "\n")
    f.writelines("bladder: " + str(bladder_detection_rate) + "\n")
    f.writelines("heart: " + str(heart_detection_rate) + "\n")
    f.writelines("spleen: " + str(spleen_detection_rate) + "\n")
    f.writelines("pancreas: " + str(pancreas_detection_rate) + "\n")
    f.writelines("lung right: " + str(lung_r_detection_rate) + "\n")
    f.writelines("lung left: " + str(lung_l_detection_rate) + "\n")
    f.writelines("global: " + str(global_detection_rate) + "\n")
