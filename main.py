import os
import torch
import cv2
import json
import glob
import warnings
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from detectron2.engine import DefaultTrainer, DefaultPredictor, launch
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.utils.logger import setup_logger


# Suppress warnings
warnings.filterwarnings("ignore")
torch.backends.cudnn.benchmark = True
setup_logger()

# Register datasets
def register_datasets():
    register_coco_instances("my_dataset_train", {}, "datasets/my_datasets/train.json", "datasets/my_datasets/train")
    register_coco_instances("my_dataset_test", {}, "datasets/my_datasets/test.json", "datasets/my_datasets/test")

# Configuration setup
def setup_cfg(backbone, output_dir):
    cfg = get_cfg()
    cfg.merge_from_file(f"configs/{backbone}")

    # --- Dataset configurations ---
    cfg.DATASETS.TRAIN = ("my_dataset_train",)
    cfg.DATASETS.TEST = ("my_dataset_test",)
    cfg.DATALOADER.NUM_WORKERS = 24  # Number of data loading threads

    # --- Model configurations ---
    # Set the number of classes for different architectures
    if "retinanet" in backbone:
        cfg.MODEL.RETINANET.NUM_CLASSES = 1
    elif "rpn" in backbone:
        cfg.MODEL.RPN.NUM_CLASSES = 1    
    else:
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

    # Anchor generator settings
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[16], [32], [64], [128], [256]]  # Anchor sizes
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.5, 1.0, 2.0]]  # Anchor aspect ratios

    # --- Input configurations ---
    cfg.INPUT.MIN_SIZE_TRAIN = 480  # Minimum image size for training
    cfg.INPUT.MAX_SIZE_TRAIN = 1024  # Maximum image size for training
    cfg.INPUT.MIN_SIZE_TEST = 480  # Minimum image size for testing
    cfg.INPUT.MAX_SIZE_TEST = 1024  # Maximum image size for testing
    cfg.INPUT.RANDOM_FLIP = "horizontal"  # Randomly flip images horizontally
    cfg.INPUT.CROP.ENABLED = True  # Enable cropping
    cfg.INPUT.CROP.SIZE = [0.5, 0.5]  # Crop size

    # --- Solver configurations ---
    cfg.SOLVER.IMS_PER_BATCH = 8  # Number of images per batch (consider increasing if GPU memory allows)
    cfg.SOLVER.BASE_LR = 0.001  # Base learning rate (tune for optimal performance)
    cfg.SOLVER.WARMUP_ITERS = 1000  # Number of warmup iterations
    cfg.SOLVER.MAX_ITER = 3000  # Maximum number of iterations
    cfg.SOLVER.STEPS = [6000, 8000]  # Learning rate decay steps (tune for optimal performance)
    cfg.SOLVER.GAMMA = 0.1  # Learning rate decay factor (tune for optimal performance)
    cfg.SOLVER.CHECKPOINT_PERIOD = 500  # Save a checkpoint every 500 iterations
    cfg.SOLVER.AMP.ENABLED = True  # Enable Automatic Mixed Precision training

    # --- ROI Heads configurations ---
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512  # Batch size per image for ROI heads
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.001  # Score threshold for testing

    # --- RetinaNet configurations ---
    cfg.MODEL.RETINANET.FOCAL_LOSS_GAMMA = 2.0  # Focal loss gamma
    cfg.MODEL.RETINANET.FOCAL_LOSS_ALPHA = 0.25  # Focal loss alpha
    cfg.MODEL.RETINANET.PRE_NMS_TOPK_TRAIN = 2000  # Pre-NMS topk for training
    cfg.MODEL.RETINANET.POST_NMS_TOPK_TRAIN = 1000  # Post-NMS topk for training

    # --- RPN configurations ---
    cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN = 2000  # Pre-NMS topk for training
    cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 1000  # Pre-NMS topk for testing
    cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN = 1000  # Post-NMS topk for training
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 500  # Post-NMS topk for testing
    cfg.MODEL.RPN.IOU_THRESHOLDS = [0.3, 0.7]  # IOU thresholds for NMS
    cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE = 512  # Batch size per image for RPN
    cfg.MODEL.RPN.POSITIVE_FRACTION = 0.5  # Target fraction of foreground samples

    # --- Output configurations ---
    cfg.OUTPUT_DIR = output_dir
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    return cfg

# Trainer customization
class CustomTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        return COCOEvaluator(dataset_name, tasks=("bbox",), distributed=True, output_dir=cfg.OUTPUT_DIR)

# Function to load ground truth and predictions
def load_data(cfg):
    with open(os.path.join(cfg.OUTPUT_DIR,"coco_instances_results.json")) as f:
        predictions = json.load(f)
    with open("./datasets/my_datasets/test.json") as f:
        ground_truth = json.load(f)
    return ground_truth, predictions

# Function to calculate metrics and log to TensorBoard
def calculate_metrics(cfg, predictor, writer, iou_threshold=0.5):       
    evaluator = COCOEvaluator("my_dataset_test", tasks=("bbox",), distributed=True, output_dir=cfg.OUTPUT_DIR)
    val_loader = build_detection_test_loader(cfg, "my_dataset_test")
    inference_on_dataset(predictor.model, val_loader, evaluator)

    ground_truth, predictions = load_data(cfg)

    gt_boxes = [ann['bbox'] for ann in ground_truth['annotations']]
    gt_matched = [False] * len(gt_boxes)  

    pred_boxes = [pred['bbox'] for pred in predictions]
    pred_scores = [pred['score'] for pred in predictions]

    sorted_indices = np.argsort(pred_scores)[::-1]
    pred_boxes = [pred_boxes[i] for i in sorted_indices]

    tp, fp, fn = 0, 0, len(gt_boxes)

    for pred_box in pred_boxes:
        matched = False
        for i, gt_box in enumerate(gt_boxes):
            if not gt_matched[i]:  
                iou = compute_iou(pred_box, gt_box)
                if iou >= iou_threshold:
                    tp += 1
                    gt_matched[i] = True
                    fn -= 1
                    matched = True
                    break
        if not matched:
            fp += 1
    print(f"True Positives: {tp}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")

    # Tính Precision, Recall, F1-Score
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")

    # Log to TensorBoard
    iteration = cfg.SOLVER.MAX_ITER
    writer.add_scalar("Evaluation/Precision", precision, iteration)
    writer.add_scalar("Evaluation/Recall", recall, iteration)
    writer.add_scalar("Evaluation/F1-Score", f1, iteration)

# Function to calculate IoU
def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    # Compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # Compute the area of both the prediction and ground truth rectangles
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]

    # Compute the intersection over union
    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou

# Training function
def train_backbone(backbone, output_dir, resume):
    cfg = setup_cfg(backbone, output_dir)
    writer = SummaryWriter(log_dir=cfg.OUTPUT_DIR)
    trainer = CustomTrainer(cfg)
    trainer.resume_or_load(resume=resume)
    trainer.train()

    # Run evaluation and log metrics at the end of training
    predictor = DefaultPredictor(cfg)
    calculate_metrics(cfg, predictor, writer)
    writer.close()
    test_and_visualize(cfg,predictor)

# Main function
def main():
    register_datasets()

    backbones = [
        # Faster R-CNN backbones
        "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml",
        "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml",
        "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml",

        # RetinaNet backbones
        "COCO-Detection/retinanet_R_50_FPN_3x.yaml",
        "COCO-Detection/retinanet_R_101_FPN_3x.yaml",

        # RPN and Fast R-CNN backbones
        "COCO-Detection/rpn_R_50_FPN_1x.yaml",
        "COCO-Detection/fast_rcnn_R_50_FPN_1x.yaml"
    ]

    output_dir = "./output"
    resume_training = False

    for backbone in backbones:
        print(f"\nTraining with backbone: {backbone}")
        backbone_name = backbone.split("/")[-1].replace(".yaml", "")
        train_backbone(backbone, os.path.join(output_dir, backbone_name), resume_training)

# Visualization and testing
def test_and_visualize(cfg, predictor):
    test_metadata = MetadataCatalog.get("my_dataset_test")
    output_dir = cfg.OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    for image_name in glob.glob("datasets/my_datasets/test/*.jpg"):
        img = cv2.imread(image_name)
        outputs = predictor(img)
        v = Visualizer(
            img[:, :, ::-1],
            metadata=test_metadata,
            scale=1.0,
            instance_mode=ColorMode.IMAGE
        )
        instances = outputs["instances"].to("cpu")       
        output_path = os.path.join(output_dir, f"prediction_{os.path.basename(image_name)}")
        cv2.imwrite(output_path, vis.get_image()[:, :, ::-1])
        print(f"Prediction saved to {output_path}")

if __name__ == "__main__":
    num_gpus = 2
    launch(
        main,
        num_gpus_per_machine=num_gpus,
        num_machines=1,
        machine_rank=0,
        dist_url="auto"
    )