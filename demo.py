import onnxruntime as ort
from pathlib import Path
import numpy as np
import cv2
import argparse
import os
import glob
import torch
import torchvision
from typing import Tuple, Optional

class OrtBase:
    def __init__(self, model_path, ort_custom_op_path ="") -> None:
        session_options = ort.SessionOptions()
        if ort_custom_op_path:
            session_options.register_custom_ops_library(ort_custom_op_path)

        # TODO - GPU지원
        self.device_type = "cpu"
        self.device_id = -1
        self.session = ort.InferenceSession(model_path, session_options, providers=['CPUExecutionProvider'])
        session_input = self.session.get_inputs()[0]
        self.img_size = session_input.shape[2]
        self.input_name = session_input.name
        self.output_names = [_.name for _ in self.session.get_outputs()]

    def ort_inference(self, input):
        return self.session.run(None, {self.input_name: input})
    
class IDCardDetectionBase(OrtBase):
    def __init__(self, model_path) -> None:
        super().__init__(model_path)
        self.conf_thres=0.85
        self.iou_thres=0.5

    def xywh2xyxy(self, x):
        # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
        return y
    
    def box_area(self, box):
        # box = xyxy(4,n)
        return (box[2] - box[0]) * (box[3] - box[1])

    def box_iou(self, box1, box2, eps=1e-7):
        # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
        """
        Return intersection-over-union (Jaccard index) of boxes.
        Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
        Arguments:
            box1 (Tensor[N, 4])
            box2 (Tensor[M, 4])
        Returns:
            iou (Tensor[N, M]): the NxM matrix containing the pairwise
                IoU values for every element in boxes1 and boxes2
        """

        # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
        (a1, a2), (b1, b2) = box1[:, None].chunk(2, 2), box2.chunk(2, 1)
        inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)

        # IoU = inter / (area1 + area2 - inter)
        return inter / (self.box_area(box1.T)[:, None] + self.box_area(box2.T) - inter + eps)

    
    def non_max_suppression(self,
                            prediction,
                            conf_thres=0.25,
                            iou_thres=0.45,
                            classes=None,
                            agnostic=False,
                            multi_label=False,
                            labels=(),
                            max_det=300,
                            nm=0,  # number of masks
                            ):
        """Non-Maximum Suppression (NMS) on inference results to reject overlapping detections

        Returns:
            list of detections, on (n,6) tensor per image [xyxy, conf, cls]
        """

        if isinstance(prediction, (list, tuple)):  # YOLOv5 model in validation model, output = (inference_out, loss_out)
            prediction = prediction[0]  # select only inference output

        device = prediction.device
        mps = 'mps' in device.type  # Apple MPS
        if mps:  # MPS not fully supported yet, convert tensors to CPU before NMS
            prediction = prediction.cpu()
        bs = prediction.shape[0]  # batch size
        nc = prediction.shape[2] - nm - 5  # number of classes
        xc = prediction[..., 4] > conf_thres  # candidates

        # Checks
        assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
        assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

        # Settings
        # min_wh = 2  # (pixels) minimum box width and height
        max_wh = 7680  # (pixels) maximum box width and height
        max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
        #time_limit = 0.5 + 0.05 * bs  # seconds to quit after
        redundant = True  # require redundant detections
        multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
        merge = False  # use merge-NMS

        #t = time.time()
        mi = 5 + nc  # mask start index
        output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
        for xi, x in enumerate(prediction):  # image index, image inference
            # Apply constraints
            # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
            x = x[xc[xi]]  # confidence

            # Cat apriori labels if autolabelling
            if labels and len(labels[xi]):
                lb = labels[xi]
                v = torch.zeros((len(lb), nc + nm + 5), device=x.device)
                v[:, :4] = lb[:, 1:5]  # box
                v[:, 4] = 1.0  # conf
                v[range(len(lb)), lb[:, 0].long() + 5] = 1.0  # cls
                x = torch.cat((x, v), 0)

            # If none remain process next image
            if not x.shape[0]:
                continue

            # Compute conf
            x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

            # Box/Mask
            box = self.xywh2xyxy(x[:, :4])  # center_x, center_y, width, height) to (x1, y1, x2, y2)
            mask = x[:, mi:]  # zero columns if no masks

            # Detections matrix nx6 (xyxy, conf, cls)
            if multi_label:
                i, j = (x[:, 5:mi] > conf_thres).nonzero(as_tuple=False).T
                x = torch.cat((box[i], x[i, 5 + j, None], j[:, None].float(), mask[i]), 1)
            else:  # best class only
                conf, j = x[:, 5:mi].max(1, keepdim=True)
                x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]

            # Filter by class
            if classes is not None:
                x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

            # Apply finite constraint
            # if not torch.isfinite(x).all():
            #     x = x[torch.isfinite(x).all(1)]

            # Check shape
            n = x.shape[0]  # number of boxes
            if not n:  # no boxes
                continue
            elif n > max_nms:  # excess boxes
                x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence
            else:
                x = x[x[:, 4].argsort(descending=True)]  # sort by confidence

            # Batched NMS
            c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
            boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
            i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
            if i.shape[0] > max_det:  # limit detections
                i = i[:max_det]
            if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
                # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
                iou = self.box_iou(boxes[i], boxes) > iou_thres  # iou matrix
                weights = iou * scores[None]  # box weights
                x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
                if redundant:
                    i = i[iou.sum(1) > 1]  # require redundancy

            output[xi] = x[i]
            if mps:
                output[xi] = output[xi].to(device)
            #if (time.time() - t) > time_limit:
            #    print(f'WARNING ⚠️ NMS time limit {time_limit:.3f}s exceeded')
            #    break  # time limit exceeded

        return output

    def _inference(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        pred, proto = self.ort_inference(img)
        pred = self.non_max_suppression(torch.from_numpy(pred), self.conf_thres, self.iou_thres, nm=32)[0].numpy()
        return pred, proto

    def resize_preserving_aspect_ratio(self, img: np.ndarray, img_size: int, scale_ratio=1.0) -> Tuple[np.ndarray, float]:
        # Resize preserving aspect ratio. scale_ratio is the scaling ratio of the img_size.
        h, w = img.shape[:2]
        scale = img_size // scale_ratio / max(h, w)
        if scale != 1:
            interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=interpolation)
        return img, scale

    def _transform_image(self, img: np.ndarray, scale_ratio=1.0) -> Tuple[np.ndarray, float]:
        """
        Resizes the input image to fit img_size while preserving aspect ratio.
        (HWC to CHW, BGR to RGB, 0~1 normalization, and adding batch dimension)
        """
        img, scale = self.resize_preserving_aspect_ratio(img, self.img_size, scale_ratio)

        pad = (0, self.img_size - img.shape[0], 0, self.img_size - img.shape[1])
        img = cv2.copyMakeBorder(img, *pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        # HWC to BCHW, BGR to RGB, uint8 to fp32
        img = np.ascontiguousarray(np.expand_dims(img.transpose((2, 0, 1))[::-1], 0), np.float32)
        img /= 255  # 0~255 to 0~1
        return img, scale
    
    def sigmoid(self, x: np.array) -> np.ndarray:
        return 1 / (1 + np.exp(-x))
    
    def crop_mask(self, masks: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        n, h, w = masks.shape
        x1, y1, x2, y2 = np.split(boxes[:, :, None], 4, 1)  # x1 shape(1,1,n)
        r = np.arange(w, dtype=x1.dtype)[None, None, :]  # rows shape(1,w,1)
        c = np.arange(h, dtype=x1.dtype)[None, :, None]  # cols shape(h,1,1)

        return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))

    def process_mask(self,
                     protos: np.ndarray,
                     masks_in: np.ndarray,
                     bboxes: np.ndarray,
                     shape: Tuple[int, int],
                     upsample=False ) -> np.ndarray:
        c, mh, mw = protos.shape
        ih, iw = shape
        masks = self.sigmoid(masks_in @ protos.reshape((c, -1))).reshape((-1, mh, mw))

        downsampled_bboxes = bboxes.copy()
        downsampled_bboxes[:, 0] *= mw / iw
        downsampled_bboxes[:, 2] *= mw / iw
        downsampled_bboxes[:, 1] *= mh / ih
        downsampled_bboxes[:, 3] *= mh / ih

        masks = self.crop_mask(masks, downsampled_bboxes)  # CHW
        if upsample:
            masks = np.array([cv2.resize(mask, shape) for mask in masks])
        return np.where(masks > 0.5, 255, 0).astype(np.uint8)

    def detection(self, img:np.ndarray):
        original_img_shape = img.shape[:2]
        img, scale = self._transform_image(img)
        pred, proto = self._inference(img)
        if pred.shape[0] > 0:
            # Process mask
            masks = self.process_mask(proto[0], pred[:, 6:], pred[:, :4], (self.img_size, self.img_size), upsample=True)
            masks = np.array([cv2.resize(mask, None, fx=1 / scale, fy=1 / scale, interpolation=cv2.INTER_NEAREST_EXACT)
                              for mask in masks])
            masks = masks[:, :original_img_shape[0], :original_img_shape[1]]

            # Rescale bboxes from inference size to input image size
            pred[:, :4] /= scale
            pred[:, [0, 2]] = pred[:, [0, 2]].clip(0, original_img_shape[1])  # x1, x2
            pred[:, [1, 3]] = pred[:, [1, 3]].clip(0, original_img_shape[0])  # y1, y2

            # Parse bbox and confidence
            bbox = pred[:, :4].round().astype(np.int32).tolist()
            conf = pred[:, 4].tolist()
            cls = pred[:, 5].tolist()
            return bbox, conf, cls, masks
        else:
            return None

class Yolov5IDCard(IDCardDetectionBase): 
    def __init__(self, model_path):
        super().__init__(model_path)

class IDCardDetection:
    def __init__(self, model_path, model_type) -> None:
        if (model_type == "yolov5"):
            self.impl = Yolov5IDCard(model_path)
        else:
            raise Exception(f"Not found Model Type:{model_type}")

    def warpPerspectivePoint(self, point: tuple, M: np.ndarray):
        x = point[0]
        y = point[1]
        
        point_mat = np.array([[x,y,1]])
        
        N = np.dot(M, point_mat.T).T
        
        transformed_x = N[0][0] / N[0][2]
        transformed_y = N[0][1] / N[0][2]
        
        return transformed_x, transformed_y

    def align_idcard(self, img: np.ndarray, keypoints: np.ndarray, cls: list, dsize_factor: int = None) -> np.ndarray:
        if cls[0] == 0:
            idcard_ratio = np.array((86, 54))
        elif cls[0] == 1:
            idcard_ratio = np.array((125, 88))
        elif cls[0] == 2:
            idcard_ratio = np.array((125, 88))
        else:
            raise ValueError(f'Wrong cls: {cls}')

        if dsize_factor is None:
            dsize_factor = round(np.sqrt(cv2.contourArea(np.expand_dims(keypoints, 1))) / idcard_ratio[0])

        dsize = idcard_ratio * dsize_factor  # idcard size unit: mm
        dst = np.array(((0, 0), (0, dsize[1]), dsize, (dsize[0], 0)), np.float32)

        M = cv2.getPerspectiveTransform(keypoints.astype(np.float32), dst)
        img = cv2.warpPerspective(img, M, dsize)
        
        return img

    def sort_corner_order(self, quadrangle: np.ndarray) -> np.ndarray:
        assert quadrangle.shape == (4, 1, 2), f'Invalid quadrangle shape: {quadrangle.shape}'

        quadrangle = quadrangle.squeeze(1)
        moments = cv2.moments(quadrangle)
        mcx = round(moments['m10'] / moments['m00'])  # mass center x
        mcy = round(moments['m01'] / moments['m00'])  # mass center y
        keypoints = np.zeros((4, 2), np.int32)
        for point in quadrangle:
            if point[0] < mcx and point[1] < mcy:
                keypoints[0] = point
            elif point[0] < mcx and point[1] > mcy:
                keypoints[1] = point
            elif point[0] > mcx and point[1] > mcy:
                keypoints[2] = point
            elif point[0] > mcx and point[1] < mcy:
                keypoints[3] = point
        return keypoints

    def get_keypoints(self, 
                      masks: np.ndarray,
                      morph_ksize=21,
                      contour_thres=0.02,
                      poly_thres=0.03) -> Optional[np.ndarray]:
        # If multiple masks, select the mask with the largest object.
        if masks.shape[0] > 1:
            masks = masks[np.count_nonzero(masks.reshape(masks.shape[0], -1), axis=1).argmax()]

        # Post-process mask
        if len(masks.shape) == 3:
            masks = masks.squeeze(0)

        # Perform morphological transformation
        masks = cv2.morphologyEx(masks, cv2.MORPH_OPEN,
                                cv2.getStructuringElement(cv2.MORPH_RECT, (morph_ksize, morph_ksize)))
        # Find contours (+remove noise)
        contours, _ = cv2.findContours(masks, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
        contours = [contour for contour in contours
                    if cv2.contourArea(contour) > (masks.shape[0] * masks.shape[1] * contour_thres)]
        # Approximate quadrangles (+remove noise)
        quadrangles = [cv2.approxPolyDP(contour, cv2.arcLength(contour, True) * poly_thres, True) for contour in contours]
        quadrangles = [quad for quad in quadrangles if quad.shape == (4, 1, 2)]

        if len(quadrangles) == 1:
            keypoints = self.sort_corner_order(quadrangles[0])
            return keypoints
        else:
            return None
        
    def draw_prediction(self, img: np.ndarray, bbox: list, conf: list, landmarks: list = None, thickness=2, hide_conf=False):
        # Draw prediction on the image. If the landmarks is None, only draw the bbox.
        assert img.ndim == 3, f'img dimension is invalid: {img.ndim}'
        assert img.dtype == np.uint8, f'img dtype must be uint8, got {img.dtype}'
        assert img.shape[-1] == 3, 'Pass BGR images. Other Image formats are not supported.'
        assert len(bbox) == len(conf), 'bbox and conf must be equal length.'
        if landmarks is None:
            landmarks = [None] * len(bbox)
        assert len(bbox) == len(conf) == len(landmarks), 'bbox, conf, and landmarks must be equal length.'

        bbox_color = (0, 255, 0)
        conf_color = (0, 255, 0)
        landmarks_colors = ((0, 165, 255), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255))
        for bbox_one, conf_one, landmarks_one in zip(bbox, conf, landmarks):
            # Draw bbox
            x1, y1, x2, y2 = bbox_one
            cv2.rectangle(img, (x1, y1), (x2, y2), bbox_color, thickness, cv2.LINE_AA)

            # Text confidence
            if not hide_conf:
                cv2.putText(img, f'{conf_one:.2f}', (x1, y1 - 2), None, 0.6, conf_color, thickness, cv2.LINE_AA)

            # Draw landmarks
            if landmarks_one is not None:
                for point_x, point_y, color in zip(landmarks_one[::2], landmarks_one[1::2], landmarks_colors):
                    cv2.circle(img, (point_x, point_y), 2, color, cv2.FILLED)

    def draw_segmentation_mask(self,
                               img: np.ndarray,
                               masks: np.ndarray,
                               colors: list,
                               alpha=0.4,
                               gamma=50) -> np.ndarray:
        assert img.dtype == np.uint8, f'The images dtype must be uint8, got {img.dtype}'
        assert img.ndim == 3, 'The images must be of shape (H, W, C)'
        assert img.shape[2] == 3, 'Pass BGR images. Other Image formats are not supported'
        assert img.shape[:2] == masks.shape[-2:], 'The images and the masks must have the same height and width'
        assert masks.ndim == 3, 'The masks must be of shape (N, H, W)'
        assert masks.dtype == np.uint8, f'The masks must be of dtype uint8. Got {masks.dtype}'
        assert 0 <= alpha <= 1, 'alpha must be between 0 and 1. 0 means full transparency, 1 means no transparency'
        assert len(colors[0]) == 3, 'The colors must be BGR format'

        mask = masks.sum(0, np.uint8)
        mask[mask > 1] = 1

        h, w = mask.shape
        colored_mask = np.zeros((h, w, 3), dtype=np.uint8)
        r = colored_mask[:, :, 0]
        g = colored_mask[:, :, 1]
        b = colored_mask[:, :, 2]
        for i, color in enumerate(colors, 1):
            r[mask == i] = color[0]
            g[mask == i] = color[1]
            b[mask == i] = color[2]

        alpha_colored_mask = cv2.addWeighted(img, alpha, colored_mask, 1 - alpha, gamma)
        return alpha_colored_mask

    def detection(self, image, draw_prediction = False, outpath = "", file_name = ""):
        draw_prediction = (draw_prediction and outpath and file_name)
        tmp_out_name = os.path.splitext(os.path.basename(file_name))[0]
        predict = self.impl.detection(image)
        if not predict:
            return None, None 

        bbox, conf, cls, masks = predict

        # Align idcard
        keypoints = self.get_keypoints(masks)
        if keypoints is not None:
            aligned_img = self.align_idcard(image, keypoints, cls)
            if draw_prediction:
                align_file_name = os.path.join(outpath, f"{tmp_out_name}_align.jpg")
                cv2.imwrite(align_file_name, aligned_img)

        if predict and draw_prediction:
            draw_img = image.copy()
            self.draw_prediction(draw_img, bbox, conf, None, 3, False)
            draw_img = self.draw_segmentation_mask(draw_img, masks, [(56, 56, 255)])
            det_file_name = os.path.join(outpath, f"{tmp_out_name}_det.jpg")
            cv2.imwrite(det_file_name, draw_img)

        return predict, aligned_img
    
def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument("-d", "--idcard_model_path", default="yolov5x-seg-id-dr-pp-best.onnx", help="ID Card Detection Model")
    parse.add_argument("-c", "--idcard_model_type", default="yolov5", help="ID Card Detection Model Type")
    
    return parse.parse_args()

def extract_high_quality_frames(video_path, quality_threshold=100):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Cannot open video.")
        return []

    high_quality_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 프레임의 화질을 평가
        variance_of_laplacian = cv2.Laplacian(frame, cv2.CV_64F).var()
        if variance_of_laplacian > quality_threshold:
            high_quality_frames.append(frame)
    
    cap.release()
    return high_quality_frames


def align_images(input_folder, output_folder):
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(exist_ok=True)  # 출력 폴더 생성

    # SIFT 초기화
    sift = cv2.SIFT_create()

    # 첫 번째 이미지 로드 및 SIFT 특징 추출 (그레이스케일)
    base_image_path = next(input_folder.glob('*.jpg'))
    base_image_color = cv2.imread(str(base_image_path))  # 컬러 이미지 로드
    base_image_gray = cv2.cvtColor(base_image_color, cv2.COLOR_BGR2GRAY)  # 그레이스케일 변환
    base_keypoints, base_descriptors = sift.detectAndCompute(base_image_gray, None)

    image_files = list(input_folder.glob('*.jpg'))
    # 이미지 목록을 첫 이미지부터 순서대로 처리
    for image_path in image_files:
        # 컬러 이미지 로드 및 그레이스케일 변환
        image_color = cv2.imread(str(image_path))
        image_gray = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)

        # SIFT 특징 추출
        keypoints, descriptors = sift.detectAndCompute(image_gray, None)

        # FLANN 매처 설정
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        # 매칭 및 좋은 매칭 필터링
        matches = flann.knnMatch(descriptors, base_descriptors, k=2)
        good_matches = [m for m, n in matches if m.distance < 0.7*n.distance]
        src_pts = np.float32([keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([base_keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # 호모그래피 계산
        M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        # 컬러 이미지에 호모그래피 적용
        h, w, _ = base_image_color.shape
        aligned_image_color = cv2.warpPerspective(image_color, M, (w, h))

        # 변환된 컬러 이미지 저장
        output_image_path = output_folder / f"aligned_{image_path.name}"
        cv2.imwrite(str(output_image_path), aligned_image_color)

        print(f"Image {image_path.name} aligned and saved to {output_image_path}")

def generate_max_mean_min_images(input_folder, output_folder):
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(exist_ok=True)

    # Initialize lists to hold the image data
    image_data = []

    # Read images and append the image data to the list
    for image_path in sorted(input_folder.glob('*.jpg')):
        img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if img is not None:
            image_data.append(img)
        else:
            print(f"Failed to load image: {image_path}")

    # Stack images along a new dimension
    stack = np.stack(image_data, axis=3)

    # Calculate max, mean, and min images
    max_img = np.max(stack, axis=3)
    mean_img = np.mean(stack, axis=3).astype(np.uint8)
    min_img = np.min(stack, axis=3)

    # Save the generated images
    cv2.imwrite(str(output_folder / 'max_image.jpg'), max_img)
    cv2.imwrite(str(output_folder / 'mean_image.jpg'), mean_img)
    cv2.imwrite(str(output_folder / 'min_image.jpg'), min_img)
    cv2.imwrite(str(output_folder / 'max_min_image.jpg'), max_img-min_img)
    cv2.imwrite(str(output_folder / 'max_mean_image.jpg'), max_img-mean_img)
    cv2.imwrite(str(output_folder / 'mean_min_image.jpg'), mean_img-min_img)
    print(f"Saved max, mean, and min images in {str(output_folder)}")






if __name__ == "__main__":
    args = parse_args()
    idcard_detection = IDCardDetection(args.idcard_model_path, args.idcard_model_type)

    frames = extract_high_quality_frames('second.mp4', quality_threshold=100)
    

    for index in range(len(frames)) :
        predict, aligned_img = idcard_detection.detection(image=frames[index], draw_prediction=False)

        if not predict:
                print(f"Detection failed:{img}")
                continue
        
        frames[index] = aligned_img     
    
    # align
    align_images('cropped_images', 'aligned_images')

    # output
    generate_max_mean_min_images('aligned_images', 'output')