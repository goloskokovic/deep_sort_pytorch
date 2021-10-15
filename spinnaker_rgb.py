import argparse
import time
from pathlib import Path
import os
import PySpin
import sys
import cv2
import numpy as np

import torch
import torch.backends.cudnn as cudnn

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages, letterbox
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

from detector import build_detector
from deep_sort import build_tracker
from utils.draw import draw_boxes
from utils.parser import get_config
from utils.log import get_logger
from utils.io import write_results



@torch.no_grad()
def detect(options, stride, model, device, names, img0, cfg):
    
    # should be moved out!
    detector = build_detector(cfg, use_cuda=use_cuda)
    deepsort = build_tracker(cfg, use_cuda=use_cuda)
    class_names = detector.class_names
    
    start = time.time()
    img = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
    
    # do detection
    bbox_xywh, cls_conf, cls_ids = detector(img)
    # select person class
    mask = cls_ids == 0
    bbox_xywh = bbox_xywh[mask]
    # bbox dilation just in case bbox too small, delete this line if using a better pedestrian detector
    bbox_xywh[:, 3:] *= 1.2
    cls_conf = cls_conf[mask]
    # do tracking
    outputs = deepsort.update(bbox_xywh, cls_conf, img)
    
    # draw boxes for visualization
    if len(outputs) > 0:
        bbox_tlwh = []
        bbox_xyxy = outputs[:, :4]
        identities = outputs[:, -1]
        ori_im = draw_boxes(ori_im, bbox_xyxy, identities)
        
        for bb_xyxy in bbox_xyxy:
            bbox_tlwh.append(self.deepsort._xyxy_to_tlwh(bb_xyxy))
        
        results.append((idx_frame - 1, bbox_tlwh, identities))

    end = time.time()
    cv2.imshow("test", ori_im)
    cv2.waitKey(1)
    
    # logging
    self.logger.info("time: {:.03f}s, fps: {:.03f}, detection numbers: {}, tracking numbers: {}" \
        .format(end - start, 1 / (end - start), bbox_xywh.shape[0], len(outputs)))
    
    # end tracking
    return
    
    
    half = device.type != 'cpu'
    imgsz = check_img_size(options.img_size, s=stride)  # check img_size
    # Padded resize
    img = letterbox(img0, imgsz, stride=stride)[0]
    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)

    # img = transforms.ToTensor()(np.array(img)).to(device)
    img = torch.from_numpy(np.array(img)).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    
    # Inference
    pred = model(img, augment=options.augment)[0]

    # Apply NMS
    pred = non_max_suppression(pred, options.conf_thres, options.iou_thres, options.classes, options.agnostic_nms,
                               max_det=options.max_det)

    # Process detections
    for i, det in enumerate(pred):  # detections per image
        s = ''
        s += '%gx%g ' % img.shape[2:]  # print string
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

            # Write results
            for *xyxy, conf, cls in reversed(det):
                c = int(cls)  # integer class
                if c == 0:
                    label = None if options.hide_labels else (
                        names[c] if options.hide_conf else f'{names[c]} {conf:.2f}')
                    label = 'Person'
                    plot_one_box(xyxy, img0, label=label, color=colors(c, True), line_thickness=options.line_thickness)

        cv2.imshow('Front Center camera', img0)
        k = cv2.waitKey(1)
        if k == ord('q'):
            cv2.destroyAllWindows()
            exit()


def acquire_images(cam_list, master_camera_sn, options, stride, model, device, names, cfg):
    """
    This function acquires and saves 10 images from each device.

    :param cam_list: List of cameras
    :type cam_list: CameraList
    :param master_camera_sn: master camera id
    :type master_camera_sn: String
    :param options: parser options
    :type options: Options
    :return: True if successful, False otherwise.
    :rtype: bool
    """
    print('*** IMAGE ACQUISITION ***\n')
    try:
        result = True

        # Prepare each camera to acquire images
        #
        # *** NOTES ***
        # For pseudo-simultaneous streaming, each camera is prepared as if it
        # were just one, but in a loop. Notice that cameras are selected with
        # an index. We demonstrate pseudo-simultaneous streaming because true
        # simultaneous streaming would require multiple process or threads,
        # which is too complex for an example.
        #
        for i, cam in enumerate(cam_list):
            device_serial_number = None
            # Retrieve device serial number
            node_device_serial_number = PySpin.CStringPtr(
                cam.GetTLDeviceNodeMap().GetNode('DeviceSerialNumber'))

            if PySpin.IsAvailable(node_device_serial_number) and PySpin.IsReadable(node_device_serial_number):
                device_serial_number = node_device_serial_number.GetValue()

            if device_serial_number is not None and device_serial_number == master_camera_sn:
                # Set acquisition mode to continuous
                node_acquisition_mode = PySpin.CEnumerationPtr(cam.GetNodeMap().GetNode('AcquisitionMode'))
                if not PySpin.IsAvailable(node_acquisition_mode) or not PySpin.IsWritable(node_acquisition_mode):
                    print(
                        'Unable to set acquisition mode to continuous (node retrieval; camera %d). Aborting... \n' % i)
                    return False

                node_acquisition_mode_continuous = node_acquisition_mode.GetEntryByName('Continuous')
                if not PySpin.IsAvailable(node_acquisition_mode_continuous) or not PySpin.IsReadable(
                        node_acquisition_mode_continuous):
                    print('Unable to set acquisition mode to continuous (entry \'continuous\' retrieval %d). \
                    Aborting... \n' % i)
                    return False

                acquisition_mode_continuous = node_acquisition_mode_continuous.GetValue()
                node_acquisition_mode.SetIntValue(acquisition_mode_continuous)
                print('Camera %d acquisition mode set to continuous...' % i)

                # Begin acquiring images
                cam.BeginAcquisition()
                print('Camera %d started acquiring images...' % i)
                while True:
                    try:
                        # Retrieve next received image and ensure image completion
                        image_result = cam.GetNextImage(1000)

                        if image_result.IsIncomplete():
                            pass
                            # print('Image incomplete with image status %d ... \n' % image_result.GetImageStatus())
                        else:
                            width = image_result.GetWidth()
                            height = image_result.GetHeight()
                            # Convert image to RGB
                            image_converted = image_result.Convert(PySpin.PixelFormat_BGR8)
                            im_cv2_format = image_converted.GetData().reshape(height, width, 3)
                            # start = time.time()
                            detect(options, stride, model, device, names, im_cv2_format, cfg)
                            # end = time.time()
                            # print("Inference time: ", str((end-start)*1000))

                        # Release image
                        image_result.Release()
                    except PySpin.SpinnakerException as ex:
                        print('Error: %s' % ex)
                        cam.EndAcquisition()
                        result = False

    except PySpin.SpinnakerException as ex:
        print('Error: %s' % ex)
        result = False

    return result


def run_master_camera(cam_list, master_camera_sn, options, stride, model, device, names, cfg):
    """
    This function acts as the body of the example; please see NodeMapInfo example
    for more in-depth comments on setting up cameras.

    :param cam_list: List of cameras
    :type cam_list: CameraList
    :param master_camera_sn: master camera id
    :type master_camera_sn: String
    :param options: parser options
    :type options: Options
    :return: True if successful, False otherwise.
    :rtype: bool
    """
    try:
        result = True

        # Initialize each camera
        #
        # *** NOTES ***
        # You may notice that the steps in this function have more loops with
        # less steps per loop; this contrasts the AcquireImages() function
        # which has less loops but more steps per loop. This is done for
        # demonstrative purposes as both work equally well.
        #
        # *** LATER ***
        # Each camera needs to be de-initialized once all images have been
        # acquired.
        for i, cam in enumerate(cam_list):
            # Initialize camera
            cam.Init()

        # Acquire images on all cameras
        result &= acquire_images(cam_list, master_camera_sn, options, stride, model, device, names, cfg)

        # Deinitialize each camera
        #
        # *** NOTES ***
        # Again, each camera must be deinitialized separately by first
        # selecting the camera and then deinitializing it.
        for cam in cam_list:
            # Deinitialize camera
            cam.DeInit()

        # Release reference to camera
        # NOTE: Unlike the C++ examples, we cannot rely on pointer objects being automatically
        # cleaned up when going out of scope.
        # The usage of del is preferred to assigning the variable to None.
        del cam

    except PySpin.SpinnakerException as ex:
        print('Error: %s' % ex)
        result = False

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov3.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.20, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum number of detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=2, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    
    # deep sort arguments
    parser.add_argument("VIDEO_PATH", type=str)
    parser.add_argument("--config_mmdetection", type=str, default="./configs/mmdet.yaml")
    parser.add_argument("--config_detection", type=str, default="./configs/yolov3.yaml")
    parser.add_argument("--config_deepsort", type=str, default="./configs/deep_sort.yaml")
    parser.add_argument("--config_fastreid", type=str, default="./configs/fastreid.yaml")
    parser.add_argument("--fastreid", action="store_true")
    parser.add_argument("--mmdet", action="store_true")
    # parser.add_argument("--ignore_display", dest="display", action="store_false", default=True)
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--frame_interval", type=int, default=1)
    parser.add_argument("--display_width", type=int, default=800)
    parser.add_argument("--display_height", type=int, default=600)
    parser.add_argument("--save_path", type=str, default="./output/")
    parser.add_argument("--cpu", dest="use_cuda", action="store_false", default=True)
    parser.add_argument("--camera", action="store", dest="cam", type=int, default="-1")
    opt = parser.parse_args()
    result = True
    
    cfg = get_config()
    if opt.mmdet:
        cfg.merge_from_file(opt.config_mmdetection)
        cfg.USE_MMDET = True
    else:
        cfg.merge_from_file(opt.config_detection)
        cfg.USE_MMDET = False
    cfg.merge_from_file(opt.config_deepsort)
    if args.fastreid:
        cfg.merge_from_file(opt.config_fastreid)
        cfg.USE_FASTREID = True
    else:
        cfg.USE_FASTREID = False

    weights, imgsz = opt.weights, opt.img_size

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names

    if half:
        model.half()  # to FP16

    # Retrieve singleton reference to system object
    system = PySpin.System.GetInstance()

    # Get current library version
    version = system.GetLibraryVersion()
    print('Library version: %d.%d.%d.%d' % (version.major, version.minor, version.type, version.build))

    # Retrieve list of cameras from the system
    cam_list = system.GetCameras()
    num_cameras = cam_list.GetSize()
    print('Number of cameras detected: %d' % num_cameras)

    master_camera_serial_number = "19444627"
    print('Master camera serial number: ', master_camera_serial_number)

    # Finish if there are no cameras
    if num_cameras == 0:
        # Clear camera list before releasing system
        cam_list.Clear()

        # Release system instance
        system.ReleaseInstance()

        print('Not enough cameras!')
        input('Done! Press Enter to exit...')
        return False

    result = run_master_camera(cam_list, master_camera_serial_number, opt, stride, model, device, names, cfg)

    print('Example complete... \n')

    # Clear camera list before releasing system
    cam_list.Clear()

    # Release system instance
    system.ReleaseInstance()

    input('Done! Press Enter to exit...')
    return result


if __name__ == '__main__':
    if main():
        sys.exit(0)
    else:
        sys.exit(1)
