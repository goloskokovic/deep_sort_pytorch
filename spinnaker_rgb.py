import argparse
import time
import PySpin
import sys
import cv2
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), 'thirdparty/fast-reid'))

from detector import build_detector
from deep_sort import build_tracker
from utils.draw import draw_boxes
from utils.parser import get_config
from utils.log import get_logger
from utils.io import write_results

logger = get_logger("root")

@torch.no_grad()
def detect(detector, deepsort, img0):
    
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
        img0 = draw_boxes(img0, bbox_xyxy, identities)
        
        #for bb_xyxy in bbox_xyxy:
        #    bbox_tlwh.append(deepsort._xyxy_to_tlwh(bb_xyxy))
        
        # results.append((idx_frame - 1, bbox_tlwh, identities))



    end = time.time()
    cv2.imshow("test", img0)
    cv2.waitKey(1)

    # save results
    #write_results(self.save_results_path, results, 'mot')

    # logging
    logger.info("time: {:.03f}s, fps: {:.03f}, detection numbers: {}, tracking numbers: {}" \
        .format(end - start, 1 / (end - start), bbox_xywh.shape[0], len(outputs)))
    


def acquire_images(cam_list, master_camera_sn, detector, deepsort):
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
                            detect(detector, deepsort, im_cv2_format)
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


def run_master_camera(cam_list, master_camera_sn, detector, deepsort):
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
        result &= acquire_images(cam_list, master_camera_sn, detector, deepsort)

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
    parser.add_argument("--config_mmdetection", type=str, default="./configs/mmdet.yaml")
    parser.add_argument("--config_detection", type=str, default="./configs/yolov3.yaml")
    parser.add_argument("--config_deepsort", type=str, default="./configs/deep_sort.yaml")
    parser.add_argument("--config_fastreid", type=str, default="./configs/fastreid.yaml")
    parser.add_argument("--fastreid", action="store_true")
    parser.add_argument("--mmdet", action="store_true")

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
    if opt.fastreid:
        cfg.merge_from_file(opt.config_fastreid)
        cfg.USE_FASTREID = True
    else:
        cfg.USE_FASTREID = False

    use_cuda = torch.cuda.is_available()

    detector = build_detector(cfg, use_cuda=use_cuda)
    deepsort = build_tracker(cfg, use_cuda=use_cuda)
    class_names = detector.class_names

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

    result = run_master_camera(cam_list, master_camera_serial_number, detector, deepsort)

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
